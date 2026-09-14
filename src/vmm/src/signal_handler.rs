// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt::{self, Write as _};
use std::os::fd::AsRawFd;

use libc::{
    SIGBUS, SIGHUP, SIGILL, SIGPIPE, SIGSEGV, SIGSYS, SIGXCPU, SIGXFSZ, c_int, c_void, siginfo_t,
};

use crate::FcExitCode;
use crate::logger::{DEFAULT_INSTANCE_ID, INSTANCE_ID, IncMetric, LOGGER, METRICS, StoreMetric};
use crate::utils::signal::register_signal_handler;

// The offset of `si_syscall` (offending syscall identifier) within the siginfo structure
// expressed as an `(u)int*`.
// Offset `6` for an `i32` field means that the needed information is located at `6 * sizeof(i32)`.
// See /usr/include/linux/signal.h for the C struct definition.
// See https://github.com/rust-lang/libc/issues/716 for why the offset is different in Rust.
const SI_OFF_SYSCALL: isize = 6;

const SYS_SECCOMP_CODE: i32 = 1;

/// Capacity of the stack buffer one emergency log line is formatted into.
const EMERGENCY_LINE_CAPACITY: usize = 512;

/// Fixed-capacity, stack-allocated `fmt::Write` sink for a single log line.
///
/// Output beyond the capacity is dropped; [`StackLine::finish`] makes sure the line still ends
/// with a newline.
struct StackLine {
    buf: [u8; EMERGENCY_LINE_CAPACITY],
    len: usize,
}

impl StackLine {
    const fn new() -> Self {
        Self {
            buf: [0; EMERGENCY_LINE_CAPACITY],
            len: 0,
        }
    }

    /// Terminates the line with a newline, replacing the last byte when the buffer is full.
    fn finish(&mut self) -> &[u8] {
        if self.len == EMERGENCY_LINE_CAPACITY {
            self.len -= 1;
        }
        self.buf[self.len] = b'\n';
        self.len += 1;
        &self.buf[..self.len]
    }
}

impl fmt::Write for StackLine {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let n = s.len().min(EMERGENCY_LINE_CAPACITY - self.len);
        self.buf[self.len..self.len + n].copy_from_slice(&s.as_bytes()[..n]);
        self.len += n;
        Ok(())
    }
}

/// Converts days since 1970-01-01 to a proleptic Gregorian `(year, month, day)`.
///
/// This is Howard Hinnant's `civil_from_days`.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// Writes a `CLOCK_REALTIME` timestamp in the `%Y-%m-%dT%H:%M:%S.%N` layout used by the logger.
///
/// The time is broken down as UTC: `localtime_r` is not async-signal-safe (musl serialises it on
/// an internal lock). Firecracker normally runs in a jail without a timezone configuration,
/// where local time is UTC anyway.
fn write_utc_timestamp(out: &mut impl fmt::Write) -> fmt::Result {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable timespec and CLOCK_REALTIME always exists.
    unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &mut ts) };
    write_timestamp(out, ts.tv_sec, ts.tv_nsec)
}

fn write_timestamp(out: &mut impl fmt::Write, secs: i64, nsecs: i64) -> fmt::Result {
    let (year, month, day) = civil_from_days(secs.div_euclid(86_400));
    let secs = secs.rem_euclid(86_400);
    write!(
        out,
        "{year}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{nsecs:09}",
        secs / 3_600,
        secs / 60 % 60,
        secs % 60,
    )
}

/// Formats a log line with the same layout as [`crate::logger::LOGGER`] produces:
/// `TIMESTAMP [ID:THREAD(:ERROR)(:FILE:LINE)] MESSAGE`.
fn format_emergency_line(
    out: &mut StackLine,
    thread: &str,
    show_level: bool,
    origin: Option<(&str, u32)>,
    args: fmt::Arguments<'_>,
) {
    // A `fmt::Error` cannot come out of a StackLine: it truncates instead.
    let _ = write_utc_timestamp(out);
    let _ = write!(
        out,
        " [{}:{thread}",
        INSTANCE_ID
            .get()
            .map_or(DEFAULT_INSTANCE_ID, String::as_str),
    );
    if show_level {
        let _ = out.write_str(":ERROR");
    }
    if let Some((file, line)) = origin {
        let _ = write!(out, ":{file}:{line}");
    }
    let _ = write!(out, "] {args}");
}

/// Writes `bytes` to `fd` with raw `write(2)` calls, retrying partial writes and `EINTR`.
///
/// Any other failure only bumps the missed log metric: the process is about to exit and there is
/// nowhere else to report it.
fn write_all_raw(fd: c_int, mut bytes: &[u8]) {
    while !bytes.is_empty() {
        // SAFETY: `bytes` is a valid, initialized slice that outlives the call.
        let written = unsafe { libc::write(fd, bytes.as_ptr().cast(), bytes.len()) };
        match usize::try_from(written) {
            Ok(n) if n > 0 => bytes = &bytes[n..],
            _ if std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) => {}
            _ => {
                METRICS.logger.missed_log_count.inc();
                return;
            }
        }
    }
}

/// Writes one error line to the log target using only async-signal-safe operations.
///
/// The line has the layout of a regular log line but is formatted into a stack buffer and
/// written with a raw `write(2)`: no heap allocation, no blocking lock and no `localtime_r`.
/// The handlers below run on whichever thread raised the signal, and a seccomp trap raised by the
/// allocator's own `mmap`, `mprotect` or `brk` leaves that thread holding musl's malloc lock, so
/// going through the regular logger (or anything else that allocates) would deadlock the process
/// instead of letting it exit.
///
/// `file` and `line` are reported as the log origin when the logger is configured to show it.
/// This is public so that the crate's `signal_safety` integration test can assert, with a
/// counting global allocator, that the path really does not allocate.
pub fn emergency_log(file: &str, line: u32, args: fmt::Arguments<'_>) {
    // `try_read` never blocks. The write lock is only taken by a pre-boot logger update; fall
    // back to the default target while it is held.
    let config = LOGGER.0.try_read().ok();
    let (fd, show_level, show_origin) = match config.as_deref() {
        Some(config) => (
            config
                .target
                .as_ref()
                .map_or(libc::STDOUT_FILENO, AsRawFd::as_raw_fd),
            config.format.show_level,
            config.format.show_log_origin,
        ),
        None => (libc::STDOUT_FILENO, false, false),
    };

    // The handle of the current thread is created when the thread is spawned (and during runtime
    // initialization for the main thread), so this only bumps a refcount; see also the pre-warm in
    // `register_signal_handlers`.
    let thread = std::thread::current();
    let mut out = StackLine::new();
    format_emergency_line(
        &mut out,
        thread.name().unwrap_or("-"),
        show_level,
        show_origin.then_some((file, line)),
        args,
    );
    write_all_raw(fd, out.finish());
}

/// Logs an error line through [`emergency_log`], recording the call site as the log origin.
macro_rules! emergency_log {
    ($($arg:tt)+) => {
        emergency_log(file!(), line!(), format_args!($($arg)+))
    };
}

/// Exits the process, first flushing the metrics unless the interrupted thread may be inside the
/// allocator (`flush_metrics == false`), since serialising the metrics allocates.
#[inline]
fn exit_with_code(exit_code: FcExitCode, flush_metrics: bool) {
    if flush_metrics && let Err(err) = METRICS.write() {
        emergency_log!("Failed to write metrics while stopping: {}", err);
    }
    // SAFETY: Safe because we're terminating the process anyway.
    unsafe { libc::_exit(exit_code as i32) };
}

macro_rules! generate_handler {
    ($fn_name:ident ,$signal_name:ident, $exit_code:ident, $signal_metric:expr, $body:ident) => {
        #[inline(always)]
        extern "C" fn $fn_name(num: c_int, info: *mut siginfo_t, _unused: *mut c_void) {
            // SAFETY: Safe because we're just reading some fields from a supposedly valid argument.
            let si_signo = unsafe { (*info).si_signo };
            // SAFETY: Safe because we're just reading some fields from a supposedly valid argument.
            let si_code = unsafe { (*info).si_code };

            if num != si_signo || num != $signal_name {
                exit_with_code(FcExitCode::UnexpectedError, true);
            }
            $signal_metric.store(1);

            emergency_log!(
                "Shutting down VM after intercepting signal {}, code {}.",
                si_signo,
                si_code
            );

            let flush_metrics = $body(si_code, info);

            match si_signo {
                $signal_name => exit_with_code(crate::FcExitCode::$exit_code, flush_metrics),
                _ => exit_with_code(FcExitCode::UnexpectedError, flush_metrics),
            };
        }
    };
}

/// Logs the syscall that tripped the seccomp filter.
///
/// Returns `false` because the metrics must not be flushed: the trap was raised synchronously by
/// whatever syscall the interrupted code was making, and when that is one of the allocator's own
/// (`mmap`, `mprotect`, `brk`, `munmap`, ...) the thread holds musl's malloc lock, so the
/// allocating serialisation would deadlock and the process would hang instead of exiting.
fn log_sigsys_err(si_code: c_int, info: *mut siginfo_t) -> bool {
    if si_code != SYS_SECCOMP_CODE {
        // We received a SIGSYS for a reason other than `bad syscall`.
        exit_with_code(FcExitCode::UnexpectedError, true);
    }

    // SAFETY: Other signals which might do async unsafe things incompatible with the rest of this
    // function are blocked due to the sa_mask used when registering the signal handler.
    let syscall = unsafe { *(info as *const i32).offset(SI_OFF_SYSCALL) };
    emergency_log!(
        "Shutting down VM after intercepting a bad syscall ({}).",
        syscall
    );
    false
}

/// The metrics can be flushed before exiting.
fn empty_fn(_si_code: c_int, _info: *mut siginfo_t) -> bool {
    true
}

generate_handler!(
    sigxfsz_handler,
    SIGXFSZ,
    SIGXFSZ,
    METRICS.signals.sigxfsz,
    empty_fn
);

generate_handler!(
    sigxcpu_handler,
    SIGXCPU,
    SIGXCPU,
    METRICS.signals.sigxcpu,
    empty_fn
);

generate_handler!(
    sigbus_handler,
    SIGBUS,
    SIGBUS,
    METRICS.signals.sigbus,
    empty_fn
);

generate_handler!(
    sigsegv_handler,
    SIGSEGV,
    SIGSEGV,
    METRICS.signals.sigsegv,
    empty_fn
);

generate_handler!(
    sigsys_handler,
    SIGSYS,
    BadSyscall,
    METRICS.seccomp.num_faults,
    log_sigsys_err
);

generate_handler!(
    sighup_handler,
    SIGHUP,
    SIGHUP,
    METRICS.signals.sighup,
    empty_fn
);
generate_handler!(
    sigill_handler,
    SIGILL,
    SIGILL,
    METRICS.signals.sigill,
    empty_fn
);

#[inline(always)]
extern "C" fn sigpipe_handler(num: c_int, info: *mut siginfo_t, _unused: *mut c_void) {
    // Just record the metric and allow the process to continue, the EPIPE error needs
    // to be handled at caller level.

    // SAFETY: Safe because we're just reading some fields from a supposedly valid argument.
    let si_signo = unsafe { (*info).si_signo };
    // SAFETY: Safe because we're just reading some fields from a supposedly valid argument.
    let si_code = unsafe { (*info).si_code };

    if num != si_signo || num != SIGPIPE {
        emergency_log!("Received invalid signal {}, code {}.", si_signo, si_code);
        return;
    }

    // Do not log here: the write that raised SIGPIPE is usually a log write, so
    // logging would re-enter the logger mid-write on this same thread.
    METRICS.signals.sigpipe.inc();
}

/// Registers all the required signal handlers.
///
/// Custom handlers are installed for: `SIGBUS`, `SIGSEGV`, `SIGSYS`
/// `SIGXFSZ` `SIGXCPU` `SIGPIPE` `SIGHUP` and `SIGILL`.
pub fn register_signal_handlers() -> vmm_sys_util::errno::Result<()> {
    // The handlers read the thread name through `std::thread::current()`. For threads created by
    // `std::thread::spawn` the handle is set before the thread runs any user code, but on the main
    // thread the standard library is free to create it lazily on the first call, and that would be
    // an allocation. Make this the first call, so no handler can be.
    let _ = std::thread::current();

    // Call to unsafe register_signal_handler which is considered unsafe because it will
    // register a signal handler which will be called in the current thread and will interrupt
    // whatever work is done on the current thread, so we have to keep in mind that the registered
    // signal handler must only do async-signal-safe operations.
    register_signal_handler(SIGSYS, sigsys_handler)?;
    register_signal_handler(SIGBUS, sigbus_handler)?;
    register_signal_handler(SIGSEGV, sigsegv_handler)?;
    register_signal_handler(SIGXFSZ, sigxfsz_handler)?;
    register_signal_handler(SIGXCPU, sigxcpu_handler)?;
    register_signal_handler(SIGPIPE, sigpipe_handler)?;
    register_signal_handler(SIGHUP, sighup_handler)?;
    register_signal_handler(SIGILL, sigill_handler)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_str(out: &mut StackLine) -> String {
        String::from_utf8(out.finish().to_vec()).unwrap()
    }

    #[test]
    fn test_civil_from_days() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
        assert_eq!(civil_from_days(10_957), (2000, 1, 1));
        assert_eq!(civil_from_days(11_016), (2000, 2, 29));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
        // 2100 is not a leap year.
        assert_eq!(civil_from_days(47_540), (2100, 2, 28));
        assert_eq!(civil_from_days(47_541), (2100, 3, 1));
    }

    #[test]
    fn test_write_timestamp() {
        let mut out = StackLine::new();
        write_timestamp(&mut out, 0, 0).unwrap();
        assert_eq!(line_str(&mut out), "1970-01-01T00:00:00.000000000\n");

        let mut out = StackLine::new();
        // 2024-03-05T07:08:09.000000123
        write_timestamp(&mut out, 1_709_622_489, 123).unwrap();
        assert_eq!(line_str(&mut out), "2024-03-05T07:08:09.000000123\n");

        let mut out = StackLine::new();
        write_utc_timestamp(&mut out).unwrap();
        let now = line_str(&mut out);
        assert_eq!(now.len(), 30);
        assert_eq!(&now[4..5], "-");
        assert_eq!(&now[7..8], "-");
        assert_eq!(&now[10..11], "T");
        assert_eq!(&now[19..20], ".");
        assert!(now[..4].parse::<u32>().unwrap() >= 2024);
    }

    #[test]
    fn test_stack_line_truncates_and_terminates() {
        let mut out = StackLine::new();
        for _ in 0..EMERGENCY_LINE_CAPACITY {
            out.write_str("ab").unwrap();
        }
        let line = out.finish();
        assert_eq!(line.len(), EMERGENCY_LINE_CAPACITY);
        assert_eq!(line[EMERGENCY_LINE_CAPACITY - 1], b'\n');
        assert!(
            line[..EMERGENCY_LINE_CAPACITY - 1]
                .iter()
                .all(|b| *b == b'a' || *b == b'b')
        );
    }

    #[test]
    fn test_format_emergency_line() {
        let mut out = StackLine::new();
        format_emergency_line(
            &mut out,
            "fc_vcpu 3",
            false,
            None,
            format_args!("bad syscall ({}).", 10),
        );
        let line = line_str(&mut out);
        let (timestamp, rest) = line.split_once(' ').unwrap();
        assert_eq!(timestamp.len(), 29);
        assert_eq!(
            rest,
            format!("[{DEFAULT_INSTANCE_ID}:fc_vcpu 3] bad syscall (10).\n")
        );

        let mut out = StackLine::new();
        format_emergency_line(
            &mut out,
            "main",
            true,
            Some(("src/vmm/src/signal_handler.rs", 42)),
            format_args!("signal {}", 31),
        );
        let line = line_str(&mut out);
        let (_, rest) = line.split_once(' ').unwrap();
        assert_eq!(
            rest,
            format!(
                "[{DEFAULT_INSTANCE_ID}:main:ERROR:src/vmm/src/signal_handler.rs:42] signal 31\n"
            )
        );
    }
}
