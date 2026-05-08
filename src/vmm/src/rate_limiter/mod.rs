// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! GCRA-backed token bucket and rate limiter.
//!
//! Each `TokenBucket` keeps a single `AtomicU64` of state representing the
//! Generic Cell Rate Algorithm "theoretical arrival time" (TAT), in
//! nanoseconds since a process-relative epoch. Calls advance TAT by
//! `T = refill_ns / size` ns per consumed token; a call is denied when
//! `new_tat - now > refill_ns` (i.e. the bucket would be drained beyond
//! its capacity).
//!
//! One-time burst is tracked in a separate atomic counter and drained
//! ahead of the GCRA bucket. Over-consumption (`tokens > size`) is
//! short-circuited and reported via `BucketReduction::OverConsumption`.

use std::os::unix::io::{AsRawFd, RawFd};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use std::{fmt, io};

use utils::time::TimerFd;

pub mod persist;

#[derive(Debug, thiserror::Error, displaydoc::Display)]
/// Describes the errors that may occur while handling rate limiter events.
pub enum RateLimiterError {
    /// Rate limiter event handler called without a present timer
    SpuriousRateLimiterEvent,
}

const NANOSEC_IN_ONE_MILLISEC: u64 = 1_000_000;

/// Process-wide reference for ns-since-epoch conversions.
fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// Returns ns since the process epoch, capped at u64.
#[inline]
fn now_ns() -> u64 {
    let d = Instant::now().saturating_duration_since(epoch());
    d.as_secs() * 1_000_000_000 + u64::from(d.subsec_nanos())
}

/// Enum describing the outcomes of a `reduce()` call on a `TokenBucket`.
#[derive(Clone, Debug, PartialEq)]
pub enum BucketReduction {
    /// There are not enough tokens to complete the operation.
    Failure,
    /// A part of the available tokens have been consumed.
    Success,
    /// A number of tokens `inner` times larger than the bucket size have been consumed.
    OverConsumption(f64),
}

/// Description of the available token types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenType {
    /// Token type used for bandwidth limiting.
    Bytes,
    /// Token type used for ops limiting.
    Ops,
}

/// Token bucket built on the Generic Cell Rate Algorithm (GCRA).
///
/// Hot path is lock-free: a single `compare_exchange_weak` against
/// `tat_ns` per call. One-time-burst is tracked in a side atomic.
#[repr(align(64))]
#[derive(Debug)]
pub struct TokenBucket {
    /// Theoretical arrival time, ns since process epoch.
    /// Zero on construction; lazily seeded on first call.
    tat_ns: AtomicU64,
    /// Remaining one-time burst tokens. Drained before the GCRA bucket.
    one_time_burst: AtomicU64,

    /// Bucket capacity in tokens.
    size: u64,
    /// Refill window length in ns. Equal to MAX_DEFICIT.
    refill_time_ns: u64,
    /// ns required to issue one token (T in GCRA terms).
    /// Stored as fixed-point Q32.32 to keep `T*tokens` exact for any
    /// `tokens <= u32::MAX` while preserving sub-ns precision.
    period_per_token_q32: u128,

    /// Initial one-time burst (preserved for `force_replenish` ceiling).
    initial_one_time_burst: u64,
    /// Refill time in ms (preserved for API compatibility).
    refill_time_ms: u64,
}

impl Clone for TokenBucket {
    fn clone(&self) -> Self {
        Self {
            tat_ns: AtomicU64::new(self.tat_ns.load(Ordering::Relaxed)),
            one_time_burst: AtomicU64::new(self.one_time_burst.load(Ordering::Relaxed)),
            size: self.size,
            refill_time_ns: self.refill_time_ns,
            period_per_token_q32: self.period_per_token_q32,
            initial_one_time_burst: self.initial_one_time_burst,
            refill_time_ms: self.refill_time_ms,
        }
    }
}

impl PartialEq for TokenBucket {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size
            && self.refill_time_ms == other.refill_time_ms
            && self.initial_one_time_burst == other.initial_one_time_burst
            && self.tat_ns.load(Ordering::Relaxed) == other.tat_ns.load(Ordering::Relaxed)
            && self.one_time_burst.load(Ordering::Relaxed)
                == other.one_time_burst.load(Ordering::Relaxed)
    }
}

impl Eq for TokenBucket {}

impl TokenBucket {
    /// Creates a `TokenBucket` wrapped in an `Option`.
    ///
    /// `size` is the steady-state capacity. The bucket refills fully in
    /// `complete_refill_time_ms` ms. `one_time_burst` is an initial extra
    /// allowance that drains before the GCRA bucket and is not refilled.
    ///
    /// Returns `None` if `size == 0`, `complete_refill_time_ms == 0`, or
    /// `complete_refill_time_ms * 1_000_000` overflows `u64`.
    pub fn new(size: u64, one_time_burst: u64, complete_refill_time_ms: u64) -> Option<Self> {
        if size == 0 || complete_refill_time_ms == 0 {
            return None;
        }
        let refill_time_ns = complete_refill_time_ms.checked_mul(NANOSEC_IN_ONE_MILLISEC)?;

        // T = refill_time_ns / size, kept in Q32.32 fixed-point.
        // `T * tokens` then fits in u128 for any reasonable token count.
        let period_per_token_q32 = (u128::from(refill_time_ns) << 32) / u128::from(size);

        Some(TokenBucket {
            tat_ns: AtomicU64::new(0),
            one_time_burst: AtomicU64::new(one_time_burst),
            size,
            refill_time_ns,
            period_per_token_q32,
            initial_one_time_burst: one_time_burst,
            refill_time_ms: complete_refill_time_ms,
        })
    }

    /// Attempts to consume `tokens` from the bucket and returns whether the action succeeded.
    pub fn reduce(&self, tokens: u64) -> BucketReduction {
        // Drain one-time burst first.
        let mut tokens = tokens;
        if self.one_time_burst.load(Ordering::Relaxed) > 0 {
            let mut burst = self.one_time_burst.load(Ordering::Relaxed);
            loop {
                if burst == 0 {
                    break;
                }
                if burst >= tokens {
                    match self.one_time_burst.compare_exchange_weak(
                        burst,
                        burst - tokens,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => return BucketReduction::Success,
                        Err(observed) => {
                            burst = observed;
                            continue;
                        }
                    }
                } else {
                    match self.one_time_burst.compare_exchange_weak(
                        burst,
                        0,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            tokens -= burst;
                            break;
                        }
                        Err(observed) => {
                            burst = observed;
                            continue;
                        }
                    }
                }
            }
        }

        // Over-consumption: requests larger than `size` borrow tokens
        // that take longer than `refill_time` to refill. Drain the
        // current budget and report the residual borrow as a ratio of
        // bucket size, matching the pre-GCRA TokenBucket semantics so
        // RateLimiter::consume's timer math is unchanged.
        if tokens > self.size {
            let now = now_ns();
            let residual;
            loop {
                let tat = self.tat_ns.load(Ordering::Relaxed);
                let earliest = tat.max(now);
                // Spendable budget right now = (refill_time - deficit) / T.
                let deficit_ns = earliest - now;
                let free_ns = self.refill_time_ns.saturating_sub(deficit_ns);
                let budget_q0 = (u128::from(free_ns) << 32) / self.period_per_token_q32;
                let budget = u64::try_from(budget_q0).unwrap_or(u64::MAX).min(self.size);
                let new_tat = now.saturating_add(self.refill_time_ns);
                match self.tat_ns.compare_exchange_weak(
                    tat,
                    new_tat,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        residual = tokens.saturating_sub(budget);
                        break;
                    }
                    Err(_) => continue,
                }
            }
            #[allow(clippy::cast_precision_loss)]
            let ratio = residual as f64 / self.size as f64;
            return BucketReduction::OverConsumption(ratio);
        }

        // GCRA hot path: one CAS on tat_ns.
        let now = now_ns();
        let advance_ns = self.advance_for(tokens);
        loop {
            let tat = self.tat_ns.load(Ordering::Relaxed);
            let earliest = tat.max(now);
            let new_tat = earliest.saturating_add(advance_ns);
            if new_tat.saturating_sub(now) > self.refill_time_ns {
                return BucketReduction::Failure;
            }
            match self.tat_ns.compare_exchange_weak(
                tat,
                new_tat,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return BucketReduction::Success,
                Err(_) => continue,
            }
        }
    }

    /// Compute T*tokens (ns) using the precomputed Q32.32 period.
    #[inline]
    fn advance_for(&self, tokens: u64) -> u64 {
        let product_q32 = self.period_per_token_q32 * u128::from(tokens);
        let ns = product_q32 >> 32;
        u64::try_from(ns).unwrap_or(u64::MAX)
    }

    /// Returns ns until the bucket would next admit a single token, or 0
    /// if a token is available now.
    pub fn next_token_wait_ns(&self) -> u64 {
        let now = now_ns();
        let tat = self.tat_ns.load(Ordering::Relaxed);
        // The earliest a single-token call would land (no slack): tat + T.
        // Deny condition is `landing - now > refill_time_ns`; the wait to
        // satisfy that is `landing - now - refill_time_ns`.
        let earliest = tat.max(now);
        let landing = earliest.saturating_add(self.advance_for(1));
        landing
            .saturating_sub(now)
            .saturating_sub(self.refill_time_ns)
    }

    /// "Manually" adds tokens to the bucket. Replenishes the one-time burst
    /// first (up to its initial cap), then walks `tat_ns` backward by
    /// `T*tokens` ns (capped so the bucket cannot exceed full).
    pub fn force_replenish(&self, tokens: u64) {
        if self.one_time_burst.load(Ordering::Relaxed) < self.initial_one_time_burst {
            // Try to add to burst first, capped at initial.
            let mut burst = self.one_time_burst.load(Ordering::Relaxed);
            loop {
                let new = burst
                    .saturating_add(tokens)
                    .min(self.initial_one_time_burst);
                match self.one_time_burst.compare_exchange_weak(
                    burst,
                    new,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return,
                    Err(observed) => burst = observed,
                }
            }
        }

        // Walk tat backward (= add tokens to GCRA bucket).
        let advance_ns = self.advance_for(tokens);
        let now = now_ns();
        let mut tat = self.tat_ns.load(Ordering::Relaxed);
        loop {
            // tat clamped to >= now (an empty bucket sits at `now`).
            let cur = tat.max(now);
            // Cannot go below `now` (would mean bucket > full).
            let new_tat = cur.saturating_sub(advance_ns).max(now);
            match self.tat_ns.compare_exchange_weak(
                tat,
                new_tat,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => tat = observed,
            }
        }
    }

    /// Returns the capacity of the token bucket.
    pub fn capacity(&self) -> u64 {
        self.size
    }

    /// Returns the remaining one time burst budget.
    pub fn one_time_burst(&self) -> u64 {
        self.one_time_burst.load(Ordering::Relaxed)
    }

    /// Returns the time in milliseconds required to to completely fill the bucket.
    pub fn refill_time_ms(&self) -> u64 {
        self.refill_time_ms
    }

    /// Returns the current effective budget (tokens currently spendable
    /// from the GCRA bucket; one-time-burst not included).
    pub fn budget(&self) -> u64 {
        let now = now_ns();
        let tat = self.tat_ns.load(Ordering::Relaxed);
        if tat <= now {
            return self.size;
        }
        let deficit_ns = tat - now;
        if deficit_ns >= self.refill_time_ns {
            return 0;
        }
        // free_ns = refill - deficit; budget = free_ns / T
        let free_ns = self.refill_time_ns - deficit_ns;
        // Inverse of advance_for: tokens = free_ns / T = (free_ns << 32) / period_q32
        let tokens_q0 = (u128::from(free_ns) << 32) / self.period_per_token_q32;
        u64::try_from(tokens_q0).unwrap_or(u64::MAX).min(self.size)
    }

    /// Returns the initially configured one time burst budget.
    pub fn initial_one_time_burst(&self) -> u64 {
        self.initial_one_time_burst
    }
}

/// Enum that describes the type of token bucket update.
#[derive(Debug)]
pub enum BucketUpdate {
    /// No Update - same as before.
    None,
    /// Rate Limiting is disabled on this bucket.
    Disabled,
    /// Rate Limiting enabled with updated bucket.
    Update(TokenBucket),
}

/// Rate Limiter that works on both bandwidth and ops/s limiting.
///
/// Bandwidth (bytes/s) and ops/s limiting can be used at the same time or individually.
///
/// Implementation uses a single timer through TimerFd to refresh either or
/// both token buckets.
///
/// Its internal buckets are 'passively' replenished as they're being used (as
/// part of `consume()` operations).
/// A timer is enabled and used to 'actively' replenish the token buckets when
/// limiting is in effect and `consume()` operations are disabled.
///
/// RateLimiters will generate events on the FDs provided by their `AsRawFd` trait
/// implementation. These events are meant to be consumed by the user of this struct.
/// On each such event, the user must call the `event_handler()` method.
pub struct RateLimiter {
    bandwidth: Option<TokenBucket>,
    ops: Option<TokenBucket>,

    timer_fd: TimerFd,
    // Internal flag that quickly determines timer state.
    timer_active: bool,
}

impl PartialEq for RateLimiter {
    fn eq(&self, other: &RateLimiter) -> bool {
        self.bandwidth == other.bandwidth && self.ops == other.ops
    }
}

impl fmt::Debug for RateLimiter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "RateLimiter {{ bandwidth: {:?}, ops: {:?} }}",
            self.bandwidth, self.ops
        )
    }
}

impl RateLimiter {
    /// Minimum wait the timer will be armed for. Avoids spinning on
    /// near-zero waits when the bucket is just barely over-drawn.
    const MIN_TIMER_WAIT: Duration = Duration::from_micros(100);

    /// Creates a new Rate Limiter that can limit on both bytes/s and ops/s.
    ///
    /// # Arguments
    ///
    /// * `bytes_total_capacity` - the total capacity of the `TokenType::Bytes` token bucket.
    /// * `bytes_one_time_burst` - initial extra credit on top of `bytes_total_capacity`, that does
    ///   not replenish and which can be used for an initial burst of data.
    /// * `bytes_complete_refill_time_ms` - number of milliseconds for the `TokenType::Bytes` token
    ///   bucket to go from zero Bytes to `bytes_total_capacity` Bytes.
    /// * `ops_total_capacity` - the total capacity of the `TokenType::Ops` token bucket.
    /// * `ops_one_time_burst` - initial extra credit on top of `ops_total_capacity`, that does not
    ///   replenish and which can be used for an initial burst of data.
    /// * `ops_complete_refill_time_ms` - number of milliseconds for the `TokenType::Ops` token
    ///   bucket to go from zero Ops to `ops_total_capacity` Ops.
    ///
    /// If either bytes/ops *size* or *refill_time* are **zero**, the limiter
    /// is **disabled** for that respective token type.
    ///
    /// # Errors
    ///
    /// If the timerfd creation fails, an error is returned.
    pub fn new(
        bytes_total_capacity: u64,
        bytes_one_time_burst: u64,
        bytes_complete_refill_time_ms: u64,
        ops_total_capacity: u64,
        ops_one_time_burst: u64,
        ops_complete_refill_time_ms: u64,
    ) -> io::Result<Self> {
        let bytes_token_bucket = TokenBucket::new(
            bytes_total_capacity,
            bytes_one_time_burst,
            bytes_complete_refill_time_ms,
        );

        let ops_token_bucket = TokenBucket::new(
            ops_total_capacity,
            ops_one_time_burst,
            ops_complete_refill_time_ms,
        );

        // We'll need a timer_fd, even if our current config effectively disables rate limiting,
        // because `Self::update_buckets()` might re-enable it later, and we might be
        // seccomp-blocked from creating the timer_fd at that time.
        let timer_fd = TimerFd::new();

        Ok(RateLimiter {
            bandwidth: bytes_token_bucket,
            ops: ops_token_bucket,
            timer_fd,
            timer_active: false,
        })
    }

    fn activate_timer(&mut self, one_shot_duration: Duration) {
        let dur = one_shot_duration.max(Self::MIN_TIMER_WAIT);
        self.timer_fd.arm(dur, None);
        self.timer_active = true;
    }

    /// Attempts to consume tokens and returns whether that is possible.
    ///
    /// If rate limiting is disabled on provided `token_type`, this function will always succeed.
    pub fn consume(&mut self, tokens: u64, token_type: TokenType) -> bool {
        if self.timer_active {
            return false;
        }

        let token_bucket = match token_type {
            TokenType::Bytes => self.bandwidth.as_ref(),
            TokenType::Ops => self.ops.as_ref(),
        };
        if let Some(bucket) = token_bucket {
            match bucket.reduce(tokens) {
                BucketReduction::Failure => {
                    // GCRA gives us the precise wait time until a single
                    // token would be admitted; arm the timer for that.
                    let wait_ns = bucket.next_token_wait_ns();
                    let wait = Duration::from_nanos(wait_ns);
                    self.activate_timer(wait);
                    false
                }
                BucketReduction::Success => true,
                BucketReduction::OverConsumption(ratio) => {
                    // Mirror pre-GCRA semantics: `ratio` is the residual
                    // borrow as a fraction of `size`, and we block for
                    // `ratio * refill_time` to let it refill naturally.
                    let refill_ms = bucket.refill_time_ms();
                    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                    let wait = Duration::from_millis((ratio * refill_ms as f64) as u64);
                    self.activate_timer(wait);
                    true
                }
            }
        } else {
            true
        }
    }

    /// Adds tokens of `token_type` to their respective bucket.
    ///
    /// Can be used to *manually* add tokens to a bucket. Useful for reverting a
    /// `consume()` if needed.
    pub fn manual_replenish(&mut self, tokens: u64, token_type: TokenType) {
        let token_bucket = match token_type {
            TokenType::Bytes => self.bandwidth.as_ref(),
            TokenType::Ops => self.ops.as_ref(),
        };
        if let Some(bucket) = token_bucket {
            bucket.force_replenish(tokens);
        }
    }

    /// Returns whether this rate limiter is blocked.
    ///
    /// The limiter 'blocks' when a `consume()` operation fails because there was not enough
    /// budget for it.
    /// An event will be generated on the exported FD when the limiter 'unblocks'.
    pub fn is_blocked(&self) -> bool {
        self.timer_active
    }

    /// This function needs to be called every time there is an event on the
    /// FD provided by this object's `AsRawFd` trait implementation.
    ///
    /// # Errors
    ///
    /// If the rate limiter is disabled or is not blocked, an error is returned.
    pub fn event_handler(&mut self) -> Result<(), RateLimiterError> {
        match self.timer_fd.read() {
            0 => Err(RateLimiterError::SpuriousRateLimiterEvent),
            _ => {
                self.timer_active = false;
                Ok(())
            }
        }
    }

    /// Updates the parameters of the token buckets associated with this RateLimiter.
    pub fn update_buckets(&mut self, bytes: BucketUpdate, ops: BucketUpdate) {
        match bytes {
            BucketUpdate::Disabled => self.bandwidth = None,
            BucketUpdate::Update(tb) => self.bandwidth = Some(tb),
            BucketUpdate::None => (),
        };
        match ops {
            BucketUpdate::Disabled => self.ops = None,
            BucketUpdate::Update(tb) => self.ops = Some(tb),
            BucketUpdate::None => (),
        };
    }

    /// Returns an immutable view of the inner bandwidth token bucket.
    pub fn bandwidth(&self) -> Option<&TokenBucket> {
        self.bandwidth.as_ref()
    }

    /// Returns an immutable view of the inner ops token bucket.
    pub fn ops(&self) -> Option<&TokenBucket> {
        self.ops.as_ref()
    }
}

impl AsRawFd for RateLimiter {
    /// Provides a FD which needs to be monitored for POLLIN events.
    ///
    /// This object's `event_handler()` method must be called on such events.
    ///
    /// Will return a negative value if rate limiting is disabled on both
    /// token types.
    fn as_raw_fd(&self) -> RawFd {
        self.timer_fd.as_raw_fd()
    }
}

impl Default for RateLimiter {
    /// Default RateLimiter is a no-op limiter with infinite budget.
    fn default() -> Self {
        // Safe to unwrap since this will not attempt to create timer_fd.
        RateLimiter::new(0, 0, 0, 0, 0, 0).expect("Failed to build default RateLimiter")
    }
}

// TODO(gcra-port): Kani proofs were tied to the old (budget, last_update)
// representation. They need to be reworked against the GCRA invariants:
//   - tat_ns is monotonically non-decreasing across reduce() calls
//   - after a successful reduce(n) at time `now`, the deficit
//     (tat_ns - now) is in [0, refill_time_ns]
//   - one_time_burst is monotonically non-increasing across reduce()
// See parent commit for the original proofs.

#[cfg(test)]
pub(crate) mod tests {
    use std::thread;
    use std::time::Duration;

    use super::*;

    // Slightly larger than the longest natural refill so refill-driven
    // tests can wait in two halves without racing the timer.
    const TEST_REFILL_TIMER_DURATION: Duration = Duration::from_millis(110);

    impl TokenBucket {
        /// Reset bucket to "freshly constructed" state.
        fn reset(&self) {
            self.tat_ns.store(0, Ordering::Relaxed);
            self.one_time_burst
                .store(self.initial_one_time_burst, Ordering::Relaxed);
        }

        // After a restore, we cannot be certain that internal timing is identical.
        pub(crate) fn partial_eq(&self, other: &TokenBucket) -> bool {
            (other.capacity() == self.capacity())
                && (other.one_time_burst() == self.one_time_burst())
                && (other.refill_time_ms() == self.refill_time_ms())
                && (other.budget() == self.budget())
        }
    }

    impl RateLimiter {
        fn get_token_bucket(&self, token_type: TokenType) -> Option<&TokenBucket> {
            match token_type {
                TokenType::Bytes => self.bandwidth.as_ref(),
                TokenType::Ops => self.ops.as_ref(),
            }
        }
    }

    #[test]
    fn test_token_bucket_create() {
        let tb = TokenBucket::new(1000, 0, 1000).unwrap();
        assert_eq!(tb.capacity(), 1000);
        assert_eq!(tb.budget(), 1000);

        // Verify invalid bucket configurations result in `None`.
        assert!(TokenBucket::new(0, 1234, 1000).is_none());
        assert!(TokenBucket::new(100, 1234, 0).is_none());
        assert!(TokenBucket::new(0, 1234, 0).is_none());
    }

    #[test]
    fn test_token_bucket_reduce_basic() {
        let capacity = 1000;
        let refill_ms = 1000;
        let tb = TokenBucket::new(capacity, 0, refill_ms).unwrap();

        assert_eq!(tb.reduce(123), BucketReduction::Success);
        // GCRA budget is the rounded-down spendable token count;
        // permit a small fuzz to account for ns-level slop.
        let b = tb.budget();
        assert!(b <= capacity - 123 && b + 5 >= capacity - 123, "budget={b}");
        assert_eq!(tb.reduce(capacity), BucketReduction::Failure);
    }

    #[test]
    fn test_token_bucket_one_time_burst() {
        let tb = TokenBucket::new(1000, 1100, 1000).unwrap();
        assert_eq!(tb.reduce(1000), BucketReduction::Success);
        assert_eq!(tb.one_time_burst(), 100);
        assert_eq!(tb.reduce(500), BucketReduction::Success);
        assert_eq!(tb.one_time_burst(), 0);
        assert_eq!(tb.reduce(500), BucketReduction::Success);
        assert_eq!(tb.reduce(500), BucketReduction::Failure);
        thread::sleep(Duration::from_millis(550));
        assert_eq!(tb.reduce(500), BucketReduction::Success);
        thread::sleep(Duration::from_millis(1100));
        assert_eq!(tb.reduce(2500), BucketReduction::OverConsumption(1.5));

        tb.reset();
        assert_eq!(tb.capacity(), 1000);
        assert_eq!(tb.budget(), 1000);
    }

    #[test]
    fn test_token_bucket_refills_over_time() {
        const SIZE: u64 = 10;
        const TIME_MS: u64 = 1000;
        let tb = TokenBucket::new(SIZE, 0, TIME_MS).unwrap();

        // Drain.
        assert_eq!(tb.reduce(SIZE), BucketReduction::Success);
        assert_eq!(tb.budget(), 0);

        // After half the refill window, expect ~half the bucket back.
        thread::sleep(Duration::from_millis(550));
        let b = tb.budget();
        assert!((4..=6).contains(&b), "budget after half refill = {b}");

        // After the full window, fully replenished.
        thread::sleep(Duration::from_millis(550));
        assert_eq!(tb.budget(), SIZE);
    }

    #[test]
    fn test_rate_limiter_default() {
        let mut l = RateLimiter::default();
        assert!(!l.is_blocked());
        assert!(l.consume(u64::MAX, TokenType::Ops));
        assert!(l.consume(u64::MAX, TokenType::Bytes));
        let err = l.event_handler().unwrap_err();
        assert!(matches!(err, RateLimiterError::SpuriousRateLimiterEvent));
    }

    #[test]
    fn test_rate_limiter_new() {
        let l = RateLimiter::new(1000, 1001, 1002, 1003, 1004, 1005).unwrap();

        let bw = l.bandwidth.as_ref().unwrap();
        assert_eq!(bw.capacity(), 1000);
        assert_eq!(bw.one_time_burst(), 1001);
        assert_eq!(bw.refill_time_ms(), 1002);
        assert_eq!(bw.budget(), 1000);

        let ops = l.ops.as_ref().unwrap();
        assert_eq!(ops.capacity(), 1003);
        assert_eq!(ops.one_time_burst(), 1004);
        assert_eq!(ops.refill_time_ms(), 1005);
        assert_eq!(ops.budget(), 1003);
    }

    #[test]
    fn test_rate_limiter_manual_replenish() {
        let mut l = RateLimiter::new(1000, 0, 1000, 1000, 0, 1000).unwrap();

        assert!(l.consume(123, TokenType::Bytes));
        l.manual_replenish(23, TokenType::Bytes);
        let b = l.get_token_bucket(TokenType::Bytes).unwrap().budget();
        assert!((895..=905).contains(&b), "budget={b}");

        assert!(l.consume(123, TokenType::Ops));
        l.manual_replenish(23, TokenType::Ops);
        let b = l.get_token_bucket(TokenType::Ops).unwrap().budget();
        assert!((895..=905).contains(&b), "budget={b}");
    }

    #[test]
    fn test_rate_limiter_bandwidth() {
        let mut l = RateLimiter::new(1000, 0, 1000, 0, 0, 0).unwrap();
        assert!(!l.is_blocked());
        assert!(l.as_raw_fd() > 0);
        assert!(l.consume(u64::MAX, TokenType::Ops));

        assert!(l.consume(1000, TokenType::Bytes));
        assert!(!l.consume(100, TokenType::Bytes));
        assert!(l.is_blocked());
        thread::sleep(TEST_REFILL_TIMER_DURATION / 2);
        assert!(l.is_blocked());
        thread::sleep(TEST_REFILL_TIMER_DURATION);
        l.event_handler().unwrap();
        assert!(!l.is_blocked());
        assert!(l.consume(100, TokenType::Bytes));
    }

    #[test]
    fn test_rate_limiter_ops() {
        let mut l = RateLimiter::new(0, 0, 0, 1000, 0, 1000).unwrap();
        assert!(!l.is_blocked());
        assert!(l.as_raw_fd() > 0);
        assert!(l.consume(u64::MAX, TokenType::Bytes));

        assert!(l.consume(1000, TokenType::Ops));
        assert!(!l.consume(100, TokenType::Ops));
        assert!(l.is_blocked());
        thread::sleep(TEST_REFILL_TIMER_DURATION / 2);
        assert!(l.is_blocked());
        thread::sleep(TEST_REFILL_TIMER_DURATION);
        l.event_handler().unwrap();
        assert!(!l.is_blocked());
        assert!(l.consume(100, TokenType::Ops));
    }

    #[test]
    fn test_rate_limiter_full() {
        let mut l = RateLimiter::new(1000, 0, 1000, 1000, 0, 1000).unwrap();
        assert!(!l.is_blocked());
        assert!(l.as_raw_fd() > 0);

        assert!(l.consume(1000, TokenType::Ops));
        assert!(l.consume(1000, TokenType::Bytes));
        assert!(!l.consume(100, TokenType::Ops));
        assert!(!l.consume(100, TokenType::Bytes));
        assert!(l.is_blocked());
        thread::sleep(TEST_REFILL_TIMER_DURATION / 2);
        assert!(l.is_blocked());
        thread::sleep(TEST_REFILL_TIMER_DURATION);
        l.event_handler().unwrap();
        assert!(!l.is_blocked());
        assert!(l.consume(100, TokenType::Ops));
        assert!(l.consume(100, TokenType::Bytes));
    }

    #[test]
    fn test_rate_limiter_overconsumption() {
        let mut l = RateLimiter::new(1000, 0, 1000, 1000, 0, 1000).unwrap();
        // 2.5x bucket. Bucket starts full so residual = 2500 - 1000 = 1500
        // tokens borrowed; ratio = 1.5; timer arms for 1.5 * refill = 1500ms.
        assert!(l.consume(2500, TokenType::Bytes));

        thread::sleep(Duration::from_millis(1000));
        l.event_handler().unwrap_err();
        assert!(l.is_blocked());

        thread::sleep(Duration::from_millis(700));
        l.event_handler().unwrap();
        assert!(!l.is_blocked());

        // 1.5x bucket. Residual = 500; ratio = 0.5; timer = 500ms.
        let mut l = RateLimiter::new(1000, 0, 1000, 1000, 0, 1000).unwrap();
        assert!(l.consume(1500, TokenType::Bytes));

        thread::sleep(Duration::from_millis(200));
        l.event_handler().unwrap_err();
        assert!(l.is_blocked());

        assert!(!l.consume(100, TokenType::Bytes));
        l.event_handler().unwrap_err();
        assert!(l.is_blocked());

        thread::sleep(Duration::from_millis(500));
        l.event_handler().unwrap();
        assert!(!l.is_blocked());
        assert!(l.consume(100, TokenType::Bytes));
    }

    #[test]
    fn test_update_buckets() {
        let mut x = RateLimiter::new(1000, 2000, 1000, 10, 20, 1000).unwrap();

        let initial_bw = x.bandwidth.clone();
        let initial_ops = x.ops.clone();

        x.update_buckets(BucketUpdate::None, BucketUpdate::None);
        assert!(x.bandwidth.as_ref().unwrap().partial_eq(initial_bw.as_ref().unwrap()));
        assert!(x.ops.as_ref().unwrap().partial_eq(initial_ops.as_ref().unwrap()));

        let new_bw = TokenBucket::new(123, 0, 57).unwrap();
        let new_ops = TokenBucket::new(321, 12346, 89).unwrap();
        x.update_buckets(
            BucketUpdate::Update(new_bw.clone()),
            BucketUpdate::Update(new_ops.clone()),
        );

        assert!(x.bandwidth.as_ref().unwrap().partial_eq(&new_bw));
        assert!(x.ops.as_ref().unwrap().partial_eq(&new_ops));

        x.update_buckets(BucketUpdate::Disabled, BucketUpdate::Disabled);
        assert!(x.bandwidth.is_none());
        assert!(x.ops.is_none());
    }

    #[test]
    fn test_rate_limiter_debug() {
        let l = RateLimiter::new(1, 2, 3, 4, 5, 6).unwrap();
        assert_eq!(
            format!("{:?}", l),
            format!(
                "RateLimiter {{ bandwidth: {:?}, ops: {:?} }}",
                l.bandwidth(),
                l.ops()
            ),
        );
    }
}
