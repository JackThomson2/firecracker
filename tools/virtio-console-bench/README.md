# virtio-console prototype benchmark

Compares guest console output through the prototype virtio console (`hvc0`)
with the 8250 serial port (`ttyS0`). Measured in October 2026 on a c7i.metal-24xl
host with the CI 6.1.172 and 5.10.253 guests, 2 vCPUs and 1 GiB of memory.
Output went to a regular file.

## Files

- `init`: the busybox initramfs init. Build the initramfs from a directory that
  holds it as `/init` and a static busybox as `/bin/busybox`. The kernel
  argument `fcmode=` selects what it does:
  - `boot`: signals the boot timer and reboots.
  - `tput`: times 2048 printk records of 100 bytes written through `/dev/kmsg`,
    then 200 KiB written to each of `/dev/ttyS0` and `/dev/hvc0`.
  - `log` and `logs`: dump `dmesg` to the console or to `ttyS0`.
- `bench.py`: boots one guest and prints the boot time and results as JSON.
- `logcheck.py`: compares what the consoles printed with the guest's `dmesg`.
- `matrix.py` and `matrix2.py`: the boot time and throughput matrices.
  Firecracker binaries and kernels are passed in through `PROTO`, `SERIES`,
  `KERNEL_LOCAL` and `KERNEL_FIXED`.
- `kernel-patches/`: the guest-side fix described below, for 6.1 and for
  mainline.

## Guest-side bug

Linux's `virtio_console` loses everything the kernel printed before the
driver probed. With `console=hvc0`, `hvc0` started at `loop: module loaded`,
about 34 ms into boot, and 262 of the 294 log records were missing. The cause
is that `init_port_console()` calls `hvc_alloc()` before it adds the port to
`pdrvdata.consoles`. Registering the console replays the log through
`put_chars()`, which cannot find the port, returns `-EPIPE`, and
`hvc_console_print()` throws the characters away. The code is the same in
6.1.172, 6.18 and 7.0.5.

Moving that insertion alone is not enough. `add_port()` also adds the port to
`portdev->ports` only after `init_port_console()`, so an output queue
interrupt during the replay makes `out_intr()` flush the queue that
`put_chars()` is polling. The driver then reports `output:id 0 is not a head!`
and breaks the queue. This happened in half of the MMIO boots. The patch
moves both insertions earlier. With it, 25 of 25 MMIO and 25 of 25 PCI boots
carried the complete log.

`earlycon=uart,io,0x3f8 console=hvc0` also gives a complete log on unpatched
guests. The serial port carries everything up to
`printk: bootconsole [uart0] disabled`, and `hvc0` continues from there with no
gap. That early part is about 90% of the boot log, though.

## Results

Boot time to init on the 6.1 guest, as the median of 10 boots. The kernel is
the CI configuration built locally, with or without the patch.

| Console | MMIO | PCI | Log printed |
|---|---|---|---|
| none (`8250.nr_uarts=0`) | 81.4 ms | 81.6 ms | none |
| `ttyS0` | 181.4 ms | 190.5 ms | all |
| `hvc0`, unpatched guest | 84.2 ms | 86.5 ms | 32 lines |
| `hvc0`, patched guest | 91.0 ms | 91.8 ms | all |

On the CI 6.1 kernel, `earlycon` with `hvc0` took 172.5 ms on MMIO and
182.3 ms on PCI. `ttyS0` took 178.5 ms and 186.7 ms, and 156.2 ms and
165.4 ms with the serial coalescing series.

Throughput on the 6.1 guest, as the median of 3 runs:

| Path | MMIO | PCI |
|---|---|---|
| printk via `ttyS0` | 2.7k lines/s | 2.7k lines/s |
| printk via `ttyS0`, coalescing series | 6.6k lines/s | 6.6k lines/s |
| printk via `hvc0` | 20.4k lines/s | 24.7k lines/s |
| writes to `/dev/ttyS0` | 249 KiB/s | 243 KiB/s |
| writes to `/dev/ttyS0`, coalescing series | 480 KiB/s | 603 KiB/s |
| writes to `/dev/hvc0` | 2.9 MiB/s | 7.1 MiB/s |

Writes to `/dev/hvc0` are slower on MMIO because every used-buffer interrupt
costs the guest two exits to Firecracker, to read and acknowledge the
interrupt status. The 5.10 guest gives similar numbers, except that printk
via `ttyS0` reaches only 1.3k lines/s, since 5.10 has no FIFO console path.
