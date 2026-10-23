#!/usr/bin/env python3
"""Run the boot-time and throughput matrix, interleaving configurations, and print medians."""
import json, os, statistics, subprocess, sys

W = os.path.dirname(os.path.abspath(__file__))
PROTO = os.environ["PROTO"]
SERIES = os.environ["SERIES"]
kernel = sys.argv[1] if len(sys.argv) > 1 else "vmlinux-6.1.172"
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
results = f"{W}/results-{kernel}.jsonl"

def run(binary, name, transport, console, mode):
    out = subprocess.run([sys.executable, f"{W}/bench.py", binary, transport, console, mode, kernel,
                          f"{mode}-{name}-{transport}-{console}"], capture_output=True, text=True)
    r = json.loads(out.stdout.strip().splitlines()[-1])
    r.update(binary=name, transport=transport, console=console, mode=mode, kernel=kernel)
    with open(results, "a") as f:
        f.write(json.dumps(r) + "\n")
    return r

boot = [("main", PROTO, c) for c in ("none", "ttyS0", "hvc0", "early")] + [("series", SERIES, "ttyS0")]
tput = [("main", PROTO, "ttyS0"), ("main", PROTO, "hvc0"), ("series", SERIES, "ttyS0")]
for i in range(reps):
    for t in ("mmio", "pci"):
        order = boot if i % 2 == 0 else boot[::-1]
        for name, b, c in order:
            run(b, name, t, c, "boot")
        if i < 3:
            for name, b, c in (tput if i % 2 == 0 else tput[::-1]):
                run(b, name, t, c, "tput")

rows = [json.loads(l) for l in open(results)]
print(f"{kernel}: medians over {reps} boots / 3 throughput runs; 2 vCPUs, output to a file")
print(f"{'binary':7} {'transport':5} {'console':6} {'boot ms':>8}")
for name, _, c in boot:
    for t in ("mmio", "pci"):
        v = [r["boot_us"] for r in rows if (r["binary"], r["transport"], r["console"], r["mode"]) == (name, t, c, "boot") and r["boot_us"]]
        print(f"{name:7} {t:5} {c:6} {statistics.median(v) / 1000:8.1f}  (n={len(v)}, min {min(v) / 1000:.1f}, max {max(v) / 1000:.1f})")
print(f"\n{'binary':7} {'transport':5} {'console':6} {'printk lines/s':>15} {'ttyS0 KiB/s':>12} {'hvc0 KiB/s':>11}")
for name, _, c in tput:
    for t in ("mmio", "pci"):
        v = [r for r in rows if (r["binary"], r["transport"], r["console"], r["mode"]) == (name, t, c, "tput")]
        k = statistics.median(2048 / (r["kmsg_ns"] / 1e9) for r in v)
        s = statistics.median(200 / (r["ttyS0_ns"] / 1e9) for r in v if "ttyS0_ns" in r)
        h = [200 / (r["hvc0_ns"] / 1e9) for r in v if "hvc0_ns" in r]
        print(f"{name:7} {t:5} {c:6} {k:15.0f} {s:12.0f} {statistics.median(h) if h else float('nan'):11.0f}")
