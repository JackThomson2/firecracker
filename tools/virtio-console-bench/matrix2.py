#!/usr/bin/env python3
import json, os, statistics, subprocess, sys
W = os.path.dirname(os.path.abspath(__file__))
FC = os.environ["PROTO"]
kernels = {"local": os.environ["KERNEL_LOCAL"], "fixed": os.environ["KERNEL_FIXED"]}
configs = [(k, c) for k in kernels for c in ("none", "ttyS0", "hvc0")]
rows = []
for i in range(10):
    for t in ("mmio", "pci"):
        for k, c in (configs if i % 2 == 0 else configs[::-1]):
            out = subprocess.run([sys.executable, f"{W}/bench.py", FC, t, c, "boot", kernels[k], f"b2-{k}-{t}-{c}"],
                                 capture_output=True, text=True).stdout
            r = json.loads(out.strip().splitlines()[-1]); r.update(kernel=k, transport=t, console=c); rows.append(r)
print("kernel  transport console  boot ms (median of 10, min-max)   console bytes")
for k, c in configs:
    for t in ("mmio", "pci"):
        v = [r for r in rows if (r["kernel"], r["transport"], r["console"]) == (k, t, c)]
        b = [r["boot_us"] / 1000 for r in v]
        print(f"{k:7} {t:9} {c:7} {statistics.median(b):6.1f} ({min(b):.1f}-{max(b):.1f})   {statistics.median(r['serial_bytes'] + r['hvc_bytes'] for r in v):.0f}")
