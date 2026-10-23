#!/usr/bin/env python3
"""Compare what each output carried with the guest's dmesg, which the guest dumps at the end."""
import json, re, subprocess, sys

fc, transport, console = sys.argv[1:4]
kernel = sys.argv[4] if len(sys.argv) > 4 else "vmlinux-6.1.172"
out = subprocess.run([sys.executable, __file__.replace("logcheck.py", "bench.py"), fc, transport,
                      console, "log", kernel, f"log-{transport}-{console}"],
                     capture_output=True, text=True).stdout
res = json.loads(out.strip().splitlines()[-1])
d = res["dir"]
serial = open(f"{d}/serial.out", "rb").read().decode(errors="replace").replace("\r", "")
try:
    hvc = open(f"{d}/hvc.out", "rb").read().decode(errors="replace").replace("\r", "")
except FileNotFoundError:
    hvc = ""
both = serial + hvc
dump = both[both.index("DMESG-BEGIN"):both.index("DMESG-END")]
records = [re.sub(r"^<\d+>", "", l) for l in dump.splitlines()[1:] if l.strip()]
ts = re.compile(r"^\[\s*\d+\.\d+\] ")
before = both[:both.index("DMESG-BEGIN")]
con_lines = set(l for l in before.splitlines() if ts.match(l))
missing = [r for r in records if ts.match(r) and r not in con_lines]
ser_lines = [l for l in serial.splitlines() if ts.match(l)]
hvc_lines = [l for l in hvc[:hvc.find("DMESG-BEGIN") if "DMESG-BEGIN" in hvc else len(hvc)].splitlines() if ts.match(l)]
print(f"== {transport} {console}: dmesg has {len(records)} records; serial printed {len(ser_lines)}, hvc0 printed {len(hvc_lines)}; missing from the consoles: {len(missing)}")
if ser_lines:
    print("   serial: first", ser_lines[0][:70], "| last", ser_lines[-1][:70])
if hvc_lines:
    print("   hvc0:   first", hvc_lines[0][:70], "| last", hvc_lines[-1][:70])
for m in missing[:4]:
    print("   missing:", m[:100])
if len(missing) > 4:
    print(f"   ... missing up to: {missing[-1][:100]}")
