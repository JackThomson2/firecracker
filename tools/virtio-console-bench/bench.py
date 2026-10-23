#!/usr/bin/env python3
"""Boot one guest with a given console setup and report boot time, outputs and in-guest results.

usage: bench.py BINARY TRANSPORT(mmio|pci) CONSOLE(ttyS0|hvc0|early|none) FCMODE(boot|tput|log) KERNEL [TAG]
"""
import json, os, re, subprocess, sys, tempfile

fc, transport, console, fcmode, kernel = sys.argv[1:6]
tag = sys.argv[6] if len(sys.argv) > 6 else "run"
W = os.path.dirname(os.path.abspath(__file__))
A = os.environ.get("FC_CI_ARTIFACTS") or next(
    f"build/artifacts/{d}/x86_64" for d in os.listdir("build/artifacts") if "firecracker-ci" in d)
run = tempfile.mkdtemp(prefix=f"{tag.replace('/', '_')}-", dir=f"{W}/runs")

args = ("reboot=k panic=1 nomodule i8042.noaux i8042.nomux i8042.dumbkbd cryptomgr.notests "
        f"printk.devkmsg=on fcmode={fcmode} ")
args += {
    "ttyS0": "console=ttyS0",
    "hvc0": "console=hvc0",
    "early": "earlycon=uart,io,0x3f8 console=hvc0",
    "none": "8250.nr_uarts=0",
}[console]
cfg = {
    "boot-source": {"kernel_image_path": kernel if kernel.startswith("/") else f"{A}/{kernel}", "initrd_path": os.environ.get("INITRAMFS", f"{W}/initramfs.cpio"),
                    "boot_args": args},
    "machine-config": {"vcpu_count": 2, "mem_size_mib": 1024},
    "drives": [],
}
with_device = console in ("hvc0", "early") or (fcmode == "tput" and console != "none")
if with_device:
    cfg["console"] = {"output_path": f"{run}/hvc.out"}
json.dump(cfg, open(f"{run}/cfg.json", "w"))
open(f"{run}/fc.log", "w").close()
cmd = [fc, "--no-api", "--config-file", f"{run}/cfg.json", "--boot-timer",
       "--log-path", f"{run}/fc.log", "--level", "Info"]
if transport == "pci":
    cmd.append("--enable-pci")
with open(f"{run}/serial.out", "wb") as out:
    p = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, timeout=60)

log = open(f"{run}/fc.log").read()
m = re.search(r"Guest-boot-time\s*=\s*(\d+) us", log)
serial = open(f"{run}/serial.out", "rb").read().decode(errors="replace")
hvc = open(f"{run}/hvc.out", "rb").read().decode(errors="replace") if os.path.exists(f"{run}/hvc.out") else ""
res = {"exit": p.returncode, "boot_us": int(m.group(1)) if m else None, "dir": run,
       "serial_bytes": len(serial), "hvc_bytes": len(hvc)}
r = re.search(r"RESULT (.*)", serial + hvc)
if r:
    res.update({k: int(v) for k, v in (kv.split("=") for kv in r.group(1).split())})
print(json.dumps(res))
