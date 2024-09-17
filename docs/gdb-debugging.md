# GDB Debugging with Firecracker

Firecracker supports debugging the guest kernel via GDB remote serial protocol.
This allows us to connect gdb to the firecracker process and step through debug
the guest kernel.

## Prerequiesites

Firstly to enable GDB debugging we need to compile Firecracker with the `debug`
feature enabled, this will enable the necessary components for the debugging
process.

To build firecracker with the `debug` feature enabled we run:

```bash
cargo build --features "gdb"
```

Secondly we need to compile a kernel with specific features enabled for
debugging to work, the key features to enable on the kernel are:

```
CONFIG_FRAME_POINTER=y
CONFIG_KGDB=y
CONFIG_KGDB_SERIAL_CONSOLE=y
CONFIG_DEBUG_INFO
```

It can also be worth disabling the multi core scheduler as this can cause issues
with GDB these are set under these configs and can be disabled if you encounter
issues

```
CONFIG_SCHED_MC=y
CONFIG_SCHED_MC_PRIO=y
```

## Starting Firecracker with GDB

With all the prerequiesites in place you can now start firecracker ready to
connect to GDB. When you start the firecracker binary now you'll notice it'll be
blocked waiting for the GDB connection this is done to allow us to set
breakpoints before the boot process begins.

With Firecracker running and waiting for GDB we are now able to start GDB and
connect to Firecracker. You may need to set the permissions of `/tmp/gdb.socket`
to `777` before connecting.

An example of the steps taken to start GDB, load the symbols and connect to
Firecracker.

1. Start the GDB process, you can attach the symbols by appending the kernel
   blob for example here `vmlinux`

   ```bash
   gdb vmlinux
   ```

1. When GDB has started set the target remote to `/tmp/gdb.socket` to connect to
   Firecracker

   ```bash
   (gdb) target remote /tmp/gdb.socket
   ```

With these steps completed you'll now see GDB has stopped at the entry point
ready for us to start inserting breakpoints and debugging.

## Notes

### Software Breakpoints not working on start

When at the initial paused state you'll notice software breakpoints won't work
and only hardware breakpoints will until pages are setup. To circumvent this one
solution is to set a hardware breakpoint at start_kernel and continue. Once
you've hit the start_kernel set the regular breakpoints as you would do
normally. E.G.

```bash
> hbreak start_kernel
> c
```

### Pausing Firecracker while it's running

While Firecracker is running you can pause vcpu 1 by pressing `Ctrl+C` which
will stop the VCPU and allow you to set breakpoints or inspect the current
location

### Halting execution of GDB and Firecracker

To end the debugging session and shutdown Firecracker you can run the `exit`
command in the GDB session which will terminate both
