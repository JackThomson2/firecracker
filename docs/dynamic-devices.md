# Dynamic Virtio Devices

## Overview

Dynamic devices allow customers to provide custom virtio device implementations
as shared libraries (`.so` files). Firecracker loads these at VMM startup via
`dlopen` and integrates them into the standard virtio transport (MMIO or PCI)
alongside built-in devices.

**Trust model:** Plugins are trusted code. No sandboxing beyond the existing
jailer + seccomp environment. Plugin code runs in-process on the VMM event loop
thread.

**Scope (v1):**
- In-process plugin loading via `dlopen`
- C ABI at the FFI boundary (language-agnostic)
- Rust SDK crate for ergonomic plugin authoring
- Synchronous queue processing only
- No snapshot/restore support

## Plugin ABI Contract

The shared library exports these C symbols. ABI version is checked on load;
mismatch causes immediate rejection.

### Version

```c
uint32_t fc_plugin_abi_version(void);
// Must return FC_PLUGIN_ABI_V1 (1)
```

### Lifecycle

```c
// Create device instance.
// config_json: user-provided JSON from VM config (null-terminated UTF-8).
// Returns opaque handle, or NULL on failure (writes error to err_buf).
void* fc_device_create(const char* config_json, char* err_buf, size_t err_buf_len);

// Destroy device instance. Called on VM teardown.
void fc_device_destroy(void* handle);
```

### Metadata

Called after `fc_device_create`, before activation.

```c
typedef struct {
    uint32_t device_type;       // virtio device type ID (must be >= 40)
    uint32_t num_queues;        // number of virtqueues (1-16)
    uint32_t queue_size;        // max queue depth (power of 2, max 1024)
    uint64_t avail_features;    // virtio feature bits offered to guest
    uint32_t config_space_size; // device config space size in bytes
    uint32_t memory_mode;       // 0 = queues_only, 1 = full_guest_memory
} FcDeviceInfo;

int fc_device_info(void* handle, FcDeviceInfo* out);
```

### Activation

Called when the guest driver completes feature negotiation and is ready for IO.

```c
typedef struct {
    void* guest_mem_base;       // non-NULL only if memory_mode == 1
    uint64_t guest_mem_size;
    struct {
        void* desc_table;
        void* avail_ring;
        void* used_ring;
        uint32_t size;
    } queues[16];
    uint32_t num_queues;
    uint64_t acked_features;    // features negotiated with guest
} FcActivationContext;

int fc_device_activate(void* handle, const FcActivationContext* ctx);
```

### IO Processing

Called on the VMM event loop thread when the guest kicks a queue. Plugin must
process descriptors and advance the used ring synchronously.

```c
int fc_device_handle_queue(void* handle, uint32_t queue_idx);
```

### Config Space

```c
int fc_device_read_config(void* handle, uint64_t offset, void* buf, uint32_t len);
int fc_device_write_config(void* handle, uint64_t offset, const void* buf, uint32_t len);
```

### Reset

```c
int fc_device_reset(void* handle);
```

### Return Conventions

All functions returning `int` use: 0 = success, negative = error (errno-style).

### Threading

All ABI functions are called from a single thread (the VMM event loop). Plugins
must NOT spawn threads.

## VMM Integration

### DynamicVirtioDevice Wrapper

A Rust struct that holds the dlopen handle and function pointers, implementing
the `VirtioDevice` trait by delegating across FFI:

```rust
pub struct DynamicVirtioDevice {
    lib: libloading::Library,
    handle: *mut c_void,
    device_info: FcDeviceInfo,
    id: String,
    queues: Vec<Queue>,
    queue_events: Vec<EventFd>,
    avail_features: u64,
    acked_features: u64,
    interrupt_trigger: IrqTrigger,
    activate_evt: EventFd,
    is_activated: bool,
    fns: PluginFns,
}
```

### Loading Sequence

1. `dlopen` the `.so` file
2. Resolve `fc_plugin_abi_version` — reject if != 1
3. Resolve all function symbols
4. Call `fc_device_create(config_json)`
5. Call `fc_device_info()` — populate metadata
6. Construct `DynamicVirtioDevice` with queues/eventfds sized per info
7. Hand to device manager for transport attachment

### Event Loop Integration

Follows the activate-then-queue-events pattern used by existing devices
(virtio-mem, virtio-rng):

- On subscriber init: register activate event
- On activate event: consume it, register per-queue events
- On queue event: consume eventfd, call `fc_device_handle_queue`, signal
  interrupt to guest

### Interrupt Signaling

After `fc_device_handle_queue` returns 0, VMM unconditionally triggers a vring
interrupt to the guest via irqfd. Plugin does not control interrupt timing in v1.

If `handle_queue` returns negative, no interrupt is sent — guest will retry on
next kick.

## Configuration API

### Endpoint

```
PUT /dynamic-devices/{device_id}
```

### Request Body

```json
{
    "device_id": "my-custom-device",
    "plugin_path": "/opt/firecracker/plugins/my_device.so",
    "device_type": 45,
    "num_queues": 2,
    "queue_size": 256,
    "memory_mode": "full_guest_memory",
    "plugin_config": {
        "arbitrary": "json passed to fc_device_create"
    }
}
```

### Fields

| Field | Required | Default | Constraints |
|-------|----------|---------|-------------|
| `device_id` | yes | — | Unique string identifier |
| `plugin_path` | yes | — | Absolute path, must exist, regular file |
| `device_type` | yes | — | >= 40 (custom virtio range) |
| `num_queues` | yes | — | 1–16 |
| `queue_size` | yes | — | Power of 2, max 1024 |
| `memory_mode` | no | `queues_only` | `queues_only` or `full_guest_memory` |
| `plugin_config` | no | `null` | Opaque JSON passed to plugin |

### Limits

- Maximum 8 dynamic devices per VM
- Transport type follows VMM-level configuration (not per-device)

### Jailer Interaction

Plugin `.so` must be inside the jail root filesystem. Users copy it in before
boot, same as kernel/rootfs images. No special jailer logic needed.

## Rust Plugin SDK (`fc-device-sdk`)

A helper crate with zero non-std dependencies. Plugin authors implement a trait;
a macro generates all C ABI glue.

### Trait

```rust
pub trait DynamicDevice: Send {
    fn info(&self) -> DeviceInfo;
    fn activate(&mut self, ctx: &ActivationContext) -> Result<(), String>;
    fn handle_queue(&mut self, queue_idx: u32) -> Result<(), String>;
    fn read_config(&self, offset: u64, buf: &mut [u8]);
    fn write_config(&mut self, offset: u64, buf: &[u8]);
    fn reset(&mut self);
}
```

### Macro Usage

```rust
use fc_device_sdk::{DynamicDevice, DeviceInfo, ActivationContext, fc_plugin};

struct MyDevice { /* ... */ }

impl DynamicDevice for MyDevice {
    // ... implement trait methods ...
}

fc_plugin!(MyDevice, |config: &str| -> Result<MyDevice, String> {
    Ok(MyDevice::new(config)?)
});
```

### What the Macro Generates

- All 8 `extern "C"` functions matching the ABI contract
- Panic safety via `catch_unwind` at each FFI entry point
- Type-safe conversion between C structs and Rust views
- Proper `Box::into_raw` / `Box::from_raw` for handle lifecycle

### Plugin Cargo.toml

```toml
[lib]
crate-type = ["cdylib"]

[dependencies]
fc-device-sdk = "0.1"
```

## Limitations (v1)

- **No snapshot/restore.** VMs with dynamic devices cannot be snapshotted.
- **No async IO.** Plugin must complete all work within `handle_queue`. No
  mechanism for plugin-initiated interrupts.
- **No hot-plug.** Dynamic devices must be configured before boot.
- **No rate limiting.** Built-in rate limiter infrastructure does not apply to
  dynamic devices.
- **Single-threaded.** Plugin cannot offload work to background threads
  (seccomp would block thread creation).

## Future Work (v2+)

- Plugin-initiated interrupts (for async backends)
- Snapshot/restore via `fc_device_save_state` / `fc_device_load_state`
- Hot-plug support
- Optional rate limiting integration
- Plugin health checks / watchdog timeout
