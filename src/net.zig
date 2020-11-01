const std = @import("std");
const engine = @import("engine.zig");
const enet = @import("enet");

const heap_allocator = std.heap.c_allocator;

// config vars
pub var serverConnectionPort: u16 = 0x4d3;
pub var addressFamily: raw.AddressFamily = .ipv4;
pub var maxConnectionsPerFrame: usize = 4;
pub var pendingConnectionTimeoutMicros: u64 = 2000000;

pub fn _init() !void {
    try enet.initialize();
    std.debug.print("ENet initialized successfully!\n", .{});
}

pub fn _deinit() void {
    enet.deinitialize();
}