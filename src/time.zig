const std = @import("std");
const Timer = std.time.Timer;

var runningTimer: Timer = undefined;

/// Maximum delta between frames.  Prevents giant timesteps.
/// Set this value to adjust the maximum.  Does not affect
/// absolute timing in any way.  Defaults to 500 mS
pub var maxFrameDeltaMicros: u64 = 500 * 1000;

/// Microseconds since program startup, snapshot at
/// beginning of current frame
pub var frameStartMicros: u64 = 0;

/// Microseconds elapsed between last two frames.
/// Clamped by maxFrameDeltaMicros.  Accumulating
/// deltas may cause error.
pub var frameDeltaMicros: u32 = 0;

/// Seconds since program start, snapshot at beginning
/// of current frame
pub var frameStartSeconds: f64 = 0;

/// Seconds elapsed between last two frames.
/// Clamped by maxFrameDeltaMicros.  Accumulating
/// deltas may cause error.
pub var frameDeltaSeconds: f32 = 0;

/// Start of this frame in epoch time
pub var frameStartEpochMillis: i64 = 0;

pub fn _init() !void {
    runningTimer = try Timer.start();
    _beginFrame();
}

pub fn _beginFrame() void {
    const currentTimeMicros = nanosSinceStartup() / 1000;
    const actualDeltaMicros = currentTimeMicros - frameStartMicros;
    const reportedDeltaMicros = std.math.min(actualDeltaMicros, maxFrameDeltaMicros);
    
    frameStartMicros = currentTimeMicros;
    frameDeltaMicros = @intCast(u32, reportedDeltaMicros);
    frameStartSeconds = @intToFloat(f64, currentTimeMicros) / 1000000.0;
    frameDeltaSeconds = @intToFloat(f32, reportedDeltaMicros) / 1000000.0;
    frameStartEpochMillis = std.time.milliTimestamp();
}

pub fn nanosSinceStartup() u64 {
    return runningTimer.read();
}
