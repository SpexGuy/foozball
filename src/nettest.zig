const std = @import("std");
const ig = @import("imgui");
const engine = @import("engine.zig");
const autogui = @import("autogui.zig");

const math = std.math;

const Allocator = std.mem.Allocator;
const warn = std.debug.warn;
const assert = std.debug.assert;

const heap_allocator = std.heap.c_allocator;

var sim_speed: f32 = 0.1;
var sim_time: f64 = 0;

var frames_per_second: f32 = 90;
var frames_shown: f32 = 30;
var current_frame: f64 = 0;

const MAX_FRAMES = 128;

var rnd = std.rand.DefaultPrng.init(20);

const WindowRenderArgs = struct {
    frames_shown: f32,
    frame_offset: f32,
    max_seq: i64,
};

fn Window(comptime State: type, comptime arraySize: comptime_int, comptime emptyState: State) type {
    return struct {
        data: [arraySize]State = [_]State{emptyState} ** arraySize,
        max_seq: i64 = 0,

        pub fn isValid(self: @This(), seq: i64) bool {
            return seq <= self.max_seq and seq + arraySize > self.max_seq; 
        }

        pub fn slideTo(self: *@This(), seq: i64, fillState: State) void {
            assert(seq + arraySize > self.max_seq);
            while (self.max_seq < seq) {
                self.max_seq += 1;
                self.at(self.max_seq).* = fillState;
            }
        }

        pub fn push(self: *@This(), frame: i64, state: State) void {
            self.slideTo(frame, emptyState);
            self.at(frame).* = state;
        }

        pub fn at(self: *@This(), seq: i64) *State {
            assert(self.isValid(seq));
            return &self.data[@intCast(usize, @mod(seq, arraySize))];
        }

        pub fn drawUI(self: *@This(), label: ?[*:0]const u8, args: WindowRenderArgs) void {
            const style = ig.GetStyle() orelse return;
            const drawList = ig.GetWindowDrawList() orelse return;

            _ = ig.InvisibleButton("window", .{
                .x = ig.CalcItemWidth(),
                .y = ig.GetTextLineHeight() + style.FramePadding.y * 2,
            });
            const min = ig.GetItemRectMin();
            const max = ig.GetItemRectMax();

            {
                drawList.PushClipRectExt(min, max, true);
                defer drawList.PopClipRect();

                const width = max.x - min.x;
                const framesShown = args.frames_shown;
                const offset = 1 - args.frame_offset;
                const framesToDraw = @floatToInt(i64, math.ceil(framesShown));
                const startFrame = args.max_seq - framesToDraw;
                const endFrame = args.max_seq;
                var currFrame = startFrame;
                while (currFrame <= endFrame) : (currFrame += 1) {
                    var state: State = if (self.isValid(currFrame)) self.at(currFrame).* else .Invalid;
                    if (state != .Invalid) {
                        const left = (@intToFloat(f32, endFrame - (currFrame - 1)) - offset) * width / framesShown;
                        const right = (@intToFloat(f32, endFrame - currFrame) - offset) * width / framesShown;
                        drawList.AddRectFilled(
                            .{ .x = max.x - left, .y = min.y },
                            .{ .x = max.x - right, .y = max.y },
                            State.colors[@enumToInt(state)],
                        );
                    }
                }
            }

            if (label != null) {
                drawList.AddTextVec2(.{.x = max.x + style.ItemInnerSpacing.x, .y = min.y + style.FramePadding.y }, ig.GetColorU32(.Text), label);
            }
        }
    };
}

const ServerFrameState = enum {
    Invalid,
    Unknown,
    Known,
    Simulated,
    Missed,
    Retired,

    pub const count = @as(comptime_int, @enumToInt(@This().Retired)) + 1;

    pub const colors = [count]u32 {
        ig.COL32(0, 0, 0, 0),
        ig.COL32(0, 0, 255, 255),
        ig.COL32(0, 255, 255, 255),
        ig.COL32(0, 255, 0, 255),
        ig.COL32(255, 0, 0, 255),
        ig.COL32(20, 20, 20, 255),
    };
};

const NetworkFrameState = enum {
    Invalid,
    Dropped,
    Transmitted,
    Received,

    pub const count = @as(comptime_int, @enumToInt(@This().Received)) + 1;

    pub const colors = [count]u32 {
        ig.COL32(0, 0, 0, 0),
        ig.COL32(255, 0, 0, 255),
        ig.COL32(255, 255, 0, 255),
        ig.COL32(0, 255, 0, 255),
    };
};

const ClientFrameState = enum {
    Invalid,
    Transmitted,
    Acked,
    Missed,
    Retired,

    pub const count = @as(comptime_int, @enumToInt(@This().Retired)) + 1;

    pub const colors = [count]u32 {
        ig.COL32(0, 0, 0, 0),
        ig.COL32(0, 0, 255, 255),
        ig.COL32(0, 255, 255, 255),
        ig.COL32(255, 0, 0, 255),
        ig.COL32(0, 255, 0, 255),
    };
};

const InputPacket = struct {
    end_frame: i64,
    start_frame: i64,
    arrival_time: f64,
};
const OutputPacket = struct {
    sim_frame: i64,
    ack_frame: i64,
    delta_latency: i8,
    arrival_time: f64,
};

const NO_DATA: i8 = -128;

const Client = struct {
    server_window: Window(ServerFrameState, MAX_FRAMES, .Invalid) = .{},
    client_window: Window(ClientFrameState, MAX_FRAMES, .Invalid) = .{},
    input_window: Window(NetworkFrameState, MAX_FRAMES, .Invalid) = .{},
    output_window: Window(NetworkFrameState, MAX_FRAMES, .Invalid) = .{},

    /// History of latency of last 64 frames.  Positive numbers indicate
    /// late packets, negative numbers indicate early packets.  The maximum
    /// value in this array indicates how many frames the client should
    /// shift in order to have ideal latency.
    server_latency_history: Window(i8, 64, NO_DATA) = .{},

    /// average round trip time
    rtt_average: f32 = 150,
    /// percent of average
    rtt_variance_percent: f32 = 40,
    /// -1 is all rtt is client -> server, 1 is all rtt server -> client
    rtt_skew: f32 = 0,
    /// percent of packets lost
    input_packet_loss_percent: f32 = 10,
    output_packet_loss_percent: f32 = 10,
    /// client -> server packets
    inputs: std.ArrayList(InputPacket) = std.ArrayList(InputPacket).init(heap_allocator),
    /// server -> client packets
    outputs: std.ArrayList(OutputPacket) = std.ArrayList(OutputPacket).init(heap_allocator),

    server_max_acked: i64 = 0,

    client_max_simmed: i64 = 0,
    client_max_acked: i64 = 0,
    client_delta_latency: i8 = 0,

    last_client_frame_seq: i64 = 0,
    last_client_frame_time: f64 = 0,
    client_time_scale: f64 = 1,

    pub fn sendInput(self: *Client, frame: i64, sendTime: f64) void {
        if (rnd.random.float(f32) * 100 < self.input_packet_loss_percent) {
            self.input_window.push(frame, .Dropped);
            return; // packet dropped
        }
        self.input_window.push(frame, .Transmitted);
        const rttVariance = self.rtt_variance_percent * self.rtt_average / 200;
        const minRtt = math.max(0, self.rtt_average - rttVariance);
        const rtt = minRtt + rnd.random.float(f32) * rttVariance * 2;
        const inputRttPart = 1 - (self.rtt_skew * 0.5 + 0.5);
        const transmitTime = rtt * inputRttPart * 0.001;

        const numFrames = math.clamp(frame - self.client_max_acked, 1, 45);
        self.inputs.append(.{
            .start_frame = frame - numFrames + 1,
            .end_frame = frame,
            .arrival_time = sendTime + transmitTime,
        }) catch unreachable;
    }

    pub fn sendOutput(self: *Client, simFrame: i64, ackFrame: i64, sendTime: f64) void {
        if (rnd.random.float(f32) * 100 < self.output_packet_loss_percent) {
            self.output_window.push(simFrame, .Dropped);
            return; // packet dropped
        }
        self.output_window.push(simFrame, .Transmitted);
        const rttVariance = self.rtt_variance_percent * self.rtt_average / 200;
        const minRtt = math.max(0, self.rtt_average - rttVariance);
        const rtt = minRtt + rnd.random.float(f32) * rttVariance * 2;
        const outputRttPart = (self.rtt_skew * 0.5 + 0.5);
        const transmitTime = rtt * outputRttPart * 0.001;
        self.outputs.append(.{
            .sim_frame = simFrame,
            .ack_frame = ackFrame,
            .delta_latency = self.calcTargetLatencyDelta(),
            .arrival_time = sendTime + transmitTime,
        }) catch unreachable;
    }

    fn processInputs(self: *Client, simTime: f64, simFrame: i64) void {
        var c = self.inputs.items.len;
        while (c > 0) {
            c -= 1;
            if (self.inputs.items[c].arrival_time <= simTime) {
                self.processInput(self.inputs.items[c], simFrame);
                _ = self.inputs.swapRemove(c);
            }
        }
    }

    fn processOutputs(self: *Client, simTime: f64) void {
        var anyProcessed = false;
        var c = self.outputs.items.len;
        while (c > 0) {
            c -= 1;
            if (self.outputs.items[c].arrival_time <= simTime) {
                anyProcessed = true;
                self.processOutput(self.outputs.items[c]);
                _ = self.outputs.swapRemove(c);
            }
        }
        if (anyProcessed) self.updateClientWindow();
    }

    fn processInput(self: *Client, input: InputPacket, serverFrame: i64) void {
        self.input_window.push(input.end_frame, .Received);
        self.server_window.slideTo(input.end_frame, .Invalid);
        self.server_latency_history.slideTo(input.end_frame, NO_DATA);
        var frame = input.start_frame;
        while (frame <= input.end_frame) : (frame += 1) {
            if (self.server_window.isValid(frame)) {
                const currState = self.server_window.at(frame).*;
                const newState = switch (currState) {
                    .Invalid, .Unknown, .Known => ServerFrameState.Known,
                    else => |state| state,
                };
                self.server_window.at(frame).* = newState;
            }
            if (self.server_latency_history.isValid(frame)) {
                const value = self.server_latency_history.at(frame).*;
                const latency = (serverFrame + 1) - frame;
                if (latency < value or value == NO_DATA) {
                    self.server_latency_history.push(frame, @intCast(i8, latency));
                }
            }
        }
    }

    fn processOutput(self: *Client, output: OutputPacket) void {
        self.output_window.push(output.sim_frame, .Received);
        // Note max sim frame may be larger than max_seq if the client is behind the server somehow
        if (output.sim_frame > self.client_max_simmed) {
            self.client_max_simmed = output.sim_frame;
            self.client_max_acked = output.ack_frame;
            self.client_delta_latency = output.delta_latency;
        }
    }

    fn updateClientWindow(self: *Client) void {
        var frame = self.client_window.max_seq + 1 - MAX_FRAMES;
        while (frame < self.client_window.max_seq) : (frame += 1) {
            const current = self.client_window.at(frame).*;
            if (current != .Invalid and current != .Missed) {
                if (frame <= self.client_max_simmed) {
                    self.client_window.at(frame).* = .Retired;
                } else if (frame <= self.client_max_acked) {
                    self.client_window.at(frame).* = .Acked;
                }
            }
        }
    }

    fn calcTargetLatencyDelta(self: *Client) i8 {
        var maxDelta: i8 = -127;
        for (self.server_latency_history.data) |delta| {
            if (delta > maxDelta) maxDelta = delta;
        }
        return maxDelta;
    }

    pub fn doClientFrame(self: *Client, simTime: f64) void {
        const frame = self.last_client_frame_seq + 1;

        self.processOutputs(simTime);
        self.client_window.push(frame, .Transmitted);
        self.sendInput(frame, simTime);
        
        self.client_time_scale = math.clamp(1 + 0.01 * @intToFloat(f64, self.client_delta_latency), 0.9, 1.4);

        self.last_client_frame_time = simTime;
        self.last_client_frame_seq = frame;
    }

    pub fn doServerFrame(self: *Client, simTime: f64, simFrame: i64) void {
        // process network inputs
        self.processInputs(simTime, simFrame);

        // sim one frame
        var missedInput = true;
        if (self.server_window.isValid(simFrame)) {
            const state = self.server_window.at(simFrame).*;
            const newState = switch(state) {
                .Unknown, .Invalid => ServerFrameState.Missed,
                .Known => ServerFrameState.Simulated,
                else => unreachable,
            };
            missedInput = (newState == .Missed);
            self.server_window.at(simFrame).* = newState;
        } else if (self.server_window.max_seq < simFrame) {
            self.server_window.slideTo(simFrame, .Missed);
        }

        // send frame results and acks
        if (!missedInput and self.server_max_acked < simFrame) {
            self.server_max_acked = simFrame;
        }
        while (self.server_window.isValid(self.server_max_acked + 1) and
               self.server_window.at(self.server_max_acked + 1).* == .Known) {
            self.server_max_acked += 1;
        }

        self.sendOutput(simFrame, self.server_max_acked, simTime);
    }

    pub fn nextClientFrameTime(self: *Client) f64 {
        const frameTime = 1 / (self.client_time_scale * @floatCast(f64, frames_per_second));
        return self.last_client_frame_time + frameTime;
    }

    pub fn calcWindowFractionalSeq(self: *Client, simTime: f64) f64 {
        const extraTime = simTime - self.last_client_frame_time;
        const extraFrames = extraTime * frames_per_second * self.client_time_scale;
        return @intToFloat(f64, self.last_client_frame_seq) + extraFrames;
    }
};

const System = struct {
    clients: []Client = &[_]Client{},
    last_server_frame_time: f64 = 0,
    last_sim_frame: i64 = 0,
    sim_time: f64 = 0,

    pub fn doServerFrame(self: *System, simTime: f64) void {
        const next_frame = self.last_sim_frame + 1;
        for (self.clients) |*client| {
            client.doServerFrame(simTime, next_frame);
        }
        self.last_sim_frame = next_frame;
        self.last_server_frame_time = simTime;
    }

    pub fn nextServerFrameTime(self: *System) f64 {
        const frameTime = 1 / @floatCast(f64, frames_per_second);
        return self.last_server_frame_time + frameTime;
    }

    pub fn update(self: *System, delta: f64) void {
        assert(delta >= 0);
        const newSimTime = self.sim_time + delta;
        // process updates until we hit the new time
        while (true) {
            // figure out what system updates next
            var nextUpdateTime = self.nextServerFrameTime();
            var nextUpdateClient: ?*Client = null;
            for (self.clients) |*client| {
                const clientUpdateTime = client.nextClientFrameTime();
                if (clientUpdateTime < nextUpdateTime) {
                    nextUpdateTime = clientUpdateTime;
                    nextUpdateClient = client;
                }
            }

            // if that update is after the new target time, we're done!
            if (nextUpdateTime > newSimTime)
                break;

            // otherwise perform that update and loop
            if (nextUpdateClient) |client| {
                client.doClientFrame(nextUpdateTime);
            } else {
                self.doServerFrame(nextUpdateTime);
            }
        }

        // finally, save the new sim time for rendering purposes
        self.sim_time = newSimTime;
    }

    pub fn calcFractionalMaxFrame(self: *System) f64 {
        const serverFraction = (self.sim_time - self.last_server_frame_time) / frames_per_second;
        var maxFrame = @intToFloat(f32, self.last_sim_frame) + serverFraction;
        for (self.clients) |*client| {
            const clientFrame = client.calcWindowFractionalSeq(self.sim_time);
            if (clientFrame > maxFrame) {
                maxFrame = clientFrame;
            }
        }
        return maxFrame;
    }
};

var system_initialized = false;
var system = System{};

pub fn drawUI(show: *bool) void {
    if (!system_initialized) {
        const numClients = 3;
        system.clients = heap_allocator.alloc(Client, numClients) catch unreachable;
        for (system.clients) |*client| {
            client.* = .{};
        }
        system_initialized = true;
    }

    const showWindow = ig.BeginExt("Network Test", show, .{});
    defer ig.End();
    if (!showWindow) return;

    ig.LabelText("Time since startup", "%f", engine.time.frameStartSeconds);
    ig.LabelText("Frame delta", "%f", @floatCast(f64, engine.time.frameDeltaSeconds));
    ig.LabelText("Micros since startup", "%llu", engine.time.frameStartMicros);
    ig.LabelText("Frame delta micros", "%u", engine.time.frameDeltaMicros);
    ig.LabelText("Epoch millis", "%u", engine.time.frameStartEpochMillis);
    ig.LabelText("FPS", "%.1f", @floatCast(f64, 1.0 / engine.time.frameDeltaSeconds));

    ig.Separator();

    _ = ig.SliderFloat("Sim Speed", &sim_speed, 0, 1);
    _ = ig.SliderFloat("Frames Shown", &frames_shown, 1, MAX_FRAMES);

    const simDelta = engine.time.frameDeltaSeconds * sim_speed;
    system.update(simDelta);
    const fractionalMaxFrame = system.calcFractionalMaxFrame();
    const frameOffset = @floatCast(f32, @mod(fractionalMaxFrame, 1));
    const maxFrame = @floatToInt(i64, fractionalMaxFrame);

    ig.LabelText("Sim Time", "%f", system.sim_time);
    ig.LabelText("Frame Offset", "%f", @floatCast(f64, frameOffset));

    ig.Separator();

    const windowArgs: WindowRenderArgs = .{
        .frames_shown = frames_shown,
        .frame_offset = frameOffset,
        .max_seq = maxFrame,
    };

    var clientBuf: [64]u8 = undefined;

    if (ig.CollapsingHeaderExt("Server", .{ .DefaultOpen = true })) {
        for (system.clients) |*client, i| {
            ig.PushIDInt(@intCast(i32, i));
            defer ig.PopID();

            const str = std.fmt.bufPrintZ(&clientBuf, "Client {}", .{i}) catch "<error>";
            client.server_window.drawUI(str.ptr, windowArgs);
        }
    }

    ig.Separator();

    for (system.clients) |*client, i| {
        ig.PushIDInt(@intCast(i32, i));
        defer ig.PopID();

        const str = std.fmt.bufPrintZ(&clientBuf, "Client {}", .{i}) catch "<error>";
        if (ig.CollapsingHeaderExt(str.ptr, .{ .DefaultOpen = true })) {
            client.server_window.drawUI("Server", windowArgs);
            client.input_window.drawUI("Input Packets", windowArgs);
            client.output_window.drawUI("Result Packets", windowArgs);
            client.client_window.drawUI("Client", windowArgs);
            ig.LabelText("Time Scale", "%f", client.client_time_scale);
            _ = ig.SliderFloat("Round Trip Time Average", &client.rtt_average, 0, 1000);
            _ = ig.SliderFloatExt("Round Trip Time Variance", &client.rtt_variance_percent, 0, 100, "%.1f%%", 1);
            _ = ig.SliderFloat("Round Trip Time Skew", &client.rtt_skew, -1, 1);
            _ = ig.SliderFloatExt("Input Packet Loss", &client.input_packet_loss_percent, 0, 100, "%.1f%%", 1);
            _ = ig.SliderFloatExt("Output Packet Loss", &client.output_packet_loss_percent, 0, 100, "%.1f%%", 1);
        }
        ig.Separator();
    }

    ig.LabelText("After", "After");
}
