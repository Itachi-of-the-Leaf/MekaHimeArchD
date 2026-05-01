/**
 * amika_client / src/main.cpp — Phase 1: "Dumb" Echo Pipe
 *
 * Target: Native Windows x64 (.exe) via MinGW-w64 cross-compile.
 * Audio:  miniaudio with WASAPI backend (zero-driver-overhead).
 *         Captures 48 kHz mono int16 PCM in 20 ms chunks (1920 bytes).
 *         Full-duplex: echoed audio plays back in real-time through speakers.
 *
 * Network: IXWebSocket binary WebSocket.
 *          TCP_NODELAY is explicitly requested to flush every 20 ms frame
 *          without Nagle batching.
 *
 * Usage:
 *   amika_client.exe [ws://HOST:PORT/audio]
 *   Default: ws://localhost:8765/audio
 *
 * Build (cross-compile from WSL):
 *   cd client && mkdir build && cd build
 *   cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchain-mingw64.cmake -DCMAKE_BUILD_TYPE=Release
 *   make -j$(nproc)
 *   # → amika_client.exe in build/
 */

// ── Windows platform first, before miniaudio ─────────────────────────────────
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>


#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXNetSystem.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// ── Audio constants ───────────────────────────────────────────────────────────

static constexpr uint32_t SAMPLE_RATE   = 48000;          // WASAPI native rate
static constexpr uint32_t CHANNELS      = 1;               // mono
static constexpr uint32_t CHUNK_MS      = 20;              // 20 ms frame
static constexpr uint32_t CHUNK_FRAMES  = (SAMPLE_RATE * CHUNK_MS) / 1000;  // 960
static constexpr uint32_t CHUNK_BYTES   = CHUNK_FRAMES * CHANNELS * sizeof(int16_t); // 1920

// ── Thread-safe capture queue (callback → sender thread) ─────────────────────

struct CaptureQueue {
    std::mutex               mtx;
    std::queue<std::string>  q;   // binary blob stored as std::string for IXWebSocket

    void push(const void* data, size_t bytes) {
        std::lock_guard<std::mutex> lk(mtx);
        q.emplace(reinterpret_cast<const char*>(data), bytes);
    }

    bool pop(std::string& out) {
        std::lock_guard<std::mutex> lk(mtx);
        if (q.empty()) return false;
        out = std::move(q.front());
        q.pop();
        return true;
    }
} g_capture;

// ── Thread-safe playback queue (WebSocket receive → playback callback) ────────

struct PlaybackQueue {
    std::mutex                        mtx;
    std::queue<std::vector<int16_t>>  q;
    static constexpr size_t           MAX_DEPTH = 8; // ~160 ms max lag before dropping

    void push(const void* data, size_t bytes) {
        std::lock_guard<std::mutex> lk(mtx);
        if (q.size() >= MAX_DEPTH) {
            q.pop(); // evict oldest to preserve low-latency
        }
        const int16_t* p = reinterpret_cast<const int16_t*>(data);
        q.emplace(p, p + bytes / sizeof(int16_t));
    }

    // Returns true if a frame was available; otherwise fills dst with silence.
    bool pop(int16_t* dst, uint32_t frames) {
        std::lock_guard<std::mutex> lk(mtx);
        if (q.empty()) return false;
        auto& front = q.front();
        uint32_t n = std::min<uint32_t>(frames, static_cast<uint32_t>(front.size()));
        std::memcpy(dst, front.data(), n * sizeof(int16_t));
        if (n < frames)
            std::memset(dst + n, 0, (frames - n) * sizeof(int16_t));
        q.pop();
        return true;
    }
} g_playback;

// ── miniaudio callbacks ───────────────────────────────────────────────────────

// Called by WASAPI capture thread — must be minimal, non-blocking.
void on_capture(ma_device* /*dev*/, void* /*output*/,
                const void* input, ma_uint32 frame_count)
{
    if (input)
        g_capture.push(input, frame_count * CHANNELS * sizeof(int16_t));
}

// Called by WASAPI playback thread — fill the output buffer or silence.
void on_playback(ma_device* /*dev*/, void* output,
                 const void* /*input*/, ma_uint32 frame_count)
{
    int16_t* dst = reinterpret_cast<int16_t*>(output);
    if (!g_playback.pop(dst, frame_count))
        std::memset(dst, 0, frame_count * CHANNELS * sizeof(int16_t));
}

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    const std::string url =
        (argc >= 2) ? argv[1] : "ws://localhost:8765/audio";

    std::printf("=== Amika Echo Client — Phase 1 ===\n");
    std::printf("  Server   : %s\n", url.c_str());
    std::printf("  Chunk    : %u bytes / %u ms @ %u Hz mono int16\n",
                CHUNK_BYTES, CHUNK_MS, SAMPLE_RATE);
    std::printf("  Backend  : WASAPI (Windows Audio Session API)\n\n");

    // ── IXWebSocket setup ─────────────────────────────────────────────────────
    ix::initNetSystem();

    ix::WebSocket ws;
    ws.setUrl(url);
    ws.disableAutomaticReconnection();    // fail fast during network testing
    ws.setPingInterval(0);                // no keepalive pings — we stream constantly
    // IXWebSocket v11.4.5 sets TCP_NODELAY unconditionally in SocketConnect::configure()
    // — no explicit API call required.

    std::atomic<bool> connected{false};
    std::atomic<uint64_t> sent_frames{0};
    std::atomic<uint64_t> recv_frames{0};

    ws.setOnMessageCallback([&](const ix::WebSocketMessagePtr& msg) {
        switch (msg->type) {
        case ix::WebSocketMessageType::Open:
            std::printf("[WS] ✓ Connected.\n");
            connected.store(true, std::memory_order_release);
            break;
        case ix::WebSocketMessageType::Close:
            std::printf("[WS] Connection closed: %s\n",
                        msg->closeInfo.reason.c_str());
            connected.store(false, std::memory_order_release);
            break;
        case ix::WebSocketMessageType::Error:
            std::fprintf(stderr, "[WS] ERROR: %s\n",
                         msg->errorInfo.reason.c_str());
            break;
        case ix::WebSocketMessageType::Message:
            if (msg->binary && !msg->str.empty()) {
                // Push echoed frame into the WASAPI playback queue
                g_playback.push(msg->str.data(), msg->str.size());
                recv_frames.fetch_add(1, std::memory_order_relaxed);
            }
            break;
        default:
            break;
        }
    });

    ws.start();

    // Wait up to 5 s for the handshake
    {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!connected.load(std::memory_order_acquire)) {
            if (std::chrono::steady_clock::now() > deadline) {
                std::fprintf(stderr, "[ERROR] Could not connect to %s — is the server running?\n",
                             url.c_str());
                ws.stop();
                ix::uninitNetSystem();
                return 1;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    // ── miniaudio: WASAPI capture ─────────────────────────────────────────────
    ma_device_config cap_cfg      = ma_device_config_init(ma_device_type_capture);
    cap_cfg.capture.format        = ma_format_s16;
    cap_cfg.capture.channels      = CHANNELS;
    cap_cfg.sampleRate            = SAMPLE_RATE;
    cap_cfg.periodSizeInFrames    = CHUNK_FRAMES;  // 960 frames = 20 ms
    cap_cfg.dataCallback          = on_capture;
    // WASAPI: prefer exclusive mode for minimal driver latency
    cap_cfg.wasapi.noAutoConvertSRC   = 0;          // allow OS resampling if needed

    ma_device cap_dev;
    if (ma_device_init(nullptr, &cap_cfg, &cap_dev) != MA_SUCCESS) {
        std::fprintf(stderr, "[ERROR] Failed to open capture device.\n");
        ws.stop(); ix::uninitNetSystem(); return 1;
    }

    // ── miniaudio: WASAPI playback ────────────────────────────────────────────
    ma_device_config pb_cfg       = ma_device_config_init(ma_device_type_playback);
    pb_cfg.playback.format        = ma_format_s16;
    pb_cfg.playback.channels      = CHANNELS;
    pb_cfg.sampleRate             = SAMPLE_RATE;
    pb_cfg.periodSizeInFrames     = CHUNK_FRAMES;
    pb_cfg.dataCallback           = on_playback;

    ma_device pb_dev;
    if (ma_device_init(nullptr, &pb_cfg, &pb_dev) != MA_SUCCESS) {
        std::fprintf(stderr, "[ERROR] Failed to open playback device.\n");
        ma_device_uninit(&cap_dev);
        ws.stop(); ix::uninitNetSystem(); return 1;
    }

    ma_device_start(&cap_dev);
    ma_device_start(&pb_dev);
    std::printf("[AUDIO] ✓ WASAPI capture + playback started.\n");
    std::printf("[AUDIO] Speak into your microphone. You should hear yourself echoed back.\n");
    std::printf("        Press ENTER to stop.\n\n");

    // ── Sender thread: drain capture queue → WebSocket ───────────────────────
    std::atomic<bool> running{true};
    std::thread sender([&] {
        std::string frame;
        while (running.load(std::memory_order_relaxed)) {
            if (g_capture.pop(frame)) {
                ws.sendBinary(frame);

                uint64_t s = sent_frames.fetch_add(1, std::memory_order_relaxed) + 1;
                // Telemetry: every 50 frames (~1 s), print lag
                if (s % 50 == 0) {
                    int64_t lag = static_cast<int64_t>(s)
                                - static_cast<int64_t>(recv_frames.load(std::memory_order_relaxed));
                    std::printf("[RTT] sent=%" PRIu64 "  echoed=%" PRIu64 "  lag=%" PRId64 " frames (%+" PRId64 " ms)\n",
                                s, recv_frames.load(), lag, lag * static_cast<int64_t>(CHUNK_MS));
                    std::fflush(stdout);
                }
            } else {
                // Nothing to send — spin-sleep to avoid 100% CPU
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        }
    });

    std::getchar(); // block until the user presses ENTER

    // ── Graceful shutdown ─────────────────────────────────────────────────────
    running.store(false, std::memory_order_relaxed);
    sender.join();

    ma_device_stop(&cap_dev);
    ma_device_stop(&pb_dev);
    ma_device_uninit(&cap_dev);
    ma_device_uninit(&pb_dev);

    ws.stop();
    ix::uninitNetSystem();

    std::printf("\n[DONE] sent=%" PRIu64 "  echoed=%" PRIu64 "\n",
                sent_frames.load(), recv_frames.load());
    return 0;
}
