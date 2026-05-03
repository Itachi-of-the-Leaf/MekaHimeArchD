#define MINIAUDIO_IMPLEMENTATION
#include "../../include/miniaudio.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>
#include <mutex>
#include <cstring>
#include <chrono>
#include <cstdio>

namespace nb = nanobind;

class AudioBridge {
public:
    AudioBridge(uint32_t sampleRate = 48000, uint32_t channels = 1, uint32_t bufferSize = 48000) 
        : m_sampleRate(sampleRate), m_channels(channels), m_chunkCount(0) {
        
        // Reserve enough space for 1 second of audio to prevent reallocation overhead
        m_unread_buffer.reserve(sampleRate * channels);
        
        ma_device_config config = ma_device_config_init(ma_device_type_capture);
        config.capture.format = ma_format_f32;
        config.capture.channels = channels;
        config.sampleRate = sampleRate;
        config.dataCallback = data_callback;
        config.pUserData = this;

        if (ma_device_init(NULL, &config, &m_device) != MA_SUCCESS) {
            throw std::runtime_error("Failed to initialize miniaudio device");
        }

        m_lastChunkTime = std::chrono::steady_clock::now();
        std::printf("[AudioBridge C++] Initialized @ %uHz, %uch\n", sampleRate, channels);
    }

    ~AudioBridge() {
        stop();
        ma_device_uninit(&m_device);
    }

    void start() {
        if (ma_device_start(&m_device) != MA_SUCCESS) {
            throw std::runtime_error("Failed to start miniaudio device");
        }
        m_lastChunkTime = std::chrono::steady_clock::now();
        std::printf("[AudioBridge C++] Capture started\n");
    }

    void stop() {
        ma_device_stop(&m_device);
        std::printf("[AudioBridge C++] Capture stopped (served %u chunks)\n", m_chunkCount);
    }

    nb::ndarray<nb::numpy, float> get_latest_chunk() {
        // Release the GIL while we acquire the lock — prevents stalling the
        // Python async event loop if the audio callback holds the mutex.
        nb::gil_scoped_release release;

        auto now = std::chrono::steady_clock::now();

        std::lock_guard<std::mutex> lock(m_mutex);
        
        size_t size = m_unread_buffer.size();
        
        // If no new audio, return empty array
        if (size == 0) {
            // Re-acquire GIL before creating Python objects
            nb::gil_scoped_acquire acquire;
            size_t shape[1] = { 0 };
            return nb::ndarray<nb::numpy, float>(nullptr, 1, shape);
        }

        // Copy data to a new pointer so Python owns it securely (Fixes Race Condition)
        float* out_data = new float[size];
        std::memcpy(out_data, m_unread_buffer.data(), size * sizeof(float));
        
        // CLEAR the buffer so we never return overlapping data (Fixes Echo)
        m_unread_buffer.clear();

        m_chunkCount++;

        // Latency logging (every 20 chunks to avoid spam)
        if (m_chunkCount % 20 == 0) {
            auto delta = std::chrono::duration_cast<std::chrono::microseconds>(now - m_lastChunkTime);
            double delta_ms = delta.count() / 1000.0;
            double audio_ms = (static_cast<double>(size) / m_sampleRate) * 1000.0;
            std::printf("[AudioBridge C++] Chunk #%u: %zu samples (%.1fms audio), interval=%.1fms\n",
                        m_chunkCount, size, audio_ms, delta_ms);
        }
        m_lastChunkTime = now;

        // Re-acquire GIL before creating Python objects
        nb::gil_scoped_acquire acquire;

        // Tell nanobind to delete the memory when Python garbage collects the array
        nb::capsule owner(out_data, [](void *p) noexcept {
            delete[] (float *)p;
        });

        size_t shape[1] = { size };
        return nb::ndarray<nb::numpy, float>(out_data, 1, shape, owner);
    }

    uint32_t get_sample_rate() const {
        return m_sampleRate;
    }

private:
    static void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
        AudioBridge* pBridge = (AudioBridge*)pDevice->pUserData;
        if (pInput == nullptr) return;

        const float* fInput = (const float*)pInput;
        size_t total_samples = frameCount * pBridge->m_channels;

        std::lock_guard<std::mutex> lock(pBridge->m_mutex);
        
        // Append strictly NEW audio to the end of the unread buffer
        pBridge->m_unread_buffer.insert(pBridge->m_unread_buffer.end(), fInput, fInput + total_samples);
    }

    ma_device m_device;
    uint32_t m_sampleRate;
    uint32_t m_channels;
    uint32_t m_chunkCount;
    std::vector<float> m_unread_buffer;
    std::mutex m_mutex;
    std::chrono::steady_clock::time_point m_lastChunkTime;
};

NB_MODULE(audio_bridge, m) {
    nb::class_<AudioBridge>(m, "AudioBridge")
        .def(nb::init<uint32_t, uint32_t, uint32_t>(), 
             nb::arg("sampleRate") = 48000, 
             nb::arg("channels") = 1, 
             nb::arg("bufferSize") = 48000)
        .def("start", &AudioBridge::start)
        .def("stop", &AudioBridge::stop)
        .def("get_latest_chunk", &AudioBridge::get_latest_chunk)
        .def("get_sample_rate", &AudioBridge::get_sample_rate);
}
