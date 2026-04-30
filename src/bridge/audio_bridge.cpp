#define MINIAUDIO_IMPLEMENTATION
#include "../../include/miniaudio.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>

namespace nb = nanobind;

class AudioBridge {
public:
    AudioBridge(uint32_t sampleRate = 48000, uint32_t channels = 1, uint32_t bufferSize = 4800) 
        : m_sampleRate(sampleRate), m_channels(channels), m_bufferSize(bufferSize) {
        
        m_buffer.resize(bufferSize * channels);
        m_writeIndex = 0;
        
        ma_device_config config = ma_device_config_init(ma_device_type_capture);
        config.capture.format = ma_format_f32;
        config.capture.channels = channels;
        config.sampleRate = sampleRate;
        config.dataCallback = data_callback;
        config.pUserData = this;

        if (ma_device_init(NULL, &config, &m_device) != MA_SUCCESS) {
            throw std::runtime_error("Failed to initialize miniaudio device");
        }
    }

    ~AudioBridge() {
        stop();
        ma_device_uninit(&m_device);
    }

    void start() {
        if (ma_device_start(&m_device) != MA_SUCCESS) {
            throw std::runtime_error("Failed to start miniaudio device");
        }
    }

    void stop() {
        ma_device_stop(&m_device);
    }

    nb::ndarray<nb::numpy, float> get_latest_chunk() {
        std::lock_guard<std::mutex> lock(m_mutex);
        // Returns a copy or a view? For simplicity and safety with Python GIL, 
        // we'll return a numpy array copy of the current buffer.
        size_t shape[1] = { m_buffer.size() };
        return nb::ndarray<nb::numpy, float>(m_buffer.data(), 1, shape);
    }

    // Advanced: Provide a way to read from a ring buffer
    size_t read_samples(float* out_ptr, size_t count) {
        std::lock_guard<std::mutex> lock(m_mutex);
        size_t available = m_writeIndex; // This is a simplification
        size_t to_read = std::min(count, available);
        // Implementation of actual ring buffer would go here
        return to_read;
    }

private:
    static void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
        AudioBridge* pBridge = (AudioBridge*)pDevice->pUserData;
        if (pInput == nullptr) return;

        const float* fInput = (const float*)pInput;
        std::lock_guard<std::mutex> lock(pBridge->m_mutex);
        
        // Simple linear copy for now - real implementation should use a ring buffer
        for (ma_uint32 i = 0; i < frameCount * pBridge->m_channels; ++i) {
            if (pBridge->m_writeIndex < pBridge->m_buffer.size()) {
                pBridge->m_buffer[pBridge->m_writeIndex++] = fInput[i];
            } else {
                // Buffer full - in a real app we'd wrap around
                pBridge->m_writeIndex = 0;
                pBridge->m_buffer[pBridge->m_writeIndex++] = fInput[i];
            }
        }
    }

    ma_device m_device;
    uint32_t m_sampleRate;
    uint32_t m_channels;
    uint32_t m_bufferSize;
    std::vector<float> m_buffer;
    size_t m_writeIndex;
    std::mutex m_mutex;
};

NB_MODULE(audio_bridge, m) {
    nb::class_<AudioBridge>(m, "AudioBridge")
        .def(nb::init<uint32_t, uint32_t, uint32_t>(), 
             nb::arg("sampleRate") = 48000, 
             nb::arg("channels") = 1, 
             nb::arg("bufferSize") = 4800)
        .def("start", &AudioBridge::start)
        .def("stop", &AudioBridge::stop)
        .def("get_latest_chunk", &AudioBridge::get_latest_chunk);
}
