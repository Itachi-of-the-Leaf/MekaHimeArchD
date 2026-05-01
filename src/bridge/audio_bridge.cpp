#define MINIAUDIO_IMPLEMENTATION
#include "../../include/miniaudio.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>
#include <mutex>
#include <cstring>

namespace nb = nanobind;

class AudioBridge {
public:
    AudioBridge(uint32_t sampleRate = 48000, uint32_t channels = 1, uint32_t bufferSize = 48000) 
        : m_sampleRate(sampleRate), m_channels(channels) {
        
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
        
        size_t size = m_unread_buffer.size();
        
        // If no new audio, return empty array
        if (size == 0) {
            size_t shape[1] = { 0 };
            return nb::ndarray<nb::numpy, float>(nullptr, 1, shape);
        }

        // Copy data to a new pointer so Python owns it securely (Fixes Race Condition)
        float* out_data = new float[size];
        std::memcpy(out_data, m_unread_buffer.data(), size * sizeof(float));
        
        // CLEAR the buffer so we never return overlapping data (Fixes Echo)
        m_unread_buffer.clear();

        // Tell nanobind to delete the memory when Python garbage collects the array
        nb::capsule owner(out_data, [](void *p) noexcept {
            delete[] (float *)p;
        });

        size_t shape[1] = { size };
        return nb::ndarray<nb::numpy, float>(out_data, 1, shape, owner);
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
    std::vector<float> m_unread_buffer;
    std::mutex m_mutex;
};

NB_MODULE(audio_bridge, m) {
    nb::class_<AudioBridge>(m, "AudioBridge")
        .def(nb::init<uint32_t, uint32_t, uint32_t>(), 
             nb::arg("sampleRate") = 48000, 
             nb::arg("channels") = 1, 
             nb::arg("bufferSize") = 48000)
        .def("start", &AudioBridge::start)
        .def("stop", &AudioBridge::stop)
        .def("get_latest_chunk", &AudioBridge::get_latest_chunk);
}
