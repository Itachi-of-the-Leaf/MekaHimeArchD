nanobind_add_module(audio_bridge audio_bridge.cpp)

target_include_directories(audio_bridge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# Platform-specific link dependencies for miniaudio
if(UNIX AND NOT APPLE)
    # Linux: miniaudio needs dl (dynamic loading), pthread (threading), m (math)
    target_link_libraries(audio_bridge PRIVATE dl pthread m)
elseif(APPLE)
    # macOS: CoreAudio frameworks
    find_library(COREAUDIO_LIB CoreAudio)
    find_library(AUDIOUNIT_LIB AudioUnit)
    find_library(COREFOUNDATION_LIB CoreFoundation)
    target_link_libraries(audio_bridge PRIVATE ${COREAUDIO_LIB} ${AUDIOUNIT_LIB} ${COREFOUNDATION_LIB})
elseif(WIN32)
    # Windows: miniaudio uses runtime dynamic loading (LoadLibrary) for WASAPI/DirectSound.
    # No explicit link dependencies needed in most cases.
    # Uncomment below if you get linker errors for specific backends:
    # target_link_libraries(audio_bridge PRIVATE winmm ole32)
endif()

install(TARGETS audio_bridge DESTINATION src/bridge)
