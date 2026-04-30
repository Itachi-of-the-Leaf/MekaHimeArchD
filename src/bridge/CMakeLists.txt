nanobind_add_module(audio_bridge audio_bridge.cpp)

target_include_directories(audio_bridge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# On Linux, miniaudio needs these
target_link_libraries(audio_bridge PRIVATE dl pthread m)
