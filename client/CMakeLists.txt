cmake_minimum_required(VERSION 3.18)
project(amika_client CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Release flags ─────────────────────────────────────────────────────────────
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native)
endif()

# ── IXWebSocket — no TLS; Cloudflare Tunnel handles encryption ────────────────
include(FetchContent)
FetchContent_Declare(
    IXWebSocket
    GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
    GIT_TAG        v11.4.5
)
set(USE_TLS     OFF CACHE BOOL "" FORCE)
set(USE_OPEN_SSL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(IXWebSocket)

# ── Main executable ───────────────────────────────────────────────────────────
add_executable(amika_client WIN32 src/main.cpp)

# WIN32 entry point still calls main() — we just suppress the console window
# for future GUI builds. For Phase 1 (console telemetry) we keep the console:
set_target_properties(amika_client PROPERTIES
    WIN32_EXECUTABLE FALSE   # keep the console so telemetry prints are visible
    SUFFIX ".exe"
)

target_include_directories(amika_client PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # miniaudio.h
)

# ── Windows libraries required by miniaudio (WASAPI) and IXWebSocket ─────────
target_link_libraries(amika_client PRIVATE
    ixwebsocket
    ws2_32      # WinSock2  — sockets
    mswsock     # WinSock extensions
    winmm       # Windows Multimedia — timer resolution for WASAPI
    bcrypt      # CNG random  — IXWebSocket
    iphlpapi    # IP Helper  — IXWebSocket interface enumeration
    ole32       # COM        — miniaudio WASAPI init
    oleaut32    # OLE        — miniaudio WASAPI
)

install(TARGETS amika_client DESTINATION bin)

