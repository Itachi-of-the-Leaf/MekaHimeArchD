cmake_minimum_required(VERSION 3.15)
project(amikas_ears)

# Find Python and nanobind
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Add the audio bridge module
add_subdirectory(src/bridge)
