cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# Set the project name and version
project(Lothlorien VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

## ADD GTests
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(USE_CUDNN 1)

if (WIN32)
    set(TORCH_DIR "C:/libTorch/libtorch" CACHE PATH "Path to libtorch")
else ()
    set(TORCH_DIR "/nfs/libs/libtorch")# CACHE PATH "Path to libtorch")
endif ()
message("TORCH_DIR=${TORCH_DIR}")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
list(APPEND CMAKE_PROGRAM_PATH "${TORCH_DIR}/lib")

message("CMAKE_PROGRAM_PATH=${CMAKE_PROGRAM_PATH}")

# Find the PyTorch package
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add the YetMoreTrees directory to the include path
include_directories(Lothlorien)

# Add the main executable
add_executable(LothlorienTests "main.cpp")

# Link the PyTorch libraries
target_link_libraries(LothlorienTests "${TORCH_LIBRARIES}")

# Link the GTest libraries
target_link_libraries(LothlorienTests GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(LothlorienTests)


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET LothlorienTests
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:LothlorienTests>)
endif (MSVC)

# Set the output directory
set_target_properties(LothlorienTests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
