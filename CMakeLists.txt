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

## ADD LibTorch
if (WIN32)
  set(TORCH_URL https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-2.0.0%2Bcpu.zip)
else ()
  set(TORCH_URL URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip)
endif ()

FetchContent_Declare(
        Torch
        URL ${TORCH_URL}
        # OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(Torch)

message("Torch_FOUND=${Torch_FOUND}")
message("torch_POPULATED=${torch_POPULATED}")
message("torch_BINARY_DIR=${torch_BINARY_DIR}")
message("torch_SOURCE_DIR=${torch_SOURCE_DIR}")
message("Torch_DIR=${Torch_DIR}")
message("CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
message("CMAKE_CURRENT_LIST_DIR=${CMAKE_CURRENT_LIST_DIR}")
message("TORCH_LIBRARIES=${TORCH_LIBRARIES}")
message("TORCH_INSTALL_PREFIX=${TORCH_INSTALL_PREFIX}")

list(APPEND CMAKE_PREFIX_PATH "${torch_SOURCE_DIR}")

# Find the PyTorch package
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add the YetMoreTrees directory to the include path
include_directories(Lothlorien)

# Add the main executable
add_executable(LothlorienTests "main.cpp" Lothlorien/SapWood.h Lothlorien/Mallorn.h)

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
