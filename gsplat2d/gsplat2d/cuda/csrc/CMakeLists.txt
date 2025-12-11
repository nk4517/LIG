cmake_minimum_required(VERSION 3.12)  # You can adjust the minimum required version

project(gsplat2d LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70 75 89)  # Ti 2080 uses 75. V100 uses 70. RTX 4090 uses 89.

find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

# our library library
add_library(gsplat2d
    forward.cu
    backward.cu
    bindings.cu
    gradient_aware_rasterizer.cu
    gradient_aware_upscaler.cu
    ext.cpp
)

target_link_libraries(gsplat2d PUBLIC CUDA::cudart pybind11::module)
target_include_directories(gsplat2d PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${pybind11_INCLUDE_DIRS}
)

target_include_directories(gsplat2d PRIVATE
        C:/Users/user/AppData/Local/pypoetry/Cache/virtualenvs/gsplat-sandbox-2l1tlsyc-py3.10/Lib/site-packages/torch/include
        C:/Users/user/AppData/Local/pypoetry/Cache/virtualenvs/gsplat-sandbox-2l1tlsyc-py3.10/Lib/site-packages/torch/include/torch/csrc/api/include
)

# # To add an executable that uses the gsplat2d library,
# # follow example in the comments for a script `run_forward.cpp`
# # Add the executable
# add_executable(run_forward run_forward.cpp)

# # Link against CUDA runtime library
# target_link_libraries(run_forward PUBLIC cuda gsplat2d)

# # Include directories for the header-only library
# target_include_directories(run_forward PRIVATE
#     ${PROJECT_SOURCE_DIR}/third_party/glm
# )

# torch include (adjust path as needed)
# target_include_directories(gsplat2d PRIVATE path/to/torch/include)

#include_directories(gsplat/cuda/csrc)
#include_directories(gsplat/cuda/csrc/third_party/glm/)
#include_directories(gsplat/cuda/include)
#include_directories(${CUDAToolkit_INCLUDE_DIRS})

