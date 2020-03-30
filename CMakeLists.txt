cmake_minimum_required(VERSION 3.10)
project(huawei_warmup)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -lpthread -D TEST -D WINDOWS_SSE")

set(SOURCE_FILES main.cpp)
add_executable(huawei_warmup main_finaltry.cpp)

find_package(Threads REQUIRED)
target_link_libraries(huawei_warmup Threads::Threads)