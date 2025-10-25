cmake_minimum_required(VERSION 3.21)
project(CPU_Benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt5 REQUIRED COMPONENTS Core Widgets Charts)

# For OpenMP support (if needed)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(OpenCL REQUIRED)

# Auto-generate MOC files
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(SOURCES
    main.cpp 
    mainwindow.cpp
    benchmark.cpp
    openclinterface.cpp
    mainwindow.ui
    mainwindow.h
    benchmark.h
    openclinterface.h
    )

qt5_wrap_cpp(MOC_SOURCES ${HEADERS})

add_executable(${PROJECT_NAME} ${SOURCES})

add_compile_definitions(CL_TARGET_OPENCL_VERSION=220)

target_link_libraries(${PROJECT_NAME} Qt5::Core Qt5::Widgets Qt5::Charts OpenCL::OpenCL)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

#target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp -foffload-options=-march=gfx1103)
target_compile_options(${PROJECT_NAME} PRIVATE -lOpenCL)

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()