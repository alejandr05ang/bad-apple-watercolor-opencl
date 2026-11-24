cmake_minimum_required(VERSION 3.16)
project(vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimización activada por defecto (importante para un benchmark justo)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# viz = módulo de visualización 3D basado en VTK
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio viz)
find_package(Threads REQUIRED)

message(STATUS "OpenCV version:  ${OpenCV_VERSION}")
message(STATUS "OpenCV include:  ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs:     ${OpenCV_LIBS}")

add_executable(vision.bin Principal.cpp)
target_include_directories(vision.bin PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../comun)   # Efecto.hpp compartido con Android
target_link_libraries(vision.bin PRIVATE ${OpenCV_LIBS} Threads::Threads)

# Carpeta raíz del proyecto: Bad Apple.mp4 y los cuadros de referencia
target_compile_definitions(vision.bin PRIVATE RUTA_RECURSOS="${CMAKE_CURRENT_SOURCE_DIR}/..")
