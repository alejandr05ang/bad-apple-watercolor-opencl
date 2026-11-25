cmake_minimum_required(VERSION 3.22)
project(acuarela LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV para Android compilado con OpenCL (OpenCV_DIR lo pasa app/build.gradle.kts)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
message(STATUS "OpenCV ${OpenCV_VERSION} en ${OpenCV_DIR}")

# Raíz del repositorio: comun/ (efecto) y escritorio/ (medición de RAM y CPU)
set(RAIZ ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)

add_library(acuarela SHARED Nativo.cpp)
target_include_directories(acuarela PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${RAIZ}/comun          # Efecto.hpp: el MISMO pipeline que en escritorio
  ${RAIZ}/escritorio)    # Metricas.hpp: ramProcesoMB() y MedidorCPU
target_link_libraries(acuarela PRIVATE ${OpenCV_LIBS} jnigraphics log android)
