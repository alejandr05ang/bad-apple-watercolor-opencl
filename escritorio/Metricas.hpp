// =============================================================================
//  Metricas.hpp - Medición de recursos del sistema (GNU/Linux)
//    - RAM del proceso  : /proc/self/status (VmRSS)
//    - CPU del proceso  : std::clock() / tiempo real
//    - GPU NVIDIA       : nvidia-smi en un hilo aparte (1 consulta por segundo)
// =============================================================================
#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>

// Memoria RAM residente del proceso en MB
inline double ramProcesoMB() {
    std::ifstream f("/proc/self/status");
    std::string linea;
    while (std::getline(f, linea)) {
        if (linea.rfind("VmRSS:", 0) == 0) {
            long kb = std::stol(linea.substr(6));
            return kb / 1024.0;
        }
    }
    return -1;
}

// % de CPU usado por el proceso (100% = un núcleo completo)
class MedidorCPU {
    std::clock_t c0 = std::clock();
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double ultimo = 0;
public:
    double porcentaje() {
        auto t1 = std::chrono::steady_clock::now();
        double real = std::chrono::duration<double>(t1 - t0).count();
        if (real >= 0.5) {  // ventana de medición de 0.5 s
            std::clock_t c1 = std::clock();
            ultimo = 100.0 * (double(c1 - c0) / CLOCKS_PER_SEC) / real;
            c0 = c1; t0 = t1;
        }
        return ultimo;
    }
};

// Consulta periódica a nvidia-smi (no bloquea el bucle de video)
class MonitorGPU {
    std::atomic<bool> activo{true};
    std::thread hilo;
public:
    std::atomic<int> utilizacion{-1};  // % de uso de la GPU
    std::atomic<int> memoriaTotal{-1}; // MB usados en la GPU (todo el sistema)
    std::atomic<int> memoriaProceso{-1}; // MB usados en la GPU por este proceso

    MonitorGPU() {
        hilo = std::thread([this] {
            const int pid = getpid();
            while (activo) {
                FILE* f = popen("nvidia-smi --query-gpu=utilization.gpu,memory.used "
                                "--format=csv,noheader,nounits 2>/dev/null", "r");
                if (f) {
                    int u, m;
                    if (fscanf(f, "%d, %d", &u, &m) == 2) { utilizacion = u; memoriaTotal = m; }
                    pclose(f);
                }
                f = popen("nvidia-smi --query-compute-apps=pid,used_memory "
                          "--format=csv,noheader,nounits 2>/dev/null", "r");
                if (f) {
                    int p, m, encontrado = 0;
                    while (fscanf(f, "%d, %d", &p, &m) == 2)
                        if (p == pid) { memoriaProceso = m; encontrado = 1; }
                    if (!encontrado) memoriaProceso = 0;
                    pclose(f);
                }
                for (int i = 0; i < 10 && activo; i++)
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }
    ~MonitorGPU() { activo = false; if (hilo.joinable()) hilo.join(); }
};
