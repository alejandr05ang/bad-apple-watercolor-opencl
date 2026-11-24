// =============================================================================
//  Audio.hpp - Reproduce el audio de Bad Apple con ffplay en un proceso aparte
//  (OpenCV no reproduce audio). El video de la máscara se sincroniza con el
//  reloj desde que se inició el audio.
// =============================================================================
#pragma once

#include <csignal>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

class ReproductorAudio {
    pid_t pid = -1;
public:
    bool iniciar(const std::string& ruta) {
        detener();
        const char* args[] = {"ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                              ruta.c_str(), nullptr};
        if (posix_spawnp(&pid, "ffplay", nullptr, nullptr,
                         const_cast<char* const*>(args), environ) != 0) {
            pid = -1;
            return false;
        }
        return true;
    }
    void detener() {
        if (pid > 0) {
            kill(pid, SIGTERM);
            waitpid(pid, nullptr, 0);
            pid = -1;
        }
    }
    ~ReproductorAudio() { detener(); }
};
