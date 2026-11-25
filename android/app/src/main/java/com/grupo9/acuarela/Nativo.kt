package com.grupo9.acuarela

import android.graphics.Bitmap

/** Funciones implementadas en C++ (app/src/main/cpp/Nativo.cpp). */
object Nativo {
    init {
        System.loadLibrary("acuarela")
    }

    /**
     * Carga los cuadros, abre Bad Apple y activa OpenCL. carpetaCache guarda los
     * kernels OpenCL compilados entre ejecuciones; rutaCsv recibe las métricas de
     * cada frame. Devuelve el estado de OpenCL o "ERROR: ...".
     */
    external fun iniciar(rutaBadApple: String, rutaCuadroGris: String, rutaCuadroColor: String,
                         carpetaCache: String, rutaCsv: String): String

    /** 0 = cámara trasera, 1 = frontal. */
    external fun abrirCamara(indice: Int): Boolean

    external fun cerrarCamara()

    /**
     * Procesa un frame y lo dibuja en los dos Bitmap.
     * parametros = [modo, etapa, bilateralD, pasadas, kernelDoG, bordes x10, grano x100, lapiz x10]
     * Devuelve [msCPU, msGPU, emaCPU, emaGPU, fps, ramMB, cpu%] o null si no hubo frame.
     */
    external fun procesar(original: Bitmap, resultado: Bitmap, parametros: IntArray, posicionAudioMs: Long): DoubleArray?
}
