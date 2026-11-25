package com.grupo9.acuarela

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.MediaPlayer
import android.os.Bundle
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.RadioGroup
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import java.io.File
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit

/**
 * Interfaz de la app. El procesamiento ocurre en C++ (Nativo.cpp); aquí solo se
 * leen los controles, se muestran los Bitmap y se reproduce el audio.
 */
class MainActivity : Activity() {

    // Sliders de los kernels (mismos rangos y valores iniciales que en escritorio).
    // alSoltar = el valor se aplica solo al soltar el slider: cada tamaño de kernel
    // nuevo obliga a OpenCL a compilar otro programa para la GPU, y aplicar cada
    // valor intermedio al arrastrar compilaría decenas de kernels seguidos.
    private class Control(val nombre: String, val min: Int, val max: Int,
                          @Volatile var valor: Int, val alSoltar: Boolean = false)

    private val controles = listOf(
        Control("Bilateral d", 1, 31, 9, alSoltar = true),
        Control("Pasadas bilateral", 1, 6, 3),
        Control("Kernel DoG (n)", 3, 61, 7, alSoltar = true),
        Control("Bordes x10", 0, 60, 32),
        Control("Grano x100", 0, 150, 55),
        Control("Lápiz x10", 0, 40, 12),
    )
    private val etapas = arrayOf(
        "Resultado final", "Máscara Bad Apple", "Bad Apple original",
        "Manchas (abstracción)", "Borde húmedo (DoG)", "Lápiz (DoG)",
    )
    private val nombresModo = arrayOf("CPU (cv::Mat)", "GPU (cv::UMat + OpenCL)", "AMBOS")

    @Volatile private var modo = 2
    @Volatile private var etapa = 0
    @Volatile private var corriendo = false
    private var indiceCamara = 0
    private var estadoOpenCL = ""
    private var listo = false

    private lateinit var imgOriginal: ImageView
    private lateinit var imgResultado: ImageView
    private lateinit var txtMetricas: TextView
    private var hilo: Thread? = null
    private var audio: MediaPlayer? = null

    // Doble buffer: C++ escribe en un par mientras la pantalla muestra el otro
    private val bitmaps = Array(2) {
        Pair(Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888),
             Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        imgOriginal = findViewById(R.id.imgOriginal)
        imgResultado = findViewById(R.id.imgResultado)
        txtMetricas = findViewById(R.id.txtMetricas)
        crearControles()

        // Los recursos van dentro del APK; OpenCV necesita rutas de archivo reales
        val video = copiarAsset("bad-apple.mp4")
        val gris = copiarAsset("gray-tree.jpeg")
        val color = copiarAsset("little-bay-at-la-ciotat.jpeg")
        // Un CSV por sesión en /sdcard/Android/data/com.grupo9.acuarela/files/ (se baja con adb pull)
        val fecha = java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.US).format(java.util.Date())
        val csv = File(getExternalFilesDir(null) ?: filesDir, "metricas_$fecha.csv")
        estadoOpenCL = Nativo.iniciar(video.path, gris.path, color.path, cacheDir.path, csv.path)
        listo = !estadoOpenCL.startsWith("ERROR")
        txtMetricas.text = estadoOpenCL

        audio = MediaPlayer().apply {
            setDataSource(video.path)   // el audio sale del mismo mp4
            isLooping = true
            prepare()
        }
    }

    override fun onResume() {
        super.onResume()
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            arrancar()
        } else {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 1)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) arrancar()
        else txtMetricas.text = "Se necesita el permiso de cámara"
    }

    override fun onPause() {
        detener()
        super.onPause()
    }

    override fun onDestroy() {
        audio?.release()
        super.onDestroy()
    }

    // ------------------------------------------------------------------ bucle
    private fun arrancar() {
        if (!listo || corriendo) return
        if (!Nativo.abrirCamara(indiceCamara)) {
            txtMetricas.text = "No se pudo abrir la cámara $indiceCamara\n$estadoOpenCL"
            return
        }
        audio?.start()
        corriendo = true
        hilo = Thread { bucle() }.apply { start() }
    }

    private fun detener() {
        corriendo = false
        hilo?.join()
        hilo = null
        Nativo.cerrarCamara()
        audio?.takeIf { it.isPlaying }?.pause()
    }

    private fun bucle() {
        var trasero = 0
        val mostrado = Semaphore(0)
        while (corriendo) {
            val (bmpOriginal, bmpResultado) = bitmaps[trasero]
            val parametros = intArrayOf(modo, etapa) + controles.map { it.valor }
            val posicion = audio?.currentPosition?.toLong() ?: 0L
            val m = Nativo.procesar(bmpOriginal, bmpResultado, parametros, posicion)
            if (m == null) { Thread.sleep(5); continue }

            runOnUiThread {
                imgOriginal.setImageBitmap(bmpOriginal)
                imgResultado.setImageBitmap(bmpResultado)
                txtMetricas.text = textoMetricas(m)
                mostrado.release()
            }
            // Espera a que la pantalla use este par (con tiempo límite: si la app se
            // está cerrando, el hilo de la interfaz está en detener() y no lo dibuja)
            while (corriendo && !mostrado.tryAcquire(50, TimeUnit.MILLISECONDS)) { }
            trasero = 1 - trasero    // el siguiente frame se escribe en el otro par
        }
    }

    // m = [msCPU, msGPU, emaCPU, emaGPU, fps, ramMB, cpu%]
    private fun textoMetricas(m: DoubleArray): String {
        val tiempos = when (modo) {
            0 -> "CPU: %.2f ms".format(m[2])
            1 -> "GPU (OpenCL): %.2f ms".format(m[3])
            else -> "CPU: %.2f ms | GPU (OpenCL): %.2f ms | Aceleración: x%.1f"
                .format(m[2], m[3], if (m[3] > 0) m[2] / m[3] else 0.0)
        }
        val k = controles.joinToString("  ") { "${it.nombre}=${it.valor}" }
        return "Modo: ${nombresModo[modo]}\n$tiempos\n" +
            "FPS: %.1f   RAM proceso: %.0f MB   CPU proceso: %.0f %%\n".format(m[4], m[5], m[6]) +
            "$estadoOpenCL\n$k"
    }

    // -------------------------------------------------------------- controles
    private fun crearControles() {
        findViewById<RadioGroup>(R.id.grupoModo).setOnCheckedChangeListener { _, id ->
            modo = when (id) { R.id.modoCPU -> 0; R.id.modoGPU -> 1; else -> 2 }
        }

        val txtEtapa = findViewById<TextView>(R.id.txtEtapa)
        findViewById<Spinner>(R.id.spinnerEtapa).apply {
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, etapas)
            onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) {
                    etapa = pos
                    txtEtapa.text = etapas[pos]
                }
                override fun onNothingSelected(p: AdapterView<*>?) {}
            }
        }

        findViewById<Button>(R.id.btnCamara).setOnClickListener {
            detener()
            indiceCamara = 1 - indiceCamara
            arrancar()
        }

        // SeekBar empieza en 0, por eso se desplaza con el mínimo de cada control
        val panel = findViewById<LinearLayout>(R.id.panelControles)
        for (c in controles) {
            val etiqueta = TextView(this).apply {
                text = "${c.nombre}: ${c.valor}"
                setTextColor(0xFFFFFFFF.toInt())
                setPadding(0, 12, 0, 0)
            }
            val barra = SeekBar(this).apply {
                max = c.max - c.min
                progress = c.valor - c.min
                setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                    override fun onProgressChanged(s: SeekBar?, v: Int, deUsuario: Boolean) {
                        etiqueta.text = "${c.nombre}: ${v + c.min}"
                        if (!c.alSoltar) c.valor = v + c.min
                    }
                    override fun onStartTrackingTouch(s: SeekBar?) {}
                    override fun onStopTrackingTouch(s: SeekBar?) {
                        c.valor = (s?.progress ?: 0) + c.min
                    }
                })
            }
            panel.addView(etiqueta)
            panel.addView(barra)
        }
    }

    // Copia un archivo de assets/ a la carpeta privada de la app (solo la primera vez)
    private fun copiarAsset(nombre: String): File {
        val destino = File(filesDir, nombre)
        if (!destino.exists()) {
            assets.open(nombre).use { entrada ->
                destino.outputStream().use { salida -> entrada.copyTo(salida) }
            }
        }
        return destino
    }
}
