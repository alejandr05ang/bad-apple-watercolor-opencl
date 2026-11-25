// =============================================================================
//  Nativo.cpp - Puente JNI entre la app Android (Kotlin) y el efecto en C++
//
//  Todo el procesamiento ocurre aquí, en C++ nativo:
//    - Cámara      : cv::VideoCapture con el backend NDK Camera2 (CAP_ANDROID)
//    - Bad Apple   : cv::VideoCapture sobre el mp4 (backend MediaNDK)
//    - Efecto      : aplicarAcuarela() de comun/Efecto.hpp, el MISMO código que
//                    en escritorio, instanciado con cv::Mat (CPU) y cv::UMat (GPU)
//    - Métricas    : ms con <chrono>, RAM (/proc/self/status) y CPU del proceso
//  Kotlin solo dibuja los Bitmap, lee los controles y reproduce el audio.
// =============================================================================
#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <chrono>
#include <cstdlib>   // setenv
#include <fstream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "Efecto.hpp"
#include "Metricas.hpp"

#define LOG(...) __android_log_print(ANDROID_LOG_INFO, "Acuarela", __VA_ARGS__)

using namespace cv;
using reloj = std::chrono::steady_clock;

static const Size TAM(640, 480);   // misma resolución de trabajo que en escritorio

// -----------------------------------------------------------------------------
// Estado de la aplicación (una sola instancia)
// -----------------------------------------------------------------------------
struct Estado {
    VideoCapture camara;
    int indiceCamara = 0;

    VideoCapture badApple;
    std::string rutaBadApple;
    double fpsBA = 30;
    long idxBA = 0;

    Mat frame, frameBA, vista;
    UMat frameGPU, badAppleGPU;

    RecursosAcuarela<Mat>  recCPU;
    RecursosAcuarela<UMat> recGPU;
    EtapasAcuarela<Mat>    etCPU;
    EtapasAcuarela<UMat>   etGPU;

    double emaCPU = 0, emaGPU = 0, fps = 0;
    std::ofstream csv;                       // métricas para las gráficas del informe
    reloj::time_point inicio = reloj::now();
    long frames = 0;
    int modoAnterior = -1;
    reloj::time_point tAnterior = reloj::now();
    MedidorCPU medidorCPU;
};
static Estado E;

// Los backends de Android pueden entregar 1 o 4 canales: el efecto espera BGR
static void aBGR(Mat& img, const char* origen) {
    static int avisos = 0;
    if (avisos < 4 && img.channels() != 3) {
        LOG("%s entrega %dx%d con %d canales", origen, img.cols, img.rows, img.channels());
        avisos++;
    }
    if (img.channels() == 4)      cvtColor(img, img, COLOR_BGRA2BGR);
    else if (img.channels() == 1) cvtColor(img, img, COLOR_GRAY2BGR);
}

// Abre (o reabre desde el inicio) el video de Bad Apple
static bool abrirBadApple() {
    E.badApple.release();
    E.badApple.open(E.rutaBadApple, CAP_ANDROID);
    if (!E.badApple.isOpened()) return false;
    double f = E.badApple.get(CAP_PROP_FPS);
    E.fpsBA = f > 0 ? f : 30.0;
    E.idxBA = 0;
    if (!E.badApple.read(E.frameBA)) return false;
    aBGR(E.frameBA, "Bad Apple");
    return true;
}

// Escribe una imagen BGR en un Bitmap ARGB_8888 de Android (memoria compartida)
static void escribirBitmap(JNIEnv* env, jobject bitmap, const Mat& bgr) {
    AndroidBitmapInfo info;
    void* pixeles = nullptr;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0 ||
        info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixeles) < 0) return;
    Mat destino(info.height, info.width, CV_8UC4, pixeles, info.stride);  // sin copia
    if (bgr.size() != destino.size()) {
        Mat tmp;
        resize(bgr, tmp, destino.size());
        cvtColor(tmp, destino, COLOR_BGR2RGBA);
    } else {
        cvtColor(bgr, destino, COLOR_BGR2RGBA);
    }
    AndroidBitmap_unlockPixels(env, bitmap);
}

// Etapa que se muestra a la derecha: resultado final o un paso intermedio.
// Las etapas en coma flotante [0,1] se escalan a [0,255] para verlas.
template <typename M>
static void etapaAVista(const EtapasAcuarela<M>& e, int etapa, Mat& vista) {
    M tmp;
    switch (etapa) {
        case 1:  e.alfa1.convertTo(tmp, CV_8U, 255); break;   // máscara Bad Apple
        case 3:  e.manchas.copyTo(tmp); break;                 // abstracción
        case 4:  e.borde.convertTo(tmp, CV_8U, 255); break;   // borde húmedo (DoG)
        case 5:  e.lapiz.convertTo(tmp, CV_8U, 255); break;   // lápiz (DoG)
        default: e.final.copyTo(tmp); break;                   // resultado
    }
    tmp.copyTo(vista);   // con UMat: descarga de la VRAM a la RAM
    if (vista.channels() == 1) cvtColor(vista, vista, COLOR_GRAY2BGR);
}

extern "C" {

// -----------------------------------------------------------------------------
// iniciar(rutaBadApple, rutaCuadroGris, rutaCuadroColor, carpetaCache, rutaCsv) -> estado
// -----------------------------------------------------------------------------
JNIEXPORT jstring JNICALL
Java_com_grupo9_acuarela_Nativo_iniciar(JNIEnv* env, jobject, jstring jBA, jstring jGris,
                                        jstring jColor, jstring jCache, jstring jCsv) {
    auto texto = [&](jstring s) {
        const char* c = env->GetStringUTFChars(s, nullptr);
        std::string r(c);
        env->ReleaseStringUTFChars(s, c);
        return r;
    };
    E.rutaBadApple = texto(jBA);
    Mat cuadroGris  = imread(texto(jGris));
    Mat cuadroColor = imread(texto(jColor));
    if (cuadroGris.empty() || cuadroColor.empty())
        return env->NewStringUTF("ERROR: no se pudieron leer los cuadros");
    if (!abrirBadApple())
        return env->NewStringUTF("ERROR: no se pudo abrir bad-apple.mp4");

    E.recCPU.cargarPaletas(cuadroGris, cuadroColor);
    E.recGPU.cargarPaletas(cuadroGris, cuadroColor);

    // ---- OpenCL: se carga libOpenCL.so del fabricante en tiempo de ejecución.
    // En Android OpenCV no tiene carpeta por defecto para guardar los kernels ya
    // compilados: sin esto, cada vez que se abre la app se recompilan todos.
    setenv("OPENCV_OPENCL_CACHE_DIR", texto(jCache).c_str(), 1);
    ocl::setUseOpenCL(true);
    std::string estado;
    if (ocl::haveOpenCL() && ocl::useOpenCL()) {
        const ocl::Device& d = ocl::Device::getDefault();
        estado = "OpenCL: " + d.name() + " (" + d.vendorName() + ", " + d.version() + ")";
    } else {
        estado = "OpenCL NO disponible: el modo GPU se ejecuta en CPU";
    }
    LOG("%s", estado.c_str());

    // ---- CSV con las mismas columnas que la app de escritorio (graficar_metricas.py).
    //      La GPU Mali no expone uso ni memoria a las apps: esas columnas van en -1.
    // Si la actividad se recrea en el mismo proceso (salir con "atrás" y volver),
    // el archivo anterior sigue abierto: hay que cerrarlo o open() falla en silencio.
    if (E.csv.is_open()) E.csv.close();
    E.csv.clear();
    E.csv.open(texto(jCsv));
    LOG("CSV: %s (%s)", texto(jCsv).c_str(), E.csv.is_open() ? "ok" : "ERROR");
    E.csv << "tiempo_s,modo,bilateral_d,pasadas,kernel_dog,ms_cpu,ms_gpu,fps,ram_mb,cpu_pct,gpu_util_pct,gpu_mem_proceso_mb\n";
    E.inicio = reloj::now();
    return env->NewStringUTF(estado.c_str());
}

// -----------------------------------------------------------------------------
// Cámara: 0 = trasera, 1 = frontal (orden de Camera2 en el Pixel)
// -----------------------------------------------------------------------------
JNIEXPORT jboolean JNICALL
Java_com_grupo9_acuarela_Nativo_abrirCamara(JNIEnv*, jobject, jint indice) {
    E.camara.release();
    E.indiceCamara = indice;
    // Sin CAP_PROP_FOURCC el backend entrega YUV crudo (1 canal, alto x 1.5)
    E.camara.open(indice, CAP_ANDROID, {CAP_PROP_FRAME_WIDTH, TAM.width,
                                        CAP_PROP_FRAME_HEIGHT, TAM.height,
                                        CAP_PROP_FOURCC, VideoWriter::fourcc('B', 'G', 'R', '3')});
    LOG("Camara %d abierta: %d", indice, (int)E.camara.isOpened());
    return E.camara.isOpened();
}

JNIEXPORT void JNICALL
Java_com_grupo9_acuarela_Nativo_cerrarCamara(JNIEnv*, jobject) {
    E.camara.release();
    E.csv.flush();
}

// -----------------------------------------------------------------------------
// procesar(bmpOriginal, bmpResultado, parametros, posicionAudioMs)
//   parametros = [modo, etapa, bilateralD, pasadas, kernelDoG,
//                 bordes x10, grano x100, lapiz x10]
//   devuelve   = [msCPU, msGPU, emaCPU, emaGPU, fps, ramMB, cpu%]  (o null)
// -----------------------------------------------------------------------------
JNIEXPORT jdoubleArray JNICALL
Java_com_grupo9_acuarela_Nativo_procesar(JNIEnv* env, jobject, jobject bmpOriginal,
                                         jobject bmpResultado, jintArray jParam, jlong posAudioMs) {
    // ---- Frame de la cámara
    if (!E.camara.isOpened() || !E.camara.read(E.frame) || E.frame.empty()) return nullptr;
    aBGR(E.frame, "Camara");
    if (E.frame.size() != TAM) resize(E.frame, E.frame, TAM);
    if (E.indiceCamara == 1) flip(E.frame, E.frame, 1);   // frontal: efecto espejo

    // ---- Frame de Bad Apple sincronizado con la posición del audio
    long objetivo = long(posAudioMs / 1000.0 * E.fpsBA);
    if (objetivo + E.fpsBA < E.idxBA) abrirBadApple();     // el audio volvió a empezar
    while (E.idxBA < objetivo) {
        Mat siguiente;
        if (!E.badApple.read(siguiente)) { abrirBadApple(); break; }
        aBGR(siguiente, "Bad Apple");
        E.frameBA = siguiente;
        E.idxBA++;
    }

    // ---- Controles de la interfaz
    jint q[8];
    env->GetIntArrayRegion(jParam, 0, 8, q);
    const int modo = q[0], etapa = q[1];
    ParametrosAcuarela p;
    p.bilateralD       = std::max(1, (int)q[2]);
    p.pasadasBilateral = std::max(1, (int)q[3]);
    p.kernelDoG        = hacerImpar(q[4], 3);
    p.fBorde           = q[5] / 10.0;
    p.fGrano           = q[6] / 100.0;
    p.fLapiz           = q[7] / 10.0;

    if (modo != E.modoAnterior) { E.emaCPU = E.emaGPU = 0; E.modoAnterior = modo; }
    E.recCPU.actualizar(p, TAM);
    E.recGPU.actualizar(p, TAM);
    const double alfa = 0.1;   // suavizado exponencial de los ms mostrados

    // ---- Pipeline CPU (cv::Mat)
    double msCPU = -1, msGPU = -1;
    if (modo == 0 || modo == 2) {
        ocl::setUseOpenCL(false);
        auto t0 = reloj::now();
        aplicarAcuarela(E.frame, E.frameBA, E.recCPU, p, E.etCPU);
        msCPU = std::chrono::duration<double, std::milli>(reloj::now() - t0).count();
        E.emaCPU = (E.emaCPU == 0) ? msCPU : (1 - alfa) * E.emaCPU + alfa * msCPU;
    }

    // ---- Pipeline GPU (cv::UMat + OpenCL): incluye la subida de los frames
    //      a la memoria de la GPU y ocl::finish() para medir el tiempo real.
    if (modo == 1 || modo == 2) {
        ocl::setUseOpenCL(true);
        auto t0 = reloj::now();
        E.frame.copyTo(E.frameGPU);
        E.frameBA.copyTo(E.badAppleGPU);
        aplicarAcuarela(E.frameGPU, E.badAppleGPU, E.recGPU, p, E.etGPU);
        ocl::finish();
        msGPU = std::chrono::duration<double, std::milli>(reloj::now() - t0).count();
        E.emaGPU = (E.emaGPU == 0) ? msGPU : (1 - alfa) * E.emaGPU + alfa * msGPU;
    }

    // ---- Vistas: izquierda la cámara original, derecha la etapa elegida
    if (etapa == 2)      E.frameBA.copyTo(E.vista);           // Bad Apple original
    else if (modo == 0)  etapaAVista(E.etCPU, etapa, E.vista);
    else                 etapaAVista(E.etGPU, etapa, E.vista);
    escribirBitmap(env, bmpOriginal, E.frame);
    escribirBitmap(env, bmpResultado, E.vista);

    // ---- Métricas
    auto ahora = reloj::now();
    double dt = std::chrono::duration<double>(ahora - E.tAnterior).count();
    E.tAnterior = ahora;
    if (dt > 0) E.fps = (E.fps == 0) ? 1.0 / dt : 0.9 * E.fps + 0.1 * (1.0 / dt);

    double r[7] = {msCPU, msGPU, E.emaCPU, E.emaGPU, E.fps, ramProcesoMB(), E.medidorCPU.porcentaje()};
    if (E.csv.is_open()) {
        E.csv << std::chrono::duration<double>(ahora - E.inicio).count() << ',' << modo << ','
              << p.bilateralD << ',' << p.pasadasBilateral << ',' << p.kernelDoG << ','
              << msCPU << ',' << msGPU << ',' << E.fps << ',' << r[5] << ',' << r[6] << ",-1,-1\n";
        if (++E.frames % 30 == 0) E.csv.flush();
    }
    jdoubleArray salida = env->NewDoubleArray(7);
    env->SetDoubleArrayRegion(salida, 0, 7, r);
    return salida;
}

} // extern "C"
