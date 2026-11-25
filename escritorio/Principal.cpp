// =============================================================================
//  Proyecto Integrador - Parte I (Interciclo)
//  "Bad Apple en Acuarela": la cámara se pinta como acuarela y el video
//  Bad Apple!! decide qué paleta (Gray Tree / La Ciotat) usa cada píxel.
//  Benchmark CPU (cv::Mat) vs GPU (cv::UMat + OpenCL / T-API) y visualización
//  3D con VTK (cv::viz).
//
//  Uso:   ./vision.bin               -> cámara 0
//         ./vision.bin 1             -> cámara 1
//         ./vision.bin video.mp4     -> archivo de video como "cámara"
//         --sin-audio                -> no reproducir la canción
//
//  Teclas (con la ventana "Controles" activa):
//         m       -> modo CPU / GPU / AMBOS
//         espacio -> rotación del cubo on/off
//         q, ESC  -> salir
//  Las métricas de cada frame se guardan en metricas.csv
// =============================================================================
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/viz.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>

#include "Audio.hpp"
#include "Efecto.hpp"
#include "Metricas.hpp"

using namespace std;
using namespace cv;

#ifndef RUTA_RECURSOS
#define RUTA_RECURSOS "../recursos"
#endif

static const char* VENTANA_CONTROLES = "Controles";
static const char* NOMBRES_MODO[] = {"CPU (cv::Mat)", "GPU (cv::UMat + OpenCL)", "AMBOS (comparativa)"};
static const Size TAM(640, 480);

// Convierte cualquier etapa (Mat o UMat, gris o color) en Mat BGR para la textura
template <typename M>
static void aTextura(const M& src, Mat& dst) {
    if (src.channels() == 1) cvtColor(src, dst, COLOR_GRAY2BGR);
    else                     src.copyTo(dst);
    if (dst.size() != TAM) resize(dst, dst, TAM);
}

// Nombre de la etapa en la esquina superior izquierda de la textura
static void etiquetar(Mat& img, const string& texto) {
    int base = 0;
    Size t = getTextSize(texto, FONT_HERSHEY_SIMPLEX, 0.8, 2, &base);
    rectangle(img, Point(0, 0), Point(t.width + 20, t.height + base + 16), Scalar(0, 0, 0), FILLED);
    putText(img, texto, Point(10, t.height + 8), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2, LINE_AA);
}

// Valor de un trackbar que solo se aplica cuando deja de moverse 300 ms.
// Cada tamaño de kernel nuevo obliga a OpenCL a compilar otro programa para la
// GPU; si se aplicara cada valor intermedio al arrastrar, se compilarían
// decenas de kernels seguidos y el video se congelaría.
struct ValorEstable {
    int valor = -1, candidato = -1;
    chrono::steady_clock::time_point desde;
    int leer(int actual) {
        auto ahora = chrono::steady_clock::now();
        if (actual != candidato) { candidato = actual; desde = ahora; }
        if (valor < 0 || (candidato != valor && ahora - desde >= chrono::milliseconds(300)))
            valor = candidato;
        return valor;
    }
};

static Mat crearPanelAyuda() {
    Mat panel(110, 560, CV_8UC3, Scalar(40, 40, 40));
    putText(panel, "m: modo CPU / GPU / AMBOS", Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    putText(panel, "espacio: rotar cubo on/off", Point(10, 55), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    putText(panel, "q / ESC: salir  (raton en VTK: girar/zoom)", Point(10, 85), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    return panel;
}

static Mat leerImagen(const string& ruta) {
    Mat img = imread(ruta);
    if (img.empty()) { cerr << "No se pudo leer: " << ruta << endl; exit(-1); }
    return img;
}

int main(int argc, char** argv) {
    const string raiz = RUTA_RECURSOS;
    const string rutaBadApple = raiz + "/video/bad-apple.mp4";
    bool conAudio = true;

    // ------------------------------------------------------- Fuente de video
    VideoCapture video;
    bool esArchivo = false;
    string fuente = "0";
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--sin-audio") conAudio = false;
        else fuente = a;
    }
    if (all_of(fuente.begin(), fuente.end(), ::isdigit)) video.open(stoi(fuente));
    else { video.open(fuente); esArchivo = true; }
    if (!video.isOpened()) { cerr << "No se pudo abrir la fuente de video\n"; return -1; }
    video.set(CAP_PROP_FRAME_WIDTH, TAM.width);
    video.set(CAP_PROP_FRAME_HEIGHT, TAM.height);

    // ------------------------------------------------ Bad Apple y paletas
    VideoCapture badApple(rutaBadApple);
    if (!badApple.isOpened()) { cerr << "No se pudo abrir " << rutaBadApple << endl; return -1; }
    const double fpsBA = badApple.get(CAP_PROP_FPS) > 0 ? badApple.get(CAP_PROP_FPS) : 30.0;

    Mat cuadroGris  = leerImagen(raiz + "/cuadros/gray-tree.jpeg");
    Mat cuadroColor = leerImagen(raiz + "/cuadros/little-bay-at-la-ciotat.jpeg");

    // ---------------------------------------------------------------- OpenCL
    if (!ocl::haveOpenCL()) cerr << "AVISO: OpenCL no disponible, el modo GPU usará la CPU\n";
    ocl::setUseOpenCL(true);
    cout << "Dispositivo OpenCL: " << ocl::Device::getDefault().name() << endl;

    // ------------------------------------------------------------- Controles
    ParametrosAcuarela p;
    namedWindow(VENTANA_CONTROLES, WINDOW_AUTOSIZE);
    struct Barra { const char* nombre; int max; int valor; };
    const Barra barras[] = {
        {"Modo 0CPU 1GPU 2AMBOS", 2,  2},
        {"Bilateral d",           31, p.bilateralD},
        {"Pasadas bilateral",     6,  p.pasadasBilateral},
        {"Kernel DoG (n)",        61, p.kernelDoG},
        {"Bordes x10",            60, int(p.fBorde * 10)},
        {"Grano x100",            150, int(p.fGrano * 100)},
        {"Turbulencia x100",      100, int(p.fTurbulencia * 100)},
        {"Lapiz x10",             40, int(p.fLapiz * 10)},
        {"Contraste paletas",     50, int(p.contrasteL)},
        {"Croma color x100",      200, int(p.cromaColor * 100)},
        {"Rotar cubo",            1,  1},
    };
    for (const Barra& b : barras) createTrackbar(b.nombre, VENTANA_CONTROLES, nullptr, b.max);
    for (const Barra& b : barras) setTrackbarPos(b.nombre, VENTANA_CONTROLES, b.valor);
    imshow(VENTANA_CONTROLES, crearPanelAyuda());   // solo texto de ayuda (no video)
    auto barra = [](const char* n) { return getTrackbarPos(n, VENTANA_CONTROLES); };
    ValorEstable estableD, estableDoG;   // solo los controles que cambian el tamaño de un kernel

    // ------------------------------------------------------- Escena 3D (VTK)
    // Cubo de 4 caras WImage3D:
    //   frente: resultado | derecha: cámara | atrás: Bad Apple | izquierda: manchas
    viz::Viz3d ventana3D("Bad Apple en Acuarela - VTK (cv::viz)");
    ventana3D.setBackgroundColor(viz::Color(20, 20, 20), viz::Color(70, 50, 30));
    ventana3D.setWindowSize(Size(1024, 768));

    const double L = 1.0;
    const Size2d tamCara(L, L * 0.75);
    Mat negro(TAM, CV_8UC3, Scalar::all(0));
    viz::WImage3D caraFrente(negro, tamCara), caraDerecha(negro, tamCara),
                  caraAtras(negro, tamCara),  caraIzquierda(negro, tamCara);

    const Affine3d poseFrente   (Vec3d(0,  0,         0), Vec3d( 0,     0,  L / 2));
    const Affine3d poseDerecha  (Vec3d(0,  CV_PI / 2, 0), Vec3d( L / 2, 0,  0));
    const Affine3d poseAtras    (Vec3d(0,  CV_PI,     0), Vec3d( 0,     0, -L / 2));
    const Affine3d poseIzquierda(Vec3d(0, -CV_PI / 2, 0), Vec3d(-L / 2, 0,  0));

    ventana3D.showWidget("frente",    caraFrente,    poseFrente);
    ventana3D.showWidget("derecha",   caraDerecha,   poseDerecha);
    ventana3D.showWidget("atras",     caraAtras,     poseAtras);
    ventana3D.showWidget("izquierda", caraIzquierda, poseIzquierda);

    viz::WText textoMetricas("Iniciando...", Point(10, 10), 15, viz::Color::white());
    ventana3D.showWidget("metricas", textoMetricas);
    ventana3D.setViewerPose(viz::makeCameraPose(Vec3d(0, 0.6, 2.4), Vec3d(0, 0, 0), Vec3d(0, 1, 0)));

    // ---------------------------------------------------- Estado del pipeline
    RecursosAcuarela<Mat>  recCPU;  EtapasAcuarela<Mat>  etCPU;
    RecursosAcuarela<UMat> recGPU;  EtapasAcuarela<UMat> etGPU;
    recCPU.cargarPaletas(cuadroGris, cuadroColor);
    recGPU.cargarPaletas(cuadroGris, cuadroColor);

    UMat frameGPU, badAppleGPU;
    Mat frame, frameBA, texFinal, texCamara, texBadApple, texManchas;

    MonitorGPU monitorGPU;
    MedidorCPU medidorCPU;
    ReproductorAudio audio;

    ofstream csv("metricas.csv");
    csv << "tiempo_s,modo,bilateral_d,pasadas,kernel_dog,ms_cpu,ms_gpu,fps,ram_mb,cpu_pct,gpu_util_pct,gpu_mem_proceso_mb\n";

    using reloj = chrono::steady_clock;
    const auto inicio = reloj::now();
    auto tAnterior = reloj::now();
    double fps = 0, emaCPU = 0, emaGPU = 0;
    const double alfa = 0.1;
    double angulo = 0;
    int modoAnterior = -1;
    long framesProcesados = 0;

    // Sincronización Bad Apple <-> audio: el frame que toca se calcula con el
    // tiempo transcurrido desde que empezó la canción.
    badApple.read(frameBA);
    long idxBA = 1;
    auto inicioBA = reloj::now();
    if (conAudio) audio.iniciar(rutaBadApple);

    // ================================================================= Bucle
    while (!ventana3D.wasStopped()) {
        video >> frame;
        if (frame.empty()) {
            if (esArchivo) { video.set(CAP_PROP_POS_FRAMES, 0); continue; }
            break;
        }
        if (frame.size() != TAM) resize(frame, frame, TAM);
        if (!esArchivo) flip(frame, frame, 1);   // efecto espejo para la webcam

        // ---- Frame de Bad Apple que corresponde al tiempo actual
        long objetivo = long(chrono::duration<double>(reloj::now() - inicioBA).count() * fpsBA);
        while (idxBA <= objetivo) {
            Mat siguiente;
            if (!badApple.read(siguiente)) {      // terminó: reinicia video y canción
                badApple.set(CAP_PROP_POS_FRAMES, 0);
                badApple.read(siguiente);
                idxBA = 0;
                inicioBA = reloj::now();
                if (conAudio) audio.iniciar(rutaBadApple);
                frameBA = siguiente;
                break;
            }
            frameBA = siguiente;
            idxBA++;
        }

        // ---- Lectura de controles
        int modo           = barra("Modo 0CPU 1GPU 2AMBOS");
        p.bilateralD       = max(1, estableD.leer(barra("Bilateral d")));
        p.pasadasBilateral = max(1, barra("Pasadas bilateral"));
        p.kernelDoG        = hacerImpar(estableDoG.leer(barra("Kernel DoG (n)")), 3);
        p.fBorde           = barra("Bordes x10") / 10.0;
        p.fGrano           = barra("Grano x100") / 100.0;
        p.fTurbulencia     = barra("Turbulencia x100") / 100.0;
        p.fLapiz           = barra("Lapiz x10") / 10.0;
        p.contrasteL       = barra("Contraste paletas");
        p.cromaColor       = barra("Croma color x100") / 100.0;
        bool rotar         = barra("Rotar cubo") == 1;

        if (modo != modoAnterior) { emaCPU = emaGPU = 0; modoAnterior = modo; }

        recCPU.actualizar(p, TAM);
        recGPU.actualizar(p, TAM);

        // ---- Pipeline CPU (cv::Mat)
        double msCPU = -1, msGPU = -1;
        if (modo == 0 || modo == 2) {
            ocl::setUseOpenCL(false);
            auto t0 = reloj::now();
            aplicarAcuarela(frame, frameBA, recCPU, p, etCPU);
            msCPU = chrono::duration<double, milli>(reloj::now() - t0).count();
            emaCPU = (emaCPU == 0) ? msCPU : (1 - alfa) * emaCPU + alfa * msCPU;
        }

        // ---- Pipeline GPU (cv::UMat + OpenCL). Incluye subir los dos frames a
        //      la VRAM y ocl::finish() para medir el tiempo real de la GPU.
        if (modo == 1 || modo == 2) {
            ocl::setUseOpenCL(true);
            auto t0 = reloj::now();
            frame.copyTo(frameGPU);
            frameBA.copyTo(badAppleGPU);
            aplicarAcuarela(frameGPU, badAppleGPU, recGPU, p, etGPU);
            ocl::finish();
            msGPU = chrono::duration<double, milli>(reloj::now() - t0).count();
            emaGPU = (emaGPU == 0) ? msGPU : (1 - alfa) * emaGPU + alfa * msGPU;
        }

        // ---- Texturas de las caras del cubo
        aTextura(frame, texCamara);
        aTextura(frameBA, texBadApple);
        if (modo == 0) { aTextura(etCPU.final, texFinal); aTextura(etCPU.manchas, texManchas); }
        else           { aTextura(etGPU.final, texFinal); aTextura(etGPU.manchas, texManchas); }
        etiquetar(texFinal,    "Resultado");
        etiquetar(texCamara,   "Camara");
        etiquetar(texBadApple, "Bad Apple original");
        etiquetar(texManchas,  "Manchas (abstraccion)");
        // WImage3D muestra la textura reflejada vista desde fuera del cubo
        for (Mat* t : {&texFinal, &texCamara, &texBadApple, &texManchas}) flip(*t, *t, 1);
        caraFrente.setImage(texFinal);
        caraDerecha.setImage(texCamara);
        caraAtras.setImage(texBadApple);
        caraIzquierda.setImage(texManchas);

        // ---- Rotación del cubo alrededor del eje Y
        if (rotar) angulo += 0.01;
        Affine3d giro(Vec3d(0, angulo, 0), Vec3d(0, 0, 0));
        ventana3D.setWidgetPose("frente",    giro * poseFrente);
        ventana3D.setWidgetPose("derecha",   giro * poseDerecha);
        ventana3D.setWidgetPose("atras",     giro * poseAtras);
        ventana3D.setWidgetPose("izquierda", giro * poseIzquierda);

        // ---- Métricas
        auto ahora = reloj::now();
        double dt = chrono::duration<double>(ahora - tAnterior).count();
        tAnterior = ahora;
        fps = (fps == 0) ? 1.0 / dt : 0.9 * fps + 0.1 * (1.0 / dt);
        double ram = ramProcesoMB();
        double cpu = medidorCPU.porcentaje();
        int gpuUtil = monitorGPU.utilizacion, gpuMem = monitorGPU.memoriaProceso;

        char buf[1024], tiempos[256];
        if (modo == 0)      snprintf(tiempos, sizeof tiempos, "CPU: %.2f ms", emaCPU);
        else if (modo == 1) snprintf(tiempos, sizeof tiempos, "GPU (OpenCL): %.2f ms", emaGPU);
        else                snprintf(tiempos, sizeof tiempos, "CPU: %.2f ms  |  GPU (OpenCL): %.2f ms  |  Aceleracion: x%.1f",
                                     emaCPU, emaGPU, emaGPU > 0 ? emaCPU / emaGPU : 0.0);
        snprintf(buf, sizeof buf,
                 "Modo: %s\n%s\nFPS: %.1f   RAM proceso: %.0f MB   CPU proceso: %.0f %%\n"
                 "GPU %s: uso %d %%  VRAM proceso %d MB\n"
                 "Bilateral d: %d x%d pasadas   Kernel DoG: %dx%d",
                 NOMBRES_MODO[modo], tiempos, fps, ram, cpu,
                 ocl::Device::getDefault().name().c_str(), gpuUtil, gpuMem,
                 p.bilateralD, p.pasadasBilateral, p.kernelDoG, p.kernelDoG);
        textoMetricas.setText(buf);

        csv << chrono::duration<double>(ahora - inicio).count() << ',' << modo << ','
            << p.bilateralD << ',' << p.pasadasBilateral << ',' << p.kernelDoG << ','
            << msCPU << ',' << msGPU << ',' << fps << ',' << ram << ',' << cpu << ','
            << gpuUtil << ',' << gpuMem << '\n';
        if (++framesProcesados % 30 == 0) csv.flush();   // no perder datos si se cierra a la fuerza

        // ---- Eventos: VTK y ventana de controles
        ventana3D.spinOnce(1, true);
        int tecla = waitKey(1);
        if (tecla == 'q' || tecla == 27) break;
        if (tecla == 'm') setTrackbarPos("Modo 0CPU 1GPU 2AMBOS", VENTANA_CONTROLES, (modo + 1) % 3);
        if (tecla == ' ') setTrackbarPos("Rotar cubo", VENTANA_CONTROLES, rotar ? 0 : 1);
    }

    audio.detener();
    video.release();
    destroyAllWindows();
    return 0;
}
