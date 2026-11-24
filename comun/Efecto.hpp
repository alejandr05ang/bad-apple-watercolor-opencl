// =============================================================================
//  Efecto.hpp  -  "Bad Apple en Acuarela" (código compartido Escritorio / Android)
//
//  Cada frame de la cámara se pinta como ACUARELA. Un frame del video
//  "Bad Apple!!" (blanco y negro, 1 bit) se usa como MÁSCARA para elegir la
//  paleta de colores de cada píxel:
//      silueta negra  -> paleta de "Gray Tree"            (P. Mondrian, 1911)
//      fondo blanco   -> paleta de "Little Bay at La Ciotat" (G. Braque, 1907)
//
//  El pipeline se escribe UNA sola vez como plantilla y se instancia con:
//      cv::Mat   -> pipeline CPU tradicional
//      cv::UMat  -> pipeline GPU con la API Transparente (T-API / OpenCL)
//
//  Pipeline (por frame):
//   0. Máscara Bad Apple        : cvtColor + threshold + GaussianBlur (borde suave)
//   1. Abstracción (manchas)    : resize + bilateralFilter xN + medianBlur
//   2. Paletas de los cuadros   : transferencia de color de Reinhard en Lab
//   3. Lavados planos           : posterizado de la luminosidad (LUT)
//   4. Bordes que "sangran"     : remap con mapas de ruido suave
//   5. Borde de cada mancha     : KERNEL PROPIO DoG (filter2D)
//   5b. Dibujo previo a lápiz   : el mismo KERNEL PROPIO sobre la cámara ecualizada
//   6. Densidad de pigmento     : modelo de Bousseau et al. (2006)
//         C' = C - (C - C^2)(d - 1)
//       d = 1 + borde + grano de papel + turbulencia + lápiz
//
//  Referencias:
//   - A. Bousseau, M. Kaplan, J. Thollot, F. Sillion. "Interactive watercolor
//     rendering with temporal coherence and abstraction". NPAR 2006.
//   - E. Reinhard, M. Ashikhmin, B. Gooch, P. Shirley. "Color transfer between
//     images". IEEE Computer Graphics and Applications, 2001.
// =============================================================================
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

// -----------------------------------------------------------------------------
// Parámetros ajustables en tiempo real (trackbars / SeekBars)
// -----------------------------------------------------------------------------
struct ParametrosAcuarela {
    // 1. Abstracción (manchas de pigmento)  -> principal carga computacional
    int    bilateralD       = 9;     // diámetro de la ventana del bilateral
    int    pasadasBilateral = 3;     // número de aplicaciones del bilateral
    int    medianaManchas   = 5;     // ventana de la mediana (3 o 5: OpenCL solo implementa
                                     // esas; con 7+ OpenCV vuelve a la CPU y pierde la GPU)
    // 5. Kernel propio DoG
    int    kernelDoG        = 7;     // tamaño n x n (impar)  -> "saturar"
    double sigmaDoG         = 1.0;   // sigma de la gaussiana pequeña (la grande = 2*sigma)
    // 6. Pesos del modelo de densidad de pigmento
    double fBorde           = 3.2;   // oscurecimiento en el borde de las manchas
    double fGrano           = 0.55;  // granulado del papel
    double fTurbulencia     = 0.25;  // agua más/menos cargada
    double fLapiz           = 1.2;   // dibujo previo a lápiz
    double transparencia    = 0.12;  // cuánto se ve el papel debajo del color
    // 2-3. Paletas
    double contrasteL       = 22.0;  // Gray Tree baja L, La Ciotat sube L
    double cromaGris        = 0.6;   // escala del color (a,b) en la paleta gris
    double cromaColor       = 1.15;  // escala del color (a,b) en la paleta de color
    int    nivelesL         = 6;     // niveles de luminosidad (lavados planos)
};

// Marca opcional para perfilar el tiempo de cada etapa (vacía por defecto)
#ifndef MARCA_ETAPA
#define MARCA_ETAPA(nombre)
#endif

inline int hacerImpar(int v, int minimo = 1) {
    if (v < minimo) v = minimo;
    return (v % 2 == 0) ? v + 1 : v;
}

// -----------------------------------------------------------------------------
// Paleta = media y desviación estándar por canal en el espacio Lab
// -----------------------------------------------------------------------------
struct Paleta {
    cv::Scalar media, desv;
};

inline Paleta paletaDeImagen(const cv::Mat& bgr) {
    cv::Mat lab;
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
    Paleta p;
    cv::meanStdDev(lab, p.media, p.desv);
    return p;
}

// -----------------------------------------------------------------------------
// KERNEL PROPIO: Diferencia de Gaussianas (DoG)
//      K = G_s - G_{2s}        (n x n, separable: G = g * g^T)
//  Suma cero: en zonas planas la respuesta es 0; en un borde la respuesta es
//  positiva del lado claro y negativa del lado oscuro (aprox. -Laplaciano de G).
//  - |K * I| sobre las manchas      -> borde húmedo (acumulación de pigmento)
//  - K * I  < 0 sobre la cámara      -> trazo de lápiz en el lado oscuro
// -----------------------------------------------------------------------------
inline cv::Mat crearKernelDoG(int n, double sigma) {
    n = hacerImpar(n, 3);
    cv::Mat g1 = cv::getGaussianKernel(n, sigma,       CV_32F);
    cv::Mat g2 = cv::getGaussianKernel(n, 2.0 * sigma, CV_32F);
    cv::Mat K = g1 * g1.t() - g2 * g2.t();
    return K;
}

// LUT de 3 canales para Lab: posteriza L y deja a, b intactos
inline cv::Mat crearLUTLavados(int niveles) {
    niveles = std::max(2, niveles);
    const int paso = 256 / niveles;
    cv::Mat lut(1, 256, CV_8UC3);
    for (int v = 0; v < 256; v++) {
        int q = std::min(255, (v / paso) * paso + paso / 2);
        lut.at<cv::Vec3b>(0, v) = cv::Vec3b((uchar)q, (uchar)v, (uchar)v);
    }
    return lut;
}

// Ruido gaussiano suavizado y normalizado a [-escala, escala]
inline cv::Mat ruidoSuave(cv::Size tam, double sigma, double escala, cv::RNG& rng) {
    cv::Mat n(tam, CV_32F);
    rng.fill(n, cv::RNG::NORMAL, 0.0, 1.0);
    cv::GaussianBlur(n, n, cv::Size(0, 0), sigma);
    double mn, mx;
    cv::minMaxLoc(n, &mn, &mx);
    n *= escala / std::max(std::abs(mn), std::abs(mx));
    return n;
}

// -----------------------------------------------------------------------------
// Recursos precalculados. Se regeneran solo si cambian parámetros o tamaño.
// Las texturas se guardan en el tipo M para que, en el pipeline GPU, ya estén
// en la VRAM y no se suban en cada frame.
// -----------------------------------------------------------------------------
template <typename M>
struct RecursosAcuarela {
    Paleta grisBase, colorBase;   // estadísticas originales de los cuadros
    Paleta gris, color;           // con contraste / croma aplicados
    cv::Mat kernelDoG, lutLavados, elementoLapiz;
    M grano, turbulencia, textura, mapaX, mapaY;

    cv::Size tam;
    int n = -1, niveles = -1;
    double sigma = -1, fGrano = -1, fTurb = -1, contL = -1, cGris = -1, cColor = -1;

    void cargarPaletas(const cv::Mat& cuadroGris, const cv::Mat& cuadroColor) {
        grisBase  = paletaDeImagen(cuadroGris);
        colorBase = paletaDeImagen(cuadroColor);
        contL = -1;  // fuerza recalcular
    }

    void actualizar(const ParametrosAcuarela& p, cv::Size tamFrame) {
        if (p.kernelDoG != n || p.sigmaDoG != sigma) {
            n = p.kernelDoG; sigma = p.sigmaDoG;
            kernelDoG = crearKernelDoG(n, sigma);
        }
        if (p.nivelesL != niveles) {
            niveles = p.nivelesL;
            lutLavados = crearLUTLavados(niveles);
        }
        if (elementoLapiz.empty())
            elementoLapiz = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));

        if (p.contrasteL != contL || p.cromaGris != cGris || p.cromaColor != cColor) {
            contL = p.contrasteL; cGris = p.cromaGris; cColor = p.cromaColor;
            gris = grisBase;  color = colorBase;
            gris.media[0]  -= contL;      color.media[0] += contL;
            gris.desv[1]   *= cGris;      gris.desv[2]   *= cGris;
            color.desv[1]  *= cColor;     color.desv[2]  *= cColor;
        }

        bool nuevoTam = (tamFrame != tam);
        if (nuevoTam) {
            tam = tamFrame;
            cv::RNG rng(3);  // semilla fija: mismas texturas en CPU y GPU
            cv::Mat g = ruidoSuave(tam, 0.8, 1.0, rng) + 0.5 * ruidoSuave(tam, 2.0, 1.0, rng);
            cv::Mat t = ruidoSuave(tam, 25, 1.0, rng) + 0.5 * ruidoSuave(tam, 10, 1.0, rng);
            cv::Mat dx = ruidoSuave(tam, 6, 4.0, rng), dy = ruidoSuave(tam, 6, 4.0, rng);
            cv::Mat mx(tam, CV_32F), my(tam, CV_32F);
            for (int y = 0; y < tam.height; y++)
                for (int x = 0; x < tam.width; x++) {
                    mx.at<float>(y, x) = x + dx.at<float>(y, x);
                    my.at<float>(y, x) = y + dy.at<float>(y, x);
                }
            g.copyTo(grano); t.copyTo(turbulencia);
            mx.copyTo(mapaX); my.copyTo(mapaY);
        }
        if (nuevoTam || p.fGrano != fGrano || p.fTurbulencia != fTurb) {
            fGrano = p.fGrano; fTurb = p.fTurbulencia;
            cv::addWeighted(grano, fGrano, turbulencia, fTurb, 0.0, textura);
        }
    }
};

// -----------------------------------------------------------------------------
// Etapas intermedias (reutilizadas entre frames: evita reservar memoria)
// -----------------------------------------------------------------------------
template <typename M>
struct EtapasAcuarela {
    // máscara
    M grisBA, grisBAEsc, silueta, siluetaSuave, alfa1, alfa3;
    // abstracción
    M pequena, bilA, bilB, medPeq, manchas;
    // paletas
    M lab8, labF, labG, labC, diff, labMezcla, labPost, bgrPost, bgrMed;
    // sangrado y bordes
    M sangrado, grisMancha, respBorde, bordeAbs, bordeEsc, bordeTrunc, borde;
    // lápiz
    M grisCam, grisCamS, grisCamEq, respLapiz, lapizEsc, lapizPos, lapizTrunc, lapiz;
    // densidad de pigmento
    M densidad, densidadTex, densidad3, dm1, C, unoMenosC, t1, t2, salida;
    M final;
};

// -----------------------------------------------------------------------------
// PIPELINE (idéntico para CPU y GPU)
//   camara   : frame BGR de la cámara
//   badApple : frame BGR del video Bad Apple (cualquier tamaño)
// -----------------------------------------------------------------------------
template <typename M>
void aplicarAcuarela(const M& camara, const M& badApple, const RecursosAcuarela<M>& r,
                     const ParametrosAcuarela& p, EtapasAcuarela<M>& e) {
    using namespace cv;
    const Size tam = camara.size();

    // ---- 0. Máscara Bad Apple: 1 = silueta negra, 0 = fondo blanco ----------
    cvtColor(badApple, e.grisBA, COLOR_BGR2GRAY);
    resize(e.grisBA, e.grisBAEsc, tam, 0, 0, INTER_LINEAR);
    threshold(e.grisBAEsc, e.silueta, 127, 255, THRESH_BINARY_INV);
    GaussianBlur(e.silueta, e.siluetaSuave, Size(0, 0), 2.0);     // transición suave
    e.siluetaSuave.convertTo(e.alfa1, CV_32F, 1.0 / 255.0);
    cvtColor(e.alfa1, e.alfa3, COLOR_GRAY2BGR);

    MARCA_ETAPA("mascara");

    // ---- 1. Abstracción: manchas de pigmento (a media resolución) -----------
    resize(camara, e.pequena, Size(), 0.5, 0.5, INTER_AREA);
    e.pequena.copyTo(e.bilA);
    for (int i = 0; i < std::max(1, p.pasadasBilateral); i++) {
        bilateralFilter(e.bilA, e.bilB, std::max(1, p.bilateralD), 60.0, 9.0);
        std::swap(e.bilA, e.bilB);
    }
    medianBlur(e.bilA, e.medPeq, hacerImpar(p.medianaManchas, 3));
    resize(e.medPeq, e.manchas, tam, 0, 0, INTER_LINEAR);

    MARCA_ETAPA("abstraccion");

    // ---- 2. Paletas: transferencia de color de Reinhard en Lab --------------
    //   x' = (x - media_cam) / desv_cam * desv_cuadro + media_cuadro
    //      = x * escala + desplazamiento                     (por canal)
    cvtColor(e.manchas, e.lab8, COLOR_BGR2Lab);
    Scalar mo, so;
    meanStdDev(e.lab8, mo, so);
    Scalar escG, despG, escC, despC;
    for (int c = 0; c < 3; c++) {
        escG[c]  = r.gris.desv[c]  / (so[c] + 1e-6);
        despG[c] = r.gris.media[c]  - mo[c] * escG[c];
        escC[c]  = r.color.desv[c] / (so[c] + 1e-6);
        despC[c] = r.color.media[c] - mo[c] * escC[c];
    }
    e.lab8.convertTo(e.labF, CV_32F);
    multiply(e.labF, escG, e.labG);  add(e.labG, despG, e.labG);
    multiply(e.labF, escC, e.labC);  add(e.labC, despC, e.labC);
    // mezcla = color + (gris - color) * alfa
    subtract(e.labG, e.labC, e.diff);
    multiply(e.diff, e.alfa3, e.diff);
    add(e.labC, e.diff, e.labMezcla);
    e.labMezcla.convertTo(e.lab8, CV_8U);                            // satura a [0,255]

    MARCA_ETAPA("paletas");

    // ---- 3. Lavados planos: posterizar L ------------------------------------
    LUT(e.lab8, r.lutLavados, e.labPost);
    cvtColor(e.labPost, e.bgrPost, COLOR_Lab2BGR);
    medianBlur(e.bgrPost, e.bgrMed, 5);

    MARCA_ETAPA("lavados");

    // ---- 4. Bordes que sangran: desplazamiento con ruido suave --------------
    remap(e.bgrMed, e.sangrado, r.mapaX, r.mapaY, INTER_LINEAR, BORDER_REFLECT);

    MARCA_ETAPA("sangrado");

    // ---- 5. Borde húmedo de cada mancha (KERNEL PROPIO) ---------------------
    cvtColor(e.sangrado, e.grisMancha, COLOR_BGR2GRAY);
    filter2D(e.grisMancha, e.respBorde, CV_32F, r.kernelDoG);
    absdiff(e.respBorde, Scalar::all(0), e.bordeAbs);                // |K * I|
    e.bordeAbs.convertTo(e.bordeEsc, CV_32F, 6.0 / 255.0);
    threshold(e.bordeEsc, e.bordeTrunc, 1.0, 1.0, THRESH_TRUNC);     // min(x, 1)
    GaussianBlur(e.bordeTrunc, e.borde, Size(3, 3), 0);

    MARCA_ETAPA("borde");

    // ---- 5b. Dibujo previo a lápiz (mismo KERNEL PROPIO) --------------------
    cvtColor(camara, e.grisCam, COLOR_BGR2GRAY);
    GaussianBlur(e.grisCam, e.grisCamS, Size(3, 3), 0);
    equalizeHist(e.grisCamS, e.grisCamEq);
    filter2D(e.grisCamEq, e.respLapiz, CV_32F, r.kernelDoG);
    e.respLapiz.convertTo(e.lapizEsc, CV_32F, -1.0 / 40.0);          // lado oscuro > 0
    threshold(e.lapizEsc, e.lapizPos, 0.0, 0.0, THRESH_TOZERO);      // max(x, 0)
    threshold(e.lapizPos, e.lapizTrunc, 1.0, 1.0, THRESH_TRUNC);     // min(x, 1)
    morphologyEx(e.lapizTrunc, e.lapiz, MORPH_OPEN, r.elementoLapiz); // quita puntos sueltos

    MARCA_ETAPA("lapiz");

    // ---- 6. Densidad de pigmento (Bousseau 2006) ----------------------------
    //   d  = 1 + fBorde*borde + fLapiz*lapiz + (fGrano*grano + fTurb*turbulencia)
    //   C  = transparencia + (1 - transparencia) * color/255
    //   C' = C - C(1 - C)(d - 1)
    addWeighted(e.borde, p.fBorde, e.lapiz, p.fLapiz, 1.0, e.densidad);
    add(e.densidad, r.textura, e.densidadTex);
    cvtColor(e.densidadTex, e.densidad3, COLOR_GRAY2BGR);
    subtract(e.densidad3, Scalar::all(1.0), e.dm1);
    e.sangrado.convertTo(e.C, CV_32F, (1.0 - p.transparencia) / 255.0, p.transparencia);
    subtract(Scalar::all(1.0), e.C, e.unoMenosC);
    multiply(e.C, e.unoMenosC, e.t1);
    multiply(e.t1, e.dm1, e.t2);
    subtract(e.C, e.t2, e.salida);
    e.salida.convertTo(e.final, CV_8U, 255.0);
    MARCA_ETAPA("densidad");
}
