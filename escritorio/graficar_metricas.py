#!/usr/bin/env python3
# =============================================================================
#  graficar_metricas.py - Gráficas para el informe a partir de metricas.csv
#
#  Uso:   python3 graficar_metricas.py [metricas.csv] [carpeta_salida]
#
#  Genera (PNG, 150 dpi):
#   1_tiempo_por_frame.png   ms por frame CPU vs GPU a lo largo de la ejecución
#   2_resumen_cpu_gpu.png    ms promedio, FPS y RAM: CPU vs GPU (modos 0 y 1)
#   3_escalado_bilateral.png ms vs diámetro del bilateral (saturación)
#   4_escalado_kernel_dog.png ms vs tamaño del kernel DoG (saturación)
#  y un resumen numérico en consola (para copiar al informe).
#
#  Columnas de metricas.csv (las escribe vision.bin):
#   tiempo_s, modo (0 CPU, 1 GPU, 2 AMBOS), bilateral_d, pasadas, kernel_dog,
#   ms_cpu, ms_gpu (-1 = no medido), fps, ram_mb, cpu_pct, gpu_util_pct,
#   gpu_mem_proceso_mb
# =============================================================================
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paleta validada (azul = CPU, naranja = GPU) y tinta de texto neutra
COLOR = {"CPU": "#2a78d6", "GPU": "#eb6834"}
TINTA, TINTA_SEC, REJILLA, FONDO = "#1f1f1e", "#5f5e58", "#e4e3de", "#fcfcfb"
DESCARTAR_S = 2.0   # segundos iniciales descartados (compilación de kernels OpenCL)

plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA_SEC, "text.color": TINTA,
    "xtick.color": TINTA_SEC, "ytick.color": TINTA_SEC, "axes.grid": True,
    "grid.color": REJILLA, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "legend.frameon": False,
})


def leer(ruta):
    filas = []
    with open(ruta, newline="") as f:
        for r in csv.DictReader(f):
            try:
                filas.append({k: float(v) for k, v in r.items()})
            except (TypeError, ValueError):
                pass  # fila incompleta (p. ej. última línea cortada)
    return [r for r in filas if r["tiempo_s"] >= DESCARTAR_S]


def media(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def guardar(fig, carpeta, nombre):
    ruta = os.path.join(carpeta, nombre)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print("  ->", ruta)


def tiempo_por_frame(filas, carpeta):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for etiqueta, col in (("CPU", "ms_cpu"), ("GPU", "ms_gpu")):
        pts = [(r["tiempo_s"], r[col]) for r in filas if r[col] >= 0]
        if pts:
            t, ms = zip(*pts)
            ax.plot(t, ms, ".", ms=3, color=COLOR[etiqueta], label=f"{etiqueta}")
    ax.set_title("Tiempo de procesamiento por frame")
    ax.set_xlabel("Tiempo de ejecución (s)")
    ax.set_ylabel("ms por frame")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", markerscale=4)
    guardar(fig, carpeta, "1_tiempo_por_frame.png")


def resumen(filas, carpeta):
    """Compara modo 0 (solo CPU) con modo 1 (solo GPU): ms, FPS y RAM.
    El modo AMBOS no se usa para FPS/RAM porque ejecuta los dos pipelines."""
    cpu = [r for r in filas if r["modo"] == 0]
    gpu = [r for r in filas if r["modo"] == 1]
    # ms: también se aprovechan las mediciones del modo AMBOS
    ms_cpu = media([r["ms_cpu"] for r in filas if r["ms_cpu"] >= 0])
    ms_gpu = media([r["ms_gpu"] for r in filas if r["ms_gpu"] >= 0])
    metricas = [
        ("Tiempo por frame (ms)", ms_cpu, ms_gpu, "{:.1f}"),
        ("FPS de la aplicación", media([r["fps"] for r in cpu]), media([r["fps"] for r in gpu]), "{:.1f}"),
        ("RAM del proceso (MB)", media([r["ram_mb"] for r in cpu]), media([r["ram_mb"] for r in gpu]), "{:.0f}"),
    ]
    fig, ejes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, (titulo, vc, vg, fmt) in zip(ejes, metricas):
        etiquetas, valores = ["CPU\n(cv::Mat)", "GPU\n(cv::UMat)"], [vc, vg]
        barras = ax.bar(etiquetas, valores, width=0.55,
                        color=[COLOR["CPU"], COLOR["GPU"]], edgecolor=FONDO, linewidth=2)
        for b, v in zip(barras, valores):
            if v == v:  # no es NaN
                ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                            ha="center", va="bottom", xytext=(0, 3), textcoords="offset points",
                            color=TINTA, fontsize=11)
        ax.set_title(titulo, fontsize=12)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max([v for v in valores if v == v] + [1]) * 1.18)
    fig.suptitle("CPU vs GPU (OpenCL)", fontweight="bold")
    guardar(fig, carpeta, "2_resumen_cpu_gpu.png")
    return ms_cpu, ms_gpu, metricas, len(cpu), len(gpu)


def escalado(filas, carpeta, parametro, titulo, eje_x, nombre):
    """ms promedio por valor del parámetro, dejando FIJOS los demás parámetros
    (en su valor más frecuente) para no mezclar efectos."""
    otros = [c for c in ("bilateral_d", "pasadas", "kernel_dog") if c != parametro]
    for c in otros:
        valores_c = [r[c] for r in filas]
        fijo = max(set(valores_c), key=valores_c.count)
        filas = [r for r in filas if r[c] == fijo]
    valores = sorted({r[parametro] for r in filas})
    if len(valores) < 2:
        print(f"  (se omite {nombre}: {parametro} no varió durante la ejecución)")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for etiqueta, col in (("CPU", "ms_cpu"), ("GPU", "ms_gpu")):
        xs, ys = [], []
        for v in valores:
            m = media([r[col] for r in filas if r[parametro] == v and r[col] >= 0])
            if m == m:
                xs.append(v); ys.append(m)
        if xs:
            ax.plot(xs, ys, "-o", lw=2, ms=6, color=COLOR[etiqueta], label=etiqueta,
                    markeredgecolor=FONDO, markeredgewidth=1.5)
            ax.annotate(etiqueta, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=TINTA, fontsize=10)
    ax.set_title(titulo)
    ax.set_xlabel(eje_x)
    ax.set_ylabel("ms por frame")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    guardar(fig, carpeta, nombre)


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else "metricas.csv"
    carpeta = sys.argv[2] if len(sys.argv) > 2 else "graficas"
    os.makedirs(carpeta, exist_ok=True)
    filas = leer(ruta)
    if not filas:
        sys.exit(f"Sin datos útiles en {ruta}")
    print(f"{len(filas)} frames leídos de {ruta} (descartados los primeros {DESCARTAR_S:.0f} s)")

    tiempo_por_frame(filas, carpeta)
    ms_cpu, ms_gpu, metricas, n_cpu, n_gpu = resumen(filas, carpeta)
    escalado(filas, carpeta, "bilateral_d", "Escalado con el filtro bilateral",
             "Diámetro de la ventana del bilateral (d)", "3_escalado_bilateral.png")
    escalado(filas, carpeta, "kernel_dog", "Escalado con el kernel propio DoG",
             "Tamaño del kernel DoG (n × n)", "4_escalado_kernel_dog.png")

    print("\nResumen para el informe:")
    for titulo, vc, vg, fmt in metricas:
        print(f"  {titulo:24s} CPU {fmt.format(vc):>8s}   GPU {fmt.format(vg):>8s}")
    if ms_gpu == ms_gpu and ms_gpu > 0:
        print(f"  Aceleración GPU/CPU      x{ms_cpu / ms_gpu:.1f}")
    if n_cpu == 0 or n_gpu == 0:
        print("  Nota: para comparar FPS y RAM hay que ejecutar un rato en modo 0 (CPU) y otro en modo 1 (GPU).")


if __name__ == "__main__":
    main()
