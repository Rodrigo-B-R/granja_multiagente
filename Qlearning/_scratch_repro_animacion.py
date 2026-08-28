import os

import agentpy as ap
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import imageio_ffmpeg

from granja import GranjaModel, PARAMETROS

mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

PARAMETROS_ENTRENAMIENTO = dict(
    PARAMETROS,
    shape=(24, 24),
    steps=5000,
    n_tractores=2,
    capacidad_tractor=40,
    gasolina_tractor=300,
    gasolina_harvester=400,
    usar_qlearning_tractor=True,
    usar_qlearning_harvester=True,
)

SEED_ANIMACION = 999999
STEPS_ANIMACION = 700
PARAMETROS_ANIMACION = dict(PARAMETROS_ENTRENAMIENTO, steps=STEPS_ANIMACION, qlearning_epsilon=0.0)

COLOR_TERRENO = mcolors.ListedColormap(
    ['#c2a878', '#7cb342', '#c9b458', '#4a4a4a'])

COLOR_HARVESTER = {
    'operando': '#1565c0',
    'esperando_tractor': '#ef6c00',
    'vertiendo': '#6a1b9a',
    'recargando': '#212121',
    'sin_gasolina': '#b71c1c',
    'descompuesto': '#4e342e',
}

COLOR_TRACTOR = {
    'libre': '#9e9e9e',
    'escoltando': '#fdd835',
    'en_camino': '#ef6c00',
    'descargando': '#6a1b9a',
    'al_silo': '#1565c0',
    'recargando': '#212121',
    'sin_gasolina': '#b71c1c',
}


def dibujar_campo(model, ax):
    ax.clear()
    ax.imshow(model.campo.terreno, cmap=COLOR_TERRENO, vmin=0, vmax=3, origin='upper')

    silo = model.silo
    base = model.base
    ax.scatter([silo[1]], [silo[0]], marker='s', c='black', s=140, zorder=5)
    ax.scatter([base[1]], [base[0]], marker='s', c='#ef6c00', s=140, zorder=5)

    if len(model.harvesters):
        hy = [h.ubicacion[0] for h in model.harvesters]
        hx = [h.ubicacion[1] for h in model.harvesters]
        colores = [COLOR_HARVESTER.get(h.estado, 'gray') for h in model.harvesters]
        ax.scatter(hx, hy, marker='^', c=colores, s=160,
                   edgecolors='white', linewidths=1, zorder=6)

    if len(model.tractores):
        ty = [t.ubicacion[0] for t in model.tractores]
        tx = [t.ubicacion[1] for t in model.tractores]
        colores_t = [COLOR_TRACTOR.get(t.estado, 'red') for t in model.tractores]
        ax.scatter(tx, ty, marker='o', c=colores_t, s=120,
                   edgecolors='white', linewidths=1, zorder=6)

    pct = 100 * model.cosechado / model.campo.total_cultivo
    ax.set_title(f"t={model.t}   cosechado={pct:.1f}%   grano entregado={model.entregado}")
    ax.set_xticks([])
    ax.set_yticks([])


def dibujar_gasolina(model, ax):
    ax.clear()
    harvesters = list(model.harvesters)
    tractores = list(model.tractores)
    agentes = harvesters + tractores
    etiquetas = [f'H{i}' for i in range(len(harvesters))] + [f'T{i}' for i in range(len(tractores))]
    niveles = [100 * a.gasolina / a.gasolina_max for a in agentes]
    colores = [COLOR_HARVESTER['operando']] * len(harvesters) + [COLOR_TRACTOR['al_silo']] * len(tractores)

    y_pos = range(len(agentes))
    ax.barh(y_pos, niveles, color=colores, edgecolor='white', height=0.7)
    for y, nivel in zip(y_pos, niveles):
        ax.text(min(nivel + 3, 90), y, f'{nivel:.0f}%', va='center', fontsize=8)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(etiquetas)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel('gasolina %')
    ax.set_title('Nivel de tanque')


def dibujar_todo(model, axs):
    ax_campo, ax_gas = axs
    dibujar_campo(model, ax_campo)
    dibujar_gasolina(model, ax_gas)


fig, axs = plt.subplots(1, 2, figsize=(10, 7), gridspec_kw={'width_ratios': [3, 1]})
m = GranjaModel(dict(PARAMETROS_ANIMACION, seed=SEED_ANIMACION))
animacion = ap.animate(m, fig, axs, dibujar_todo)
plt.close(fig)

os.makedirs('GIFS', exist_ok=True)
ruta = 'GIFS/repro_mas_gasolina.mp4'
animacion.save(ruta, writer='ffmpeg', fps=15)
print('guardado:', ruta)
print('pasos:', m.t, 'cosechado_pct:', 100 * m.cosechado / m.campo.total_cultivo)
for h in m.harvesters:
    print('harvester', h.estado, 'gasolina', h.gasolina, 'carga', h.carga)
for t in m.tractores:
    print('tractor', t.estado, 'gasolina', t.gasolina)
