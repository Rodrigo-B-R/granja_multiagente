"""Estado, acciones y tabla Q inicial para la decision del Harvester.

Decision que modela: en cada paso en que el harvester esta operando (no
lleno, no vertiendo, no ya recargando/esperando tractor, no descompuesto),
elegir entre seguir cosechando o desviarse a recargar gasolina. En
`granja.py` esa decision hoy es un umbral fijo (`Maquina.necesitaGasolina`,
fraccion constante `p.umbral_gasolina`); esta tabla es el reemplazo
aprendido de ese mismo punto de decision -- no toca el resto de la maquina
de estados (llenado, vertido, averias siguen siendo logica determinista).
"""

from .q_table import tabla_vacia
from .utils import (
    CORTES_CARGA,
    CORTES_DISTANCIA,
    CORTES_GASOLINA,
    N_CARGA,
    N_DISTANCIA,
    N_GASOLINA,
    discretizar,
    distancia_manhattan,
)

SEGUIR_COSECHANDO = 0
IR_A_RECARGAR = 1
ACCIONES = (SEGUIR_COSECHANDO, IR_A_RECARGAR)

# cortes de cultivo restante en la zona propia, como fraccion del area de la zona
CORTES_CULTIVO_RESTANTE = (0.15, 0.60)
N_CULTIVO_RESTANTE = len(CORTES_CULTIVO_RESTANTE) + 1  # 3: poco, medio, mucho

N_TRACTOR_DISPONIBLE = 2  # booleano: hay o no un tractor libre/escoltando con gasolina

# orden de las variables de estado en la tupla que arma discretizar_estado()
FORMA_ESTADOS = (N_GASOLINA, N_CARGA, N_DISTANCIA, N_CULTIVO_RESTANTE, N_TRACTOR_DISPONIBLE)


def _cultivo_restante_zona_frac(harvester):
    """Fraccion de celdas LISTO que quedan dentro de la zona actual del
    harvester, sobre el area total de esa zona (0.0 si no tiene zona)."""
    if harvester.zona is None:
        return 0.0
    r0, r1, c0, c1 = harvester.zona
    area = (r1 - r0) * (c1 - c0)
    if area == 0:
        return 0.0
    # import diferido: granja.py importa este modulo a nivel de archivo, asi
    # que importar LISTO arriba del todo crearia un ciclo (granja -> Qlearning
    # -> granja) que rompe al cargar granja.py. Para cuando esta funcion se
    # llama de verdad (en medio de una corrida) granja.py ya termino de cargar.
    from granja import LISTO
    terreno = harvester.model.campo.terreno[r0:r1, c0:c1]
    listas = int((terreno == LISTO).sum())
    return listas / area


def _hay_tractor_disponible(model):
    return any(t.estado in ('libre', 'escoltando') and not t.necesitaGasolina()
               for t in model.tractores)


def discretizar_estado(harvester):
    """Estado discreto del harvester: (gasolina, carga, distancia_a_base,
    cultivo_restante_zona, tractor_disponible)."""
    model = harvester.model
    frac_gasolina = harvester.gasolina / harvester.gasolina_max
    frac_carga = harvester.carga / harvester.capacidad if harvester.capacidad else 0.0
    dist_base = distancia_manhattan(harvester.ubicacion, model.base)
    frac_cultivo = _cultivo_restante_zona_frac(harvester)

    return (
        discretizar(frac_gasolina, CORTES_GASOLINA),
        discretizar(frac_carga, CORTES_CARGA),
        discretizar(dist_base, CORTES_DISTANCIA),
        discretizar(frac_cultivo, CORTES_CULTIVO_RESTANTE),
        int(_hay_tractor_disponible(model)),
    )


def crear_tabla_inicial():
    return tabla_vacia(FORMA_ESTADOS, len(ACCIONES))
