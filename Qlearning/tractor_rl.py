"""Estado, acciones y tabla Q inicial para la decision del Tractor.

Decision que modela: en cada paso en que el tractor no esta comprometido con
un cliente ni cargando/descargando (`libre` o `escoltando`), elegir entre
seguir esperando cerca de su harvester asignado, ir a recargar gasolina
preventivamente, o atender la llamada de un harvester que necesita
descarga. En `granja.py` la asignacion hoy la decide el modelo por cercania
pura (`GranjaModel.solicitar_tractor`); aca cada tractor le asigna un valor Q
a "atender" en su propio estado, y el despachador puede compararlos entre
los tractores candidatos en vez de usar solo distancia.
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

ESPERAR = 0
IR_A_RECARGAR = 1
ATENDER_LLAMADA = 2
ACCIONES = (ESPERAR, IR_A_RECARGAR, ATENDER_LLAMADA)

# distancia a la llamada: bin 0 reservado para "nadie esta llamando"; el
# resto usa los mismos cortes de distancia que el resto del estado
N_DISTANCIA_LLAMADA = N_DISTANCIA + 1  # 4: sin_llamada, cerca, media, lejos

# orden de las variables de estado en la tupla que arma discretizar_estado()
FORMA_ESTADOS = (N_GASOLINA, N_CARGA, N_DISTANCIA, N_DISTANCIA_LLAMADA)


def _harvester_llamando_mas_cercano(tractor):
    candidatos = [h for h in tractor.model.harvesters
                  if h.estado == 'esperando_tractor' and h.tractor_asignado is None]
    if not candidatos:
        return None
    aqui = tractor.ubicacion
    return min(candidatos, key=lambda h: distancia_manhattan(h.ubicacion, aqui))


def discretizar_estado(tractor):
    """Estado discreto del tractor: (gasolina, carga, distancia_a_base,
    distancia_a_la_llamada_mas_cercana)."""
    model = tractor.model
    frac_gasolina = tractor.gasolina / tractor.gasolina_max
    frac_carga = tractor.carga / tractor.capacidad if tractor.capacidad else 0.0
    dist_base = distancia_manhattan(tractor.ubicacion, model.base)

    llamando = _harvester_llamando_mas_cercano(tractor)
    if llamando is None:
        idx_llamada = 0
    else:
        dist_llamada = distancia_manhattan(tractor.ubicacion, llamando.ubicacion)
        idx_llamada = 1 + discretizar(dist_llamada, CORTES_DISTANCIA)

    return (
        discretizar(frac_gasolina, CORTES_GASOLINA),
        discretizar(frac_carga, CORTES_CARGA),
        discretizar(dist_base, CORTES_DISTANCIA),
        idx_llamada,
    )


def crear_tabla_inicial():
    return tabla_vacia(FORMA_ESTADOS, len(ACCIONES))
