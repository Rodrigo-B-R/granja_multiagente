"""Utilidades compartidas para discretizar variables continuas en bins."""


def discretizar(valor, cortes):
    """Indice del bin al que pertenece `valor` segun `cortes` (umbrales
    crecientes). Con cortes=(a, b, c) hay 4 bins: valor < a -> 0,
    a <= valor < b -> 1, b <= valor < c -> 2, valor >= c -> 3."""
    for i, corte in enumerate(cortes):
        if valor < corte:
            return i
    return len(cortes)


def distancia_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Cortes compartidos entre Harvester y Tractor: ambos razonan sobre la misma
# nocion de "fraccion de tanque/capacidad" y "celdas de distancia", asi que
# conviene una sola definicion en vez de que cada modulo invente sus propios
# umbrales y terminen divergiendo sin querer.

# fraccion del tanque: <10% critico, <30% bajo, <60% medio, resto alto
CORTES_GASOLINA = (0.10, 0.30, 0.60)
N_GASOLINA = len(CORTES_GASOLINA) + 1  # 4: critico, bajo, medio, alto

# fraccion de la capacidad de carga: <10% vacio, <50% bajo, <90% medio, resto lleno
CORTES_CARGA = (0.10, 0.50, 0.90)
N_CARGA = len(CORTES_CARGA) + 1  # 4: vacio, bajo, medio, lleno

# distancia en celdas: <10 cerca, <25 media, resto lejos
CORTES_DISTANCIA = (10, 25)
N_DISTANCIA = len(CORTES_DISTANCIA) + 1  # 3: cerca, media, lejos
