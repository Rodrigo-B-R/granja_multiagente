"""Tablas Q genericas: diccionario {estado: [valor_por_accion, ...]}.

Un estado es una tupla de enteros (los indices discretos que arman
`harvester_rl.discretizar_estado` / `tractor_rl.discretizar_estado`). Se usa
un diccionario en vez de un array numpy denso porque el espacio de estados
esta definido por combinaciones de bins categoricos con distinto significado
por posicion, no por indices continuos: el diccionario deja construirlo,
recorrerlo y guardarlo sin atarse a la forma exacta de esas combinaciones.
"""

import itertools
import json


def tabla_vacia(forma_estados, n_acciones, valor_inicial=0.0):
    """Tabla Q con una entrada por cada combinacion posible de estado.

    `forma_estados` es una tupla con la cantidad de bins de cada variable de
    estado (p.ej. (4, 4, 3, 3, 2)). Cada estado arranca con `valor_inicial`
    para las `n_acciones` acciones disponibles -- 0.0 por defecto (sin
    conocimiento previo; la exploracion durante el entrenamiento es la que
    va a diferenciar los valores).
    """
    tabla = {}
    for estado in itertools.product(*(range(n) for n in forma_estados)):
        tabla[estado] = [valor_inicial] * n_acciones
    return tabla


def mejor_accion(tabla, estado, acciones):
    """Accion de mayor valor Q para `estado` (desempate por orden de `acciones`)."""
    valores = tabla[estado]
    mejor_idx = max(range(len(acciones)), key=lambda i: valores[i])
    return acciones[mejor_idx]


def guardar(tabla, ruta):
    """Serializa la tabla a JSON (las tuplas de estado se guardan como
    "i|j|k" porque JSON no admite tuplas como clave)."""
    serializable = {'|'.join(map(str, estado)): valores for estado, valores in tabla.items()}
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def cargar(ruta):
    with open(ruta, encoding='utf-8') as f:
        serializable = json.load(f)
    return {tuple(int(x) for x in clave.split('|')): valores
            for clave, valores in serializable.items()}
