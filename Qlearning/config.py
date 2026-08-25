"""Hiperparametros por defecto del entrenamiento y rutas de persistencia."""

ALPHA = 0.1     # tasa de aprendizaje
GAMMA = 0.9     # descuento sobre la recompensa futura estimada
EPSILON = 0.1   # probabilidad de explorar (accion al azar) en vez de explotar

RUTA_TABLA_HARVESTER = 'Qlearning/tabla_q_harvester.json'
RUTA_TABLA_TRACTOR = 'Qlearning/tabla_q_tractor.json'
