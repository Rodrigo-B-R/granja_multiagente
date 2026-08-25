"""Politica de Q-learning tabular: eleccion epsilon-greedy y actualizacion
por la regla de Bellman.

Una instancia de `PoliticaQ` envuelve una tabla Q (dict estado -> lista de
valores por accion) y la lista de acciones posibles. `GranjaModel` crea una
sola instancia por tipo de agente (todos los harvesters de la corrida
aprenden sobre la misma tabla, todos los tractores sobre la otra) para
acumular experiencia mas rapido que si cada agente individual aprendiera
por separado con su propia tabla.
"""

import numpy as np


class PoliticaQ:

    def __init__(self, tabla, acciones, alpha, gamma, epsilon, rng=None):
        self.tabla = tabla
        self.acciones = tuple(acciones)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = rng if rng is not None else np.random.default_rng()

    def elegir_accion(self, estado):
        if self.rng.random() < self.epsilon:
            return self.acciones[self.rng.integers(len(self.acciones))]
        valores = self.tabla[estado]
        mejor_idx = max(range(len(self.acciones)), key=lambda i: valores[i])
        return self.acciones[mejor_idx]

    def actualizar(self, estado, accion, recompensa, estado_siguiente):
        """Regla de Bellman de un paso. `estado_siguiente=None` marca una
        transicion terminal (el agente no vuelve a decidir nunca mas: quedo
        sin gasolina lejos de la base o se rompio) y no hace bootstrap sobre
        un futuro que no va a existir."""
        idx_accion = self.acciones.index(accion)
        valor_actual = self.tabla[estado][idx_accion]
        futuro = 0.0 if estado_siguiente is None else max(self.tabla[estado_siguiente])
        objetivo = recompensa + self.gamma * futuro
        self.tabla[estado][idx_accion] = valor_actual + self.alpha * (objetivo - valor_actual)
