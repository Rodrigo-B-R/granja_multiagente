"""Simulacion multiagente de cosecha: harvesters + tractores sobre un campo en grid."""

import heapq

import numpy as np
import agentpy as ap

CAMINO = 0
LISTO = 1
COSECHADO = 2
OBSTACULO = 3

TRANSITABLE = (CAMINO, LISTO, COSECHADO)

NORTE = (-1, 0)
SUR = (1, 0)
ESTE = (0, 1)
OESTE = (0, -1)
DIRECCIONES = (NORTE, SUR, ESTE, OESTE)


def a_estrella(terreno, inicio, meta, bloqueadas=frozenset()):
    """Ruta de menor costo entre dos celdas evitando obstaculos. Devuelve [] si no hay."""
    if inicio == meta:
        return []
    filas, columnas = terreno.shape

    def h(p):
        return abs(p[0] - meta[0]) + abs(p[1] - meta[1])

    abiertos = [(h(inicio), 0, inicio)]
    costo = {inicio: 0}
    padre = {}
    visitados = set()

    while abiertos:
        _, g, actual = heapq.heappop(abiertos)
        if actual == meta:
            ruta = []
            while actual != inicio:
                ruta.append(actual)
                actual = padre[actual]
            ruta.reverse()
            return ruta
        if actual in visitados:
            continue
        visitados.add(actual)

        for df, dc in DIRECCIONES:
            vecino = (actual[0] + df, actual[1] + dc)
            if not (0 <= vecino[0] < filas and 0 <= vecino[1] < columnas):
                continue
            if terreno[vecino] == OBSTACULO:
                continue
            if vecino in bloqueadas and vecino != meta:
                continue
            nuevo_g = g + 1
            if nuevo_g < costo.get(vecino, np.inf):
                costo[vecino] = nuevo_g
                padre[vecino] = actual
                heapq.heappush(abiertos, (nuevo_g + h(vecino), nuevo_g, vecino))

    return []


class Campo(ap.Grid):
    """Grid con un camino de tierra perimetral y central, cultivo y obstaculos."""

    def setup(self, ancho_camino=2, pct_obstaculos=0.02):
        filas, columnas = self.shape
        self.terreno = np.full(self.shape, LISTO, dtype=np.int8)

        w = ancho_camino
        self.terreno[:w, :] = CAMINO
        self.terreno[-w:, :] = CAMINO
        self.terreno[:, :w] = CAMINO
        self.terreno[:, -w:] = CAMINO

        centro_f = filas // 2
        centro_c = columnas // 2
        self.terreno[centro_f - w // 2: centro_f - w // 2 + w, :] = CAMINO
        self.terreno[:, centro_c - w // 2: centro_c - w // 2 + w] = CAMINO

        celdas_cultivo = np.argwhere(self.terreno == LISTO)
        n_obstaculos = int(len(celdas_cultivo) * pct_obstaculos)
        elegidas = self.model.nprandom.choice(
            len(celdas_cultivo), size=n_obstaculos, replace=False)
        for idx in elegidas:
            f, c = celdas_cultivo[idx]
            self.terreno[f, c] = OBSTACULO

        self.total_cultivo = int((self.terreno == LISTO).sum())

    def cosechar(self, pos):
        if self.terreno[pos] == LISTO:
            self.terreno[pos] = COSECHADO
            return True
        return False

    def celdas_listas(self):
        return [tuple(p) for p in np.argwhere(self.terreno == LISTO)]


class Maquina(ap.Agent):
    """Base comun: se mueve por una ruta A*, gira, se detiene y gasta gasolina."""

    velocidad = 1
    capacidad = 0
    gasolina_max = 300.0
    consumo = 1.0

    def setup(self):
        self.ruta = []
        self.gasolina = self.gasolina_max
        self.direccion = ESTE
        self.detenido = False
        self.carga = 0
        self.distancia = 0
        self.giros = 0

    @property
    def ubicacion(self):
        return self.model.campo.positions[self]

    def definirRuta(self, meta, evitar_agentes=True):
        bloqueadas = self.model.celdas_ocupadas(excepto=self) if evitar_agentes else frozenset()
        self.ruta = a_estrella(self.model.campo.terreno, self.ubicacion, meta, bloqueadas)
        return bool(self.ruta)

    def girar(self, nueva_direccion):
        if nueva_direccion != self.direccion:
            self.giros += 1
            self.direccion = nueva_direccion

    def detenerse(self):
        self.detenido = True
        self.ruta = []

    def reanudar(self):
        self.detenido = False

    def moverse(self):
        """Avanza hasta `velocidad` celdas por paso. Devuelve las celdas recorridas."""
        recorridas = []
        if self.detenido or self.gasolina <= 0:
            return recorridas

        for _ in range(self.velocidad):
            if not self.ruta:
                break
            siguiente = self.ruta[0]
            if siguiente in self.model.celdas_ocupadas(excepto=self):
                self.ruta = []  # sensor de proximidad: replanear el proximo paso
                break
            actual = self.ubicacion
            self.girar((siguiente[0] - actual[0], siguiente[1] - actual[1]))
            self.model.campo.move_to(self, siguiente)
            self.ruta.pop(0)
            self.gasolina -= self.consumo
            self.distancia += 1
            recorridas.append(siguiente)
            if self.gasolina <= 0:
                break

        return recorridas


class Harvester(Maquina):
    """Cosechadora: recorre el cultivo, y al llenarse llama a un tractor."""

    def setup(self):
        super().setup()
        self.tamano = self.p.tamano_harvester
        self.velocidad = self.p.velocidad_harvester
        self.capacidad = self.p.capacidad_harvester
        self.gasolina_max = self.p.gasolina_harvester
        self.gasolina = self.gasolina_max
        self.consumo = self.p.consumo_harvester
        self.estado = 'operando'
        self.objetivo = None
        self.tractor_asignado = None

    def buscar_objetivo(self):
        listas = [c for c in self.model.campo.celdas_listas()
                  if c not in self.model.reservadas]
        if not listas:
            self.objetivo = None
            return False
        aqui = self.ubicacion
        listas.sort(key=lambda c: abs(c[0] - aqui[0]) + abs(c[1] - aqui[1]))
        for candidata in listas[:20]:
            if self.definirRuta(candidata):
                self.model.reservadas.discard(self.objetivo)
                self.objetivo = candidata
                self.model.reservadas.add(candidata)
                return True
        self.objetivo = None
        return False

    def llamarTractor(self):
        tractor = self.model.solicitar_tractor(self)
        self.tractor_asignado = tractor
        return tractor

    def vertirDeposito(self, tractor):
        transferido = min(self.p.tasa_vertido, self.carga,
                          tractor.capacidad - tractor.carga)
        self.carga -= transferido
        tractor.carga += transferido
        self.model.transferido += transferido
        return transferido

    def pararVertimiento(self):
        self.estado = 'operando'
        self.tractor_asignado = None
        self.reanudar()

    def step(self):
        if self.gasolina <= 0:
            self.estado = 'sin_gasolina'
            return

        if self.estado == 'esperando_tractor':
            return  # inmovil hasta que llegue el tractor

        if self.estado == 'vertiendo':
            tractor = self.tractor_asignado
            if tractor is None or tractor.carga >= tractor.capacidad:
                self.pararVertimiento()
            else:
                self.vertirDeposito(tractor)
                if self.carga == 0:
                    tractor.liberar()
                    self.pararVertimiento()
            return

        if self.carga >= self.capacidad:
            self.detenerse()
            self.estado = 'esperando_tractor'
            self.llamarTractor()
            return

        if not self.ruta:
            self.buscar_objetivo()

        for celda in self.moverse():
            if self.model.campo.cosechar(celda):
                self.carga += 1
                self.model.cosechado += 1
                self.model.reservadas.discard(celda)
                if self.carga >= self.capacidad:
                    break


class Tractor(Maquina):
    """Contenedor movil: acude al harvester que lo llama y lleva el grano al silo."""

    def setup(self):
        super().setup()
        self.velocidad = self.p.velocidad_tractor
        self.capacidad = self.p.capacidad_tractor
        self.gasolina_max = self.p.gasolina_tractor
        self.gasolina = self.gasolina_max
        self.consumo = self.p.consumo_tractor
        self.estado = 'libre'
        self.cliente = None

    def asignar(self, harvester):
        self.cliente = harvester
        self.estado = 'en_camino'

    def liberar(self):
        self.cliente = None
        self.estado = 'libre'
        self.ruta = []
        self.reanudar()

    def adyacente_a(self, otro):
        a, b = self.ubicacion, otro.ubicacion
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1

    def step(self):
        if self.gasolina <= 0:
            self.estado = 'sin_gasolina'
            return

        if self.carga >= self.capacidad and self.estado != 'descargando':
            self.cliente = None
            self.estado = 'al_silo'

        if self.estado == 'al_silo':
            if self.ubicacion == self.model.silo:
                self.model.entregado += self.carga
                self.carga = 0
                self.estado = 'libre'
            else:
                if not self.ruta:
                    self.definirRuta(self.model.silo)
                self.moverse()
            return

        if self.estado == 'en_camino':
            harvester = self.cliente
            if harvester is None:
                self.estado = 'libre'
                return
            if self.adyacente_a(harvester):
                self.detenerse()
                self.estado = 'descargando'
                harvester.estado = 'vertiendo'
                harvester.tractor_asignado = self
            else:
                self.reanudar()
                if not self.ruta:
                    self.definirRuta(harvester.ubicacion)
                self.moverse()
            return

        if self.estado == 'descargando':
            if self.cliente is None or self.cliente.estado != 'vertiendo':
                self.reanudar()
                self.liberar()
            return

        if self.estado == 'libre' and self.carga > 0 and self.ubicacion != self.model.silo:
            self.estado = 'al_silo'


class GranjaModel(ap.Model):

    def setup(self):
        self.nprandom = np.random.default_rng(self.p.get('seed', 1))
        self.campo = Campo(self, self.p.shape,
                           ancho_camino=self.p.ancho_camino,
                           pct_obstaculos=self.p.pct_obstaculos)

        w = self.p.ancho_camino
        self.silo = (w // 2, w // 2)
        self.reservadas = set()
        self.cosechado = 0
        self.transferido = 0
        self.entregado = 0

        caminos = [tuple(p) for p in np.argwhere(self.campo.terreno == CAMINO)]

        self.harvesters = ap.AgentList(self, self.p.n_harvesters, Harvester)
        self.tractores = ap.AgentList(self, self.p.n_tractores, Tractor)

        idx = self.nprandom.choice(len(caminos),
                                   size=len(self.harvesters) + len(self.tractores),
                                   replace=False)
        arranques = [caminos[i] for i in idx]
        self.campo.add_agents(self.harvesters, arranques[:len(self.harvesters)])
        self.campo.add_agents(self.tractores, arranques[len(self.harvesters):])

    def celdas_ocupadas(self, excepto=None):
        return {pos for agente, pos in self.campo.positions.items() if agente is not excepto}

    def solicitar_tractor(self, harvester):
        """Asignacion por cercania: el tractor libre mas proximo atiende la llamada."""
        libres = [t for t in self.tractores
                  if t.estado == 'libre' and t.carga < t.capacidad and t.gasolina > 0]
        if not libres:
            return None
        aqui = harvester.ubicacion
        tractor = min(libres, key=lambda t: abs(t.ubicacion[0] - aqui[0])
                      + abs(t.ubicacion[1] - aqui[1]))
        tractor.asignar(harvester)
        return tractor

    def step(self):
        self.harvesters.step()
        self.tractores.step()
        if self.cosechado >= self.campo.total_cultivo:
            self.stop()

    def update(self):
        self.record('cosechado_pct', 100 * self.cosechado / self.campo.total_cultivo)
        self.record('grano_entregado', self.entregado)
        self.record('gasolina_total', float(sum(self.harvesters.gasolina)
                                            + sum(self.tractores.gasolina)))
        self.record('distancia_total', int(sum(self.harvesters.distancia)
                                           + sum(self.tractores.distancia)))

    def end(self):
        self.report('pasos', self.t)
        self.report('cosechado_pct', 100 * self.cosechado / self.campo.total_cultivo)
        self.report('grano_entregado', self.entregado)
        self.report('distancia_total', int(sum(self.harvesters.distancia)
                                           + sum(self.tractores.distancia)))
        self.report('combustible_usado',
                    float(sum(h.gasolina_max - h.gasolina for h in self.harvesters)
                          + sum(t.gasolina_max - t.gasolina for t in self.tractores)))
        self.report('giros_totales', int(sum(self.harvesters.giros)
                                         + sum(self.tractores.giros)))


PARAMETROS = {
    'shape': (40, 40),
    'ancho_camino': 2,
    'pct_obstaculos': 0.02,
    'n_harvesters': 3,
    'n_tractores': 2,
    'tamano_harvester': 2,
    'velocidad_harvester': 1,
    'capacidad_harvester': 25,
    'gasolina_harvester': 1200,
    'consumo_harvester': 1.0,
    'velocidad_tractor': 2,
    'capacidad_tractor': 80,
    'gasolina_tractor': 1500,
    'consumo_tractor': 0.7,
    'tasa_vertido': 8,
    'seed': 1,
    'steps': 1200,
}
