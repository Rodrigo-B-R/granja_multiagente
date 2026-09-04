"""Simulacion multiagente de cosecha: harvesters + tractores sobre un campo en grid."""

import heapq
import os

import numpy as np
import agentpy as ap

from Qlearning import config as rl_config
from Qlearning import harvester_rl, recompensas as rl_recompensas, tractor_rl
from Qlearning import q_table as rl_q_table
from Qlearning.agente_q import PoliticaQ

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


def a_estrella(terreno, inicio, meta, bloqueadas=frozenset(), transitables=TRANSITABLE, costos=None):
    """Ruta de menor costo entre dos celdas evitando obstaculos. Devuelve [] si no hay.

    `costos` permite penalizar (sin prohibir) pisar cierto tipo de terreno,
    mapeando valor-de-terreno -> costo por celda (default 1 para todos los
    transitables). Con una penalizacion, A* solo elige esa celda cuando de
    verdad conviene (rodear por camino le costaria mas que la penalizacion).
    """
    if inicio == meta:
        return []
    filas, columnas = terreno.shape
    costos = costos or {}

    def costo_celda(pos):
        return costos.get(int(terreno[pos]), 1)

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
            if terreno[vecino] not in transitables:
                continue
            if vecino in bloqueadas and vecino != meta:
                continue
            nuevo_g = g + costo_celda(vecino)
            if nuevo_g < costo.get(vecino, np.inf):
                costo[vecino] = nuevo_g
                padre[vecino] = actual
                heapq.heappush(abiertos, (nuevo_g + h(vecino), nuevo_g, vecino))

    return []


def particionar_rectangulos(filas, columnas, n):
    """Divide el campo en `n` bloques rectangulares lo mas compactos posible,
    biseccionando recursivamente el bloque de mayor area por su lado mas
    largo (particion tipo guillotina). Da areas mas grandes y contiguas que
    repartir en simples franjas de columnas."""
    rects = [(0, filas, 0, columnas)]
    while len(rects) < n:
        rects.sort(key=lambda r: (r[1] - r[0]) * (r[3] - r[2]), reverse=True)
        r0, r1, c0, c1 = rects.pop(0)
        alto, ancho = r1 - r0, c1 - c0
        if alto >= ancho and alto > 1:
            corte = r0 + alto // 2
            rects.append((r0, corte, c0, c1))
            rects.append((corte, r1, c0, c1))
        elif ancho > 1:
            corte = c0 + ancho // 2
            rects.append((r0, r1, c0, corte))
            rects.append((r0, r1, corte, c1))
        else:
            rects.append((r0, r1, c0, c1))
            break
    return rects


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
        self.celdas_camino = [tuple(p) for p in np.argwhere(self.terreno == CAMINO)]

    def cosechar(self, pos):
        if self.terreno[pos] == LISTO:
            self.terreno[pos] = COSECHADO
            return True
        return False

    def celdas_listas(self):
        return [tuple(p) for p in np.argwhere(self.terreno == LISTO)]

    def camino_cercano(self, pos):
        """Celda de camino mas cercana (distancia Manhattan) a `pos`."""
        return min(self.celdas_camino,
                   key=lambda c: abs(c[0] - pos[0]) + abs(c[1] - pos[1]))

    def celda_adyacente_transitable(self, pos):
        """Celda transitable vecina a `pos`, para pararse "al lado" de algo
        (silo, base) en vez de encima. Prefiere camino sobre cultivo."""
        filas, columnas = self.shape
        vecinas = [(pos[0] + df, pos[1] + dc) for df, dc in DIRECCIONES
                   if 0 <= pos[0] + df < filas and 0 <= pos[1] + dc < columnas]
        transitables = [v for v in vecinas if self.terreno[v] in TRANSITABLE]
        if not transitables:
            return pos  # rodeado de obstaculos: no deberia pasar con el terreno actual
        caminos = [v for v in transitables if self.terreno[v] == CAMINO]
        return caminos[0] if caminos else transitables[0]


class Maquina(ap.Agent):
    """Base comun: se mueve por una ruta A*, gira, se detiene y gasta gasolina."""

    velocidad = 1
    capacidad = 0
    gasolina_max = 300.0
    consumo = 1.0
    # penalizacion de varado por defecto para Q-learning; cada subclase la
    # sobreescribe con la propia (ver Qlearning/recompensas.py).
    rl_penalizacion_varado = rl_recompensas.PENALIZACION_VARADO_TRACTOR

    def setup(self):
        self.ruta = []
        self.gasolina = self.gasolina_max
        self.direccion = ESTE
        self.detenido = False
        self.carga = 0
        self.distancia = 0
        self.giros = 0
        self.recargas = 0
        # estado de Q-learning (solo se usa si el parametro correspondiente
        # de GranjaModel esta activado; sin activar, estos campos quedan
        # sin tocar y no cambian el comportamiento de la maquina de estados).
        self.rl_estado_pendiente = None
        self.rl_accion_pendiente = None
        self.rl_pasos_acumulados = 0
        self.rl_eventos_acumulados = 0.0
        self.rl_varado = False
        self.rl_costo_paso = rl_recompensas.COSTO_PASO

    def _rl_tick(self):
        self.rl_pasos_acumulados += 1

    def _rl_sumar_evento(self, valor):
        self.rl_eventos_acumulados += valor

    def _rl_marcar_varado(self):
        self.rl_varado = True

    def _rl_reset(self):
        self.rl_estado_pendiente = None
        self.rl_accion_pendiente = None
        self.rl_pasos_acumulados = 0
        self.rl_eventos_acumulados = 0.0
        self.rl_varado = False
        self.rl_costo_paso = rl_recompensas.COSTO_PASO

    def _rl_cerrar(self, politica, terminal, estado_siguiente=None):
        """Cierra la excursion de decision pendiente (si hay una) con una
        actualizacion de Q, y reinicia los acumuladores para la proxima.

        Una "excursion" es el tramo entre dos decisiones: el agente puede
        quedar encerrado varios pasos en un estado sin voz (`recargando`,
        `esperando_tractor`, `vertiendo`...) antes de volver a elegir algo,
        asi que la recompensa de la decision que lo mando ahi se arma recien
        aca, sumando lo que paso en todo ese tramo (ver Qlearning/recompensas.py).

        El costo por paso usado es `self.rl_costo_paso`, fijado al elegir la
        accion (ver Harvester.step/Tractor.step): mas bajo durante toda la
        excursion de ir a recargar que durante la de seguir operando, para
        que el viaje no compita en la misma escala que el costo de demora
        normal (ver COSTO_PASO_RECARGA en Qlearning/recompensas.py).
        """
        if self.rl_estado_pendiente is None:
            self._rl_reset()
            return
        recompensa = (self.rl_eventos_acumulados
                      - self.rl_costo_paso * self.rl_pasos_acumulados
                      - (self.rl_penalizacion_varado if self.rl_varado else 0.0))
        politica.actualizar(self.rl_estado_pendiente, self.rl_accion_pendiente,
                            recompensa, None if terminal else estado_siguiente)
        self._rl_reset()

    @property
    def ubicacion(self):
        return self.model.campo.positions[self]

    def definirRuta(self, meta, evitar_agentes=True, transitables=TRANSITABLE, costos=None):
        # los agentes parados-pero-con-gasolina no cuentan como obstaculo fijo
        # al planear (ver Modelo.celdas_ocupadas): evita que un tractor
        # escoltando en un camino angosto bloquee para siempre el unico
        # acceso a un rincon de cultivo.
        bloqueadas = (self.model.celdas_ocupadas(excepto=self, incluir_parados=False)
                      if evitar_agentes else frozenset())
        # silo y base son solo puntos de referencia visuales (el edificio),
        # nadie debe planear su ruta pisandolos: se para en punto_silo /
        # punto_base, la celda transitable de al lado (ver GranjaModel.setup).
        bloqueadas = bloqueadas | {self.model.silo, self.model.base}
        self.ruta = a_estrella(self.model.campo.terreno, self.ubicacion, meta, bloqueadas, transitables, costos)
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

    def necesitaGasolina(self):
        return self.gasolina <= self.gasolina_max * self.p.umbral_gasolina

    def cargarGasolina(self):
        self.gasolina = self.gasolina_max
        self.recargas += 1

    def moverse(self):
        """Avanza hasta `velocidad` celdas por paso. Devuelve las celdas recorridas."""
        recorridas = []
        if self.detenido or self.gasolina <= 0:
            return recorridas

        for _ in range(self.velocidad):
            if not self.ruta:
                break
            siguiente = self.ruta[0]
            # igual que al planear: un agente parado-pero-con-gasolina no
            # cuenta como obstaculo. Si contara, cuando la ruta lo atraviesa
            # (porque definirRuta ya lo ignoro) este chequeo la tumbaria cada
            # vez sin que nadie avance nunca -- un bloqueo mutuo con el
            # agente parado, que tampoco tiene motivo para moverse.
            if siguiente in self.model.celdas_ocupadas(excepto=self, incluir_parados=False):
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

    rl_penalizacion_varado = rl_recompensas.PENALIZACION_VARADO_HARVESTER

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
        self.zona = None
        self.averiado = False
        self.cosechado_total = 0  # celdas cosechadas acumuladas (no baja al vertir, a diferencia de self.carga)
        # penaliza (sin prohibir) pisar cultivo ya cosechado en los viajes
        # de ida/vuelta a la gasolinera: rodea por camino de tierra si el
        # desvio no sale mucho mas caro que ir en linea recta.
        self.costos_ruta = {COSECHADO: self.p.get('penalizacion_cosechado_harvester', 3.0)}

    def _zona_tiene_libres(self, zona, libres):
        # ojo: cuenta celdas realmente disponibles (sin reservar por otro
        # harvester), no solo LISTO en crudo, o reclamarZona() podria creer
        # reclamable una zona cuyo unico cultivo ya esta tomado por otro y
        # quedar en un ciclo infinito con buscar_objetivo.
        r0, r1, c0, c1 = zona
        return any(r0 <= f < r1 and c0 <= c < c1 for f, c in libres)

    def reclamarZona(self, libres):
        """Cuando ya no le queda cultivo disponible en su zona, reclama la
        zona libre (sin dueno activo) mas cercana que aun tenga alguna celda
        realmente disponible, para no invadir la zona de un hermano que
        sigue trabajando ahi."""
        disponibles = [i for i, dueno in enumerate(self.model.propietario_zona)
                       if dueno in (None, self) and self._zona_tiene_libres(self.model.zonas[i], libres)]
        if not disponibles:
            return False

        aqui = self.ubicacion

        def distancia(i):
            r0, r1, c0, c1 = self.model.zonas[i]
            return abs((r0 + r1) / 2 - aqui[0]) + abs((c0 + c1) / 2 - aqui[1])

        elegida = min(disponibles, key=distancia)
        if self.zona in self.model.zonas:
            idx_actual = self.model.zonas.index(self.zona)
            if self.model.propietario_zona[idx_actual] is self:
                self.model.propietario_zona[idx_actual] = None
        self.model.propietario_zona[elegida] = self
        self.zona = self.model.zonas[elegida]
        return True

    def _soltar_objetivo(self):
        # si el objetivo pendiente no se va a alcanzar, hay que liberar su
        # reserva aqui: si no, queda "fantasma" en model.reservadas para
        # siempre (nadie mas la va a soltar) y esa celda nunca se vuelve a
        # ofrecer a nadie, aunque siga siendo cultivo real sin cosechar.
        if self.objetivo is not None:
            self.model.reservadas.discard(self.objetivo)
        self.objetivo = None

    def buscar_objetivo(self):
        while True:
            libres = [c for c in self.model.campo.celdas_listas()
                      if c not in self.model.reservadas]
            if not libres:
                self._soltar_objetivo()
                return False

            r0, r1, c0, c1 = self.zona
            propias = [c for c in libres if r0 <= c[0] < r1 and c0 <= c[1] < c1]

            if propias:
                # barrido en serpentina (fila por fila, alternando sentido) para
                # cubrir toda la zona sin dejar huecos sueltos que haya que
                # revisitar despues, pisando cultivo ya cosechado.
                candidatas = sorted(propias, key=lambda c: (c[0], c[1] if c[0] % 2 == 0 else -c[1]))
            elif self.reclamarZona(libres):
                continue  # reintenta ya con la zona nueva
            else:
                # no quedan zonas propias ni reclamables: ayuda donde haga falta
                aqui = self.ubicacion
                candidatas = sorted(libres, key=lambda c: abs(c[0] - aqui[0]) + abs(c[1] - aqui[1]))

            # si todavia esta sobre camino de tierra (recien salio de la
            # gasolinera, por ejemplo) prefiere seguir por camino el mayor
            # trecho posible en vez de cortar por cultivo ya cosechado; una
            # vez que pisa cultivo esto deja de tener efecto (no hay camino
            # cerca para preferir) y vuelve al trayecto directo de siempre.
            costos = (self.costos_ruta
                      if self.model.campo.terreno[self.ubicacion] == CAMINO else None)

            for candidata in candidatas[:20]:
                if self.definirRuta(candidata, costos=costos):
                    self.model.reservadas.discard(self.objetivo)
                    self.objetivo = candidata
                    self.model.reservadas.add(candidata)
                    return True
            self._soltar_objetivo()
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
        self.estado = 'descompuesto' if self.averiado else 'operando'
        self.tractor_asignado = None
        self.reanudar()

    def _descomponer(self):
        """Falla mecanica permanente: deja de cosechar y cede su zona para
        que otro harvester la reclame (mismo mecanismo que usan al terminar
        una zona: `model.propietario_zona` la deja libre y `reclamarZona`
        de cualquier hermano la puede tomar). Si ya traia carga a bordo,
        todavia pide un tractor para no perder ese grano ya cosechado antes
        de quedar inmovil para siempre."""
        self.averiado = True
        self._soltar_objetivo()
        self.detenerse()
        if self.zona in self.model.zonas:
            idx = self.model.zonas.index(self.zona)
            if self.model.propietario_zona[idx] is self:
                self.model.propietario_zona[idx] = None
        if self.carga > 0:
            self.estado = 'esperando_tractor'
            self.llamarTractor()
        else:
            self.estado = 'descompuesto'

    def step(self):
        usar_rl = self.p.get('usar_qlearning_harvester', False)
        if usar_rl:
            self._rl_tick()

        if self.gasolina <= 0:
            # si justo se quedo sin gasolina al llegar a la base (moverse
            # gasta el ultimo tanque en el mismo paso que lo deja ahi), hay
            # que recargarlo aqui: si no, queda sin_gasolina para siempre
            # parado exactamente sobre la gasolinera, bloqueando a cualquiera
            # que necesite pasar por esa unica celda.
            if self.ubicacion == self.model.punto_base:
                self.cargarGasolina()
                self.estado = 'operando'
            else:
                self.estado = 'sin_gasolina'
                if usar_rl:
                    # peor desenlace posible para la decision pendiente:
                    # quedo varado e inmovil el resto de la corrida, no va a
                    # volver a decidir nada -- cierre terminal, sin bootstrap.
                    self._rl_marcar_varado()
                    self._rl_cerrar(self.model.politica_q_harvester, terminal=True)
                return

        if self.estado == 'descompuesto':
            return  # averia permanente: ya entrego lo que traia (si tenia), inmovil para siempre

        if self.estado == 'esperando_tractor':
            # si nadie estaba libre cuando llamo la primera vez, reintenta
            # cada paso: si no, se queda esperando para siempre aunque
            # despues se libere un tractor justo al lado.
            if self.tractor_asignado is None:
                self.llamarTractor()
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

        if self.estado == 'recargando':
            if self.ubicacion == self.model.punto_base:
                if usar_rl:
                    # proporcional a lo que realmente hacia falta: si no,
                    # un harvester ya parado sobre la base podria entrar y
                    # salir de 'recargando' sin moverse (excursion de costo
                    # ~0) y cobrar el bono completo cada vez con el tanque
                    # ya lleno -- un ciclo gratis que no cuesta nada y no
                    # cosecha nada (se observo en pruebas: un harvester
                    # recargando miles de veces sin avanzar).
                    frac_faltante = 1.0 - self.gasolina / self.gasolina_max
                    self._rl_sumar_evento(rl_recompensas.R_TANQUE_LLENADO * frac_faltante)
                self.cargarGasolina()
                self.estado = 'operando'
            else:
                if not self.ruta:
                    self.definirRuta(self.model.punto_base, costos=self.costos_ruta)
                self.moverse()
            return

        if self.carga >= self.capacidad:
            self.detenerse()
            self.estado = 'esperando_tractor'
            self.llamarTractor()
            return

        if self.estado == 'operando':
            if usar_rl:
                # punto de decision: reemplaza el umbral fijo de
                # necesitaGasolina() por la politica aprendida. Se decide en
                # todos los pasos operando (no solo cuando el tanque baja del
                # umbral) para que el agente tambien pueda aprender a
                # anticiparse. Antes de elegir, cierra la excursion anterior
                # (la que llevo hasta este punto de decision) con el estado
                # actual como resultado observado.
                estado_actual = harvester_rl.discretizar_estado(self)
                self._rl_cerrar(self.model.politica_q_harvester, terminal=False,
                                estado_siguiente=estado_actual)
                accion = self.model.politica_q_harvester.elegir_accion(estado_actual)
                self.rl_estado_pendiente = estado_actual
                self.rl_accion_pendiente = accion
                if accion == harvester_rl.IR_A_RECARGAR:
                    # costo por paso mas bajo durante esta excursion (ver
                    # COSTO_PASO_RECARGA en Qlearning/recompensas.py): el
                    # viaje a la base no debe competir en la misma escala
                    # que el costo de demora de seguir cosechando.
                    self.rl_costo_paso = rl_recompensas.COSTO_PASO_RECARGA
                    self._soltar_objetivo()
                    self.estado = 'recargando'
                    self.ruta = []
                    return
            elif self.necesitaGasolina():
                self._soltar_objetivo()
                self.estado = 'recargando'
                self.ruta = []
                return

        if self.p.prob_descompostura > 0 and self.model.nprandom.random() < self.p.prob_descompostura:
            self._descomponer()
            if usar_rl:
                # se rompio: como en el varado, no va a volver a decidir.
                self._rl_cerrar(self.model.politica_q_harvester, terminal=True)
            return

        if not self.ruta:
            self.buscar_objetivo()

        for celda in self.moverse():
            if self.model.campo.cosechar(celda):
                self.carga += 1
                self.cosechado_total += 1
                self.model.cosechado += 1
                self.model.reservadas.discard(celda)
                if usar_rl:
                    self._rl_sumar_evento(rl_recompensas.R_CELDA_COSECHADA)
                if self.carga >= self.capacidad:
                    break


class Tractor(Maquina):
    """Contenedor movil: sigue a un harvester mientras cosecha, acude a su
    llamada al llenarse, le recibe la carga y la lleva al silo."""

    rl_penalizacion_varado = rl_recompensas.PENALIZACION_VARADO_TRACTOR

    def setup(self):
        super().setup()
        self.velocidad = self.p.velocidad_tractor
        self.capacidad = self.p.capacidad_tractor
        self.gasolina_max = self.p.gasolina_tractor
        self.gasolina = self.gasolina_max
        self.consumo = self.p.consumo_tractor
        self.estado = 'libre'
        self.cliente = None
        self.harvester_seguido = None
        # penaliza (sin prohibir) pisar cultivo ya cosechado: rodea por
        # camino cuando el desvio no sale mucho mas caro que ir en linea recta
        self.costos_ruta = {COSECHADO: self.p.get('penalizacion_cosechado_tractor', 3.0)}

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

    def _ir_hacia(self, destino, transitables=TRANSITABLE, costos=None):
        """Se mueve hacia `destino`, recalculando la ruta si el objetivo cambio."""
        self.reanudar()
        if costos is None:
            costos = self.costos_ruta
        if not self.ruta or self.ruta[-1] != destino:
            self.definirRuta(destino, transitables=transitables, costos=costos)
        self.moverse()

    def _acercarse_a(self, harvester):
        """Se acerca al harvester sin llegar a pisar su celda: la ruta hacia
        su ubicacion se recorta un paso antes del final, asi queda a
        distancia 1 (ya vale para `adyacente_a`) en vez de superponerse con
        el cuando llega a recoger la carga."""
        destino = harvester.ubicacion
        self.reanudar()
        if not self.ruta or self.ruta[-1] != destino:
            self.definirRuta(destino, costos=self.costos_ruta)
            if self.ruta and self.ruta[-1] == destino:
                self.ruta.pop()
        self.moverse()

    def step(self):
        usar_rl = self.p.get('usar_qlearning_tractor', False)
        if usar_rl:
            self._rl_tick()

        if self.gasolina <= 0:
            # mismo caso que en Harvester: si el ultimo tanque alcanzo justo
            # para llegar al deposito, recargar aqui en vez de quedar
            # sin_gasolina parado para siempre sobre una celda que ademas es
            # de paso obligado para otros (silo o base). Silo y base son
            # celdas vecinas pero distintas: solo entrega grano si de
            # casualidad quedo tirado justo sobre el silo.
            if self.ubicacion == self.model.punto_silo and self.carga > 0:
                if usar_rl:
                    self._rl_sumar_evento(rl_recompensas.R_GRANO_ENTREGADO * self.carga)
                self.model.entregado += self.carga
                self.carga = 0
            if self.ubicacion in (self.model.punto_base, self.model.punto_silo):
                self.cargarGasolina()
                if self.cliente is not None:
                    if self.cliente.tractor_asignado is self:
                        self.cliente.tractor_asignado = None
                    self.cliente = None
                self.estado = 'libre'
            else:
                self.estado = 'sin_gasolina'
                if usar_rl:
                    self._rl_marcar_varado()
                    self._rl_cerrar(self.model.politica_q_tractor, terminal=True)
                return

        if self.carga >= self.capacidad and self.estado != 'descargando':
            self.cliente = None
            self.estado = 'al_silo'

        if self.estado == 'recargando':
            if self.ubicacion == self.model.punto_base:
                self.cargarGasolina()
                self.estado = 'libre'
            else:
                self._ir_hacia(self.model.punto_base)
            return

        if self.estado == 'al_silo':
            if self.ubicacion == self.model.punto_silo:
                if usar_rl and self.carga > 0:
                    self._rl_sumar_evento(rl_recompensas.R_GRANO_ENTREGADO * self.carga)
                self.model.entregado += self.carga
                self.carga = 0
                self.estado = 'libre'
            else:
                self._ir_hacia(self.model.punto_silo)
            return

        if self.estado == 'en_camino':
            harvester = self.cliente
            if harvester is None:
                self.estado = 'libre'
                return
            if self.necesitaGasolina():
                # todavia no recogio carga: mejor abortar e ir a recargar que
                # arriesgarse a quedar varado a mitad de camino sin poder
                # llegar ni de vuelta a la base. El harvester vuelve a pedir
                # tractor (Harvester.step reintenta cuando tractor_asignado
                # queda en None).
                if harvester.tractor_asignado is self:
                    harvester.tractor_asignado = None
                self.cliente = None
                self.estado = 'recargando'
                self.ruta = []
                return
            if self.adyacente_a(harvester):
                self.detenerse()
                self.estado = 'descargando'
                harvester.estado = 'vertiendo'
                harvester.tractor_asignado = self
            else:
                self._acercarse_a(harvester)
            return

        if self.estado == 'descargando':
            if self.cliente is None or self.cliente.estado != 'vertiendo':
                self.reanudar()
                self.liberar()
            return

        # estado 'libre' o 'escoltando': sin carga y sin llamada activa --
        # punto de decision para Q-learning (esperar/escoltar vs recargar
        # preventivamente). La asignacion de llamadas (que tractor atiende a
        # que harvester) sigue siendo por cercania via
        # GranjaModel.solicitar_tractor, sin Q-learning todavia.
        if self.carga > 0 and self.ubicacion != self.model.punto_silo:
            self.estado = 'al_silo'
            return

        if usar_rl:
            estado_actual = tractor_rl.discretizar_estado(self)
            self._rl_cerrar(self.model.politica_q_tractor, terminal=False,
                            estado_siguiente=estado_actual)
            accion = self.model.politica_q_tractor.elegir_accion(estado_actual)
            self.rl_estado_pendiente = estado_actual
            self.rl_accion_pendiente = accion
            if accion == tractor_rl.IR_A_RECARGAR:
                # mismo costo por paso reducido que Harvester.step (ver
                # COSTO_PASO_RECARGA en Qlearning/recompensas.py).
                self.rl_costo_paso = rl_recompensas.COSTO_PASO_RECARGA
                self.estado = 'recargando'
                self.ruta = []
                return
        elif self.necesitaGasolina():
            self.estado = 'recargando'
            self.ruta = []
            return

        self.seguirHarvester()

    def seguirHarvester(self):
        """Se queda cerca del harvester asignado, pero sin salirse de las
        casillas de camino, para no meterse a pisar el cultivo mientras
        espera su llamada."""
        objetivo = self.harvester_seguido
        if objetivo is None or objetivo not in self.model.harvesters or objetivo.estado == 'sin_gasolina':
            self.estado = 'libre'
            self.detenerse()
            return

        punto_espera = self.model.campo.camino_cercano(objetivo.ubicacion)
        if self.ubicacion == punto_espera or self.adyacente_a(objetivo):
            self.detenerse()
        else:
            self._ir_hacia(punto_espera, transitables=(CAMINO,))
        self.estado = 'escoltando'


class GranjaModel(ap.Model):

    def setup(self):
        self.nprandom = np.random.default_rng(self.p.get('seed', 1))
        self.campo = Campo(self, self.p.shape,
                           ancho_camino=self.p.ancho_camino,
                           pct_obstaculos=self.p.pct_obstaculos)

        # silo (entrega de grano) y base (gasolinera) en celdas de camino
        # distintas: si comparten una sola celda, un tractor descargando ahi
        # le tapa el paso a los harvesters que solo quieren recargar (y
        # viceversa). Van separadas por varias celdas (no solo vecinas): si
        # quedan a distancia 1, un tractor yendo de silo a base y otro yendo
        # de base a silo al mismo tiempo generan un swap imposible de
        # resolver (cada uno planea el unico paso directo hacia la celda
        # que el otro ocupa, el sensor lo rechaza cada vez, y ninguno prueba
        # un rodeo porque el camino directo "existe" sobre el papel). La
        # fila 0 siempre es camino perimetral completo sin importar
        # `ancho_camino`, asi que estas celdas estan garantizadas transitables.
        # `silo_pos`/`base_pos` permiten ubicarlas en otro lado siempre que
        # el llamador respete las mismas dos condiciones de arriba: caer en
        # camino y quedar separadas por varias celdas, no solo vecinas. Por
        # default van sobre la fila de la cruz de camino central (esa fila
        # es camino en todas sus columnas, ver Campo.setup), una a cada lado
        # del centro para no quedar vecinas entre si.
        filas, columnas = self.campo.shape
        centro_f, centro_c = filas // 2, columnas // 2
        separacion = max(4, columnas // 4)
        self.silo = self.p.get('silo_pos') or (centro_f, max(0, centro_c - separacion))
        self.base = self.p.get('base_pos') or (centro_f, min(columnas - 1, centro_c + separacion))

        # los agentes nunca planean su ruta pisando la celda del silo/base en
        # si (ver Maquina.definirRuta): se paran en la celda transitable de
        # al lado para cargar gasolina o descargar grano, no encima del
        # edificio.
        self.punto_silo = self.campo.celda_adyacente_transitable(self.silo)
        self.punto_base = self.campo.celda_adyacente_transitable(self.base)

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

        self.zonas = particionar_rectangulos(*self.campo.shape, len(self.harvesters))
        self.propietario_zona = [None] * len(self.zonas)
        disponibles = list(range(len(self.zonas)))
        for harvester in self.harvesters:
            aqui = harvester.ubicacion

            def distancia(i, aqui=aqui):
                r0, r1, c0, c1 = self.zonas[i]
                return abs((r0 + r1) / 2 - aqui[0]) + abs((c0 + c1) / 2 - aqui[1])

            elegida = min(disponibles, key=distancia)
            disponibles.remove(elegida)
            self.propietario_zona[elegida] = harvester
            harvester.zona = self.zonas[elegida]

        for i, tractor in enumerate(self.tractores):
            tractor.harvester_seguido = self.harvesters[i % len(self.harvesters)]

        # Q-learning: desactivado por defecto (ver PARAMETROS). Cuando esta
        # activo, retoma la tabla guardada por una corrida anterior (asi el
        # aprendizaje se acumula corrida a corrida) en vez de arrancar de
        # cero cada vez; `end()` la vuelve a guardar al terminar.
        self.usar_qlearning_harvester = self.p.get('usar_qlearning_harvester', False)
        self.usar_qlearning_tractor = self.p.get('usar_qlearning_tractor', False)
        alpha = self.p.get('qlearning_alpha', rl_config.ALPHA)
        gamma = self.p.get('qlearning_gamma', rl_config.GAMMA)
        epsilon = self.p.get('qlearning_epsilon', rl_config.EPSILON)

        if self.usar_qlearning_harvester:
            tabla = self._cargar_o_crear_tabla_q(rl_config.RUTA_TABLA_HARVESTER,
                                                 harvester_rl.crear_tabla_inicial)
            self.politica_q_harvester = PoliticaQ(tabla, harvester_rl.ACCIONES,
                                                  alpha, gamma, epsilon, rng=self.nprandom)

        if self.usar_qlearning_tractor:
            tabla = self._cargar_o_crear_tabla_q(rl_config.RUTA_TABLA_TRACTOR,
                                                 tractor_rl.crear_tabla_inicial)
            # de las 3 acciones definidas en tractor_rl, por ahora solo se
            # entrena la decision propia del tractor (esperar/recargar);
            # ATENDER_LLAMADA queda definida para una integracion futura con
            # el despacho de GranjaModel.solicitar_tractor.
            acciones = (tractor_rl.ESPERAR, tractor_rl.IR_A_RECARGAR)
            self.politica_q_tractor = PoliticaQ(tabla, acciones,
                                                alpha, gamma, epsilon, rng=self.nprandom)

    @staticmethod
    def _cargar_o_crear_tabla_q(ruta, crear_inicial):
        if os.path.exists(ruta):
            return rl_q_table.cargar(ruta)
        return crear_inicial()

    def celdas_ocupadas(self, excepto=None, incluir_parados=True):
        """Celdas con un agente encima. Con `incluir_parados=False` ignora los
        agentes detenidos-pero-con-gasolina (tractor escoltando, harvester
        esperando su tractor, etc.): siguen respetandose al mismo tiempo con
        el sensor de proximidad en `Maquina.moverse`, pero no cuentan como
        obstaculo fijo al planear toda la ruta, para no generar un bloqueo
        permanente si uno de ellos queda parado justo en el unico acceso a
        un rincon de cultivo."""
        ocupadas = set()
        for agente, pos in self.campo.positions.items():
            if agente is excepto:
                continue
            if not incluir_parados and agente.detenido and agente.gasolina > 0:
                continue
            ocupadas.add(pos)
        return ocupadas

    def solicitar_tractor(self, harvester):
        """Asignacion por cercania: el tractor libre mas proximo atiende la llamada.

        Exige gasolina por encima del umbral de recarga (no solo > 0): un
        tractor casi vacio que aceptara el viaje podria quedarse tirado a
        mitad de camino sin llegar ni a la cosechadora ni de vuelta a la base.
        """
        libres = [t for t in self.tractores
                  if t.estado in ('libre', 'escoltando')
                  and t.carga < t.capacidad
                  and not t.necesitaGasolina()]
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
        self.report('recargas_totales', int(sum(self.harvesters.recargas)
                                            + sum(self.tractores.recargas)))

        # guarda las tablas Q actualizadas para que la proxima corrida
        # retome el aprendizaje de esta en vez de arrancar de cero.
        if self.usar_qlearning_harvester:
            rl_q_table.guardar(self.politica_q_harvester.tabla, rl_config.RUTA_TABLA_HARVESTER)
        if self.usar_qlearning_tractor:
            rl_q_table.guardar(self.politica_q_tractor.tabla, rl_config.RUTA_TABLA_TRACTOR)


PARAMETROS = {
    'shape': (30, 30),
    'ancho_camino': 2,
    'pct_obstaculos': 0.02,
    'n_harvesters': 3,
    'n_tractores': 2,
    'tamano_harvester': 2,
    'velocidad_harvester': 1,
    'capacidad_harvester': 25,
    'gasolina_harvester': 900,
    'consumo_harvester': 1.0,
    'velocidad_tractor': 2,
    'capacidad_tractor': 80,
    'gasolina_tractor': 600,
    'consumo_tractor': 0.7,
    'tasa_vertido': 8,
    'umbral_gasolina': 0.2,
    'penalizacion_cosechado_tractor': 3.0,
    'penalizacion_cosechado_harvester': 3.0,
    'prob_descompostura': 0.0,
    # None = ubicacion por defecto en la esquina (ver GranjaModel.setup);
    # se puede sobreescribir con cualquier celda de camino, siempre que
    # ambas queden separadas por varias celdas (no solo vecinas).
    'silo_pos': None,
    'base_pos': None,
    'seed': 1,
    'steps': 500,
    # desactivado por defecto: con esto en False el modelo se comporta
    # exactamente igual que antes (logica de umbral fijo). Ver Qlearning/.
    'usar_qlearning_harvester': False,
    'usar_qlearning_tractor': False,
    'qlearning_alpha': rl_config.ALPHA,
    'qlearning_gamma': rl_config.GAMMA,
    'qlearning_epsilon': rl_config.EPSILON,
}
