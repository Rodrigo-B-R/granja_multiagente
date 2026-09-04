"""Puente WebSocket entre GranjaModel (granja.py) y el visualizador en Unity.

Python corre la simulacion paso a paso (agentpy `sim_setup`/`sim_step`) y hace
push del estado por WebSocket: un mensaje `init` con el entorno completo al
conectar (o al reiniciar), un mensaje `paso` por cada paso simulado (agentes +
celdas recien cosechadas, no el grid completo), y un mensaje `fin` con los
reportes.

La comunicacion es bidireccional: Unity puede mandar comandos de control en
cualquier momento, en el mismo socket, como mensajes JSON `{"tipo": ...}`:

    {"tipo": "pausar"}
    {"tipo": "reanudar"}
    {"tipo": "reiniciar", "parametros": {"shape": [F, C], "n_harvesters": N,
                                          "n_tractores": M, "seed": S,
                                          "steps": ST}}

`reiniciar` puede traer cualquier subconjunto de PARAMETROS (ver mas abajo);
los que no se manden conservan su valor actual. Como el grid y el numero de
agentes son fijos una vez creado el modelo de agentpy, reiniciar siempre
recrea el GranjaModel desde cero y vuelve a mandar un `init` fresco, sin
cerrar el socket.

Uso:
    python puente_unity.py [--host localhost] [--port 8765] [--intervalo 0.2]
                            [--shape F C] [--steps N] [--seed S]
"""

import argparse
import asyncio
import contextlib
import json
import random

import numpy as np
import websockets

from granja import GranjaModel, PARAMETROS, CAMINO, COSECHADO, LISTO, OBSTACULO


def _agente_init(agente, clase):
    fila, col = agente.ubicacion
    return {
        'id': agente.id,
        'clase': clase,
        'fila': int(fila),
        'col': int(col),
        'capacidad': agente.capacidad,
        'gasolina_max': agente.gasolina_max,
    }


def _construir_init(model, seed):
    terreno = model.campo.terreno

    # variedad visual (roca vs poste) puramente cosmetica: RNG propio,
    # separado de model.nprandom, para no alterar el azar de la simulacion.
    rng_visual = random.Random(seed)
    obstaculos = [
        {'fila': int(fila), 'col': int(col), 'tipo_visual': rng_visual.randint(0, 1)}
        for fila, col in np.argwhere(terreno == OBSTACULO)
    ]
    trigo_listo = [[int(fila), int(col)] for fila, col in np.argwhere(terreno == LISTO)]
    camino = [[int(fila), int(col)] for fila, col in np.argwhere(terreno == CAMINO)]

    return {
        'tipo': 'init',
        'filas': int(terreno.shape[0]),
        'columnas': int(terreno.shape[1]),
        'tam_celda': 1.0,
        'silo': list(model.silo),
        'base': list(model.base),
        'obstaculos': obstaculos,
        'trigo_listo': trigo_listo,
        'camino': camino,
        'harvesters': [_agente_init(h, 'harvester') for h in model.harvesters],
        'tractores': [_agente_init(t, 'tractor') for t in model.tractores],
    }


def _estado_agente(agente, clase):
    fila, col = agente.ubicacion
    return {
        'id': agente.id,
        'clase': clase,
        'fila': int(fila),
        'col': int(col),
        'direccion': [int(d) for d in agente.direccion],
        'estado': agente.estado,
        'gasolina': round(float(agente.gasolina), 2),
        'carga': int(agente.carga),
    }


def _construir_paso(model, mascara_cosechado_previo):
    terreno = model.campo.terreno
    mascara_actual = terreno == COSECHADO
    nuevas = np.argwhere(mascara_actual & ~mascara_cosechado_previo)
    mascara_cosechado_previo[:] = mascara_actual

    agentes = [_estado_agente(h, 'harvester') for h in model.harvesters]
    agentes += [_estado_agente(t, 'tractor') for t in model.tractores]

    return {
        'tipo': 'paso',
        't': model.t,
        'agentes': agentes,
        'cosechadas': [[int(fila), int(col)] for fila, col in nuevas],
        'metricas': {
            'cosechado_pct': 100 * model.cosechado / model.campo.total_cultivo,
            'grano_entregado': model.entregado,
            'gasolina_total': float(sum(model.harvesters.gasolina)
                                    + sum(model.tractores.gasolina)),
        },
    }


class ControlSimulacion:
    """Estado compartido entre la tarea que corre la simulacion y la que
    escucha comandos del cliente, para el mismo websocket."""

    def __init__(self, parametros):
        self.parametros = dict(parametros)
        self.pausado = asyncio.Event()
        self.reiniciar = asyncio.Event()
        self.cerrado = False


async def _escuchar_comandos(websocket, control):
    try:
        async for mensaje in websocket:
            try:
                datos = json.loads(mensaje)
            except json.JSONDecodeError:
                print(f'Comando invalido (no es JSON): {mensaje!r}')
                continue

            tipo = datos.get('tipo')
            if tipo == 'pausar':
                control.pausado.set()
            elif tipo == 'reanudar':
                control.pausado.clear()
            elif tipo == 'reiniciar':
                nuevos = datos.get('parametros') or {}
                if 'shape' in nuevos:
                    nuevos['shape'] = tuple(nuevos['shape'])
                control.parametros.update(nuevos)
                control.pausado.clear()
                control.reiniciar.set()
            else:
                print(f'Comando desconocido del cliente: {tipo!r}')
    except websockets.ConnectionClosed:
        pass
    finally:
        control.cerrado = True
        control.reiniciar.set()  # despierta el loop de simulacion si estaba en pausa


async def _correr_simulacion(websocket, control, intervalo):
    while not control.cerrado:
        control.reiniciar.clear()

        # nunca aprendizaje automatico en el puente, sin importar lo que
        # traiga `parametros`: es una visualizacion de la politica fija.
        p = dict(control.parametros, usar_qlearning_harvester=False,
                 usar_qlearning_tractor=False)
        seed = p.get('seed') or 1
        model = GranjaModel(p)
        model.sim_setup(steps=p.get('steps'), seed=seed)
        await websocket.send(json.dumps(_construir_init(model, seed)))

        mascara_cosechado_previo = model.campo.terreno == COSECHADO
        terminado_naturalmente = False
        while model.running:
            if control.reiniciar.is_set() or control.cerrado:
                break
            if control.pausado.is_set():
                await asyncio.sleep(0.05)
                continue
            model.sim_step()
            await websocket.send(json.dumps(_construir_paso(model, mascara_cosechado_previo)))
            await asyncio.sleep(intervalo)
        else:
            terminado_naturalmente = True

        if control.reiniciar.is_set() and not control.cerrado:
            continue  # vuelve a armar el modelo con los parametros nuevos

        model.end()
        if not control.cerrado and terminado_naturalmente:
            await websocket.send(json.dumps({'tipo': 'fin', 'reportes': dict(model.reporters)}))
        break


async def _manejar_cliente(websocket, parametros, intervalo):
    control = ControlSimulacion(parametros)
    tarea_comandos = asyncio.create_task(_escuchar_comandos(websocket, control))
    try:
        await _correr_simulacion(websocket, control, intervalo)
    finally:
        tarea_comandos.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tarea_comandos


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--intervalo', type=float, default=0.6,
                        help='segundos reales de espera entre pasos simulados')
    parser.add_argument('--shape', type=int, nargs=2, metavar=('FILAS', 'COLUMNAS'))
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    parametros = dict(PARAMETROS)
    if args.shape:
        parametros['shape'] = tuple(args.shape)
    if args.steps:
        parametros['steps'] = args.steps
    if args.seed:
        parametros['seed'] = args.seed

    async def manejador(websocket):
        print('Cliente Unity conectado, arrancando simulacion...')
        await _manejar_cliente(websocket, parametros, args.intervalo)
        print('Simulacion terminada.')

    async with websockets.serve(manejador, args.host, args.port):
        print(f'Puente WS escuchando en ws://{args.host}:{args.port}')
        await asyncio.Future()


if __name__ == '__main__':
    asyncio.run(main())
