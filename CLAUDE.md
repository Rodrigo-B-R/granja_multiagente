# granja_multiagente

Simulación multiagente de una cosecha agrícola con [agentpy](https://agentpy.readthedocs.io/) (0.1.5). Sobre un grid (`ap.Grid`) se mueven `Harvester`s (cosechadoras) y `Tractor`s (acarreo) que cosechan un campo, evitan obstáculos y colisiones, y gestionan gasolina y descarga en un silo.

## Archivos

- `granja.py` — todo el modelo: terreno, A*, agentes, `GranjaModel`, y `PARAMETROS` por defecto. Es la única fuente de verdad del comportamiento; los notebooks solo importan desde aquí.
- `granja_v2.ipynb` — notebook de análisis/visualización: corre el modelo con distintos parámetros, grafica métricas (`cosechado_pct`, `grano_entregado`, `gasolina_total`) y genera animaciones (`ap.animate`) guardadas como GIF numerado en `GIFS/` (no pisa corridas anteriores).
- `GIFS/` — salidas de animación generadas por el notebook.
- `puente_unity.py` — servidor WebSocket que corre `GranjaModel` paso a paso (`sim_setup`/`sim_step` de agentpy) y hace *push* del estado al visualizador 3D en Unity (`Simulacion-Cosecha-Equipo-1`): un mensaje `init` con el entorno completo (incluye `camino` con las celdas de camino de tierra, no solo `obstaculos`/`trigo_listo`) al conectar, un mensaje `paso` por cada paso simulado (agentes + solo las celdas recién cosechadas, no el grid completo) y un mensaje `fin` con `model.reporters`. Fuerza `usar_qlearning_harvester`/`usar_qlearning_tractor` en `False` sin importar los parámetros que reciba — el puente es para visualizar una política fija, nunca para entrenar. La variedad visual de obstáculos (roca vs poste de riego) es puramente cosmética, asignada por un `random.Random(seed)` propio del puente en el mensaje `init` (`tipo_visual`); `granja.py` sigue teniendo un solo tipo de terreno `OBSTACULO`. Uso: `python puente_unity.py [--host] [--port] [--intervalo] [--shape F C] [--steps N] [--seed S]`.
- `.venv/` — entorno virtual local (no tocar/versionar).

`requirements.txt` — dependencias fijadas por versión: `agentpy`, `numpy`, `matplotlib`, `pillow`, `ipython`/`ipykernel` (notebook), `imageio-ffmpeg` (guardar animaciones como video en `granja_v3.ipynb`) y `websockets` (`puente_unity.py`).

## Modelo (`granja.py`)

**Terreno (`Campo`, subclase de `ap.Grid`)**: matriz `terreno` con valores `CAMINO`, `LISTO` (cultivo), `COSECHADO`, `OBSTACULO`. Caminos en el perímetro y en cruz central (ancho `ancho_camino`); obstáculos aleatorios (`pct_obstaculos`) sobre celdas de cultivo.

**Pathfinding**: `a_estrella(terreno, inicio, meta, bloqueadas)` — A* estándar en grid 4-direcciones, evita `OBSTACULO` y celdas `bloqueadas` (usado para no chocar con otros agentes).

**`Maquina`** (base de `Harvester`/`Tractor`, subclase de `ap.Agent`): gasolina, dirección, ruta A*, conteo de giros/distancia/recargas. `moverse()` avanza hasta `velocidad` celdas/paso; si la siguiente celda de la ruta está ocupada por otro agente, aborta la ruta (replanea el próximo paso) — es el mecanismo anticolisión.

**Ocupación "blanda" para evitar bloqueos mutuos**: `GranjaModel.celdas_ocupadas(..., incluir_parados=False)` deja de contar como obstáculo a un agente **detenido pero con gasolina** (tractor escoltando, harvester esperando su tractor, etc.). Tanto `definirRuta` (planeación) como el sensor de proximidad dentro de `moverse` (chequeo en tiempo real) usan ese mismo criterio — si solo uno de los dos fuera laxo, un agente parado justo en el único acceso a un rincón de cultivo generaría un bloqueo mutuo permanente (el que planea encuentra ruta pero el sensor la tumba cada vez, sin que nadie avance). Un agente con `gasolina <= 0` sí sigue contando siempre como obstáculo duro.

**`Harvester`**: máquina de estados (`operando`, `esperando_tractor`, `vertiendo`, `recargando`, `sin_gasolina`, `descompuesto`).
- Zonas: `GranjaModel.setup` particiona el campo en `n_harvesters` rectángulos compactos (`particionar_rectangulos`, bisección tipo guillotina — bloques grandes y contiguos, no franjas de columnas) y asigna cada uno al harvester más cercano (`model.zonas` + `model.propietario_zona`).
- `buscar_objetivo()`: dentro de su zona, recorre en **serpentina** (fila por fila, alternando sentido) las celdas `LISTO` no reservadas, para cubrir toda la zona sin dejar huecos sueltos que obliguen a volver después pisando cultivo ya cosechado. `model.reservadas` sigue siendo compartido entre todos.
- `reclamarZona()`: cuando su zona se queda sin cultivo, reclama la zona más cercana que no tenga dueño activo y aún tenga cultivo (transfiere la propiedad en `model.propietario_zona`), evitando invadir la zona de un hermano que sigue trabajando ahí. Si no hay ninguna reclamable, cae al comportamiento anterior de ayudar en cualquier celda libre del mapa.
- Al llenarse, se detiene y llama a un tractor (`llamarTractor` → `model.solicitar_tractor`); vierte carga cuando el tractor llega.
- `cosechado_total`: contador acumulado de celdas cosechadas por el propio harvester (a diferencia de `carga`, no baja al vertir en el tractor) — pensado para reportar/graficar cuánto grano produjo cada uno.
- Si la gasolina baja del umbral (`umbral_gasolina`) y no está a mitad de una entrega, se desvía a `model.base` a recargar.
- **Averías** (`p.prob_descompostura`, default `0.0` — desactivado): cada paso que está `operando` (no al llenarse, ni recargando, ni entregando), tira este dado; si sale, `_descomponer()` la deja `averiado=True` para siempre. Suelta su celda reservada (`_soltar_objetivo`) y cede la propiedad de su zona en `model.propietario_zona` (mismo mecanismo de `reclamarZona` que usan al terminar su propia zona — no hace falta lógica nueva para que un hermano la reclame). Si ya traía carga a bordo, llama a un tractor igual que al llenarse, para no perder ese grano; una vez entregado (o si no traía nada), queda en `descompuesto` inmóvil el resto de la corrida. `pararVertimiento()` decide a qué estado volver (`'descompuesto'` si `self.averiado`, si no `'operando'`) — por eso el flag vive aparte de `estado`, no como un valor más de `estado`. Si **todos** los harvesters se averían antes de terminar el campo, la corrida se queda sin completar — resultado esperado del azar, no un bug (confirmar mirando si el `%` queda congelado con todos los harvesters en `descompuesto`/`sin_gasolina`, no si simplemente faltan pasos).

**`Tractor`**: máquina de estados (`libre`, `escoltando`, `en_camino`, `descargando`, `al_silo`, `recargando`, `sin_gasolina`).
- Cuando está libre, `seguirHarvester()` lo acerca al harvester asignado (`harvester_seguido`) pero **sin salirse de las casillas de camino**: calcula la celda de camino más cercana a la posición del harvester (`Campo.camino_cercano`) y solo se mueve por camino (`definirRuta(..., transitables=(CAMINO,))`), nunca pisa cultivo mientras espera.
- Al ser llamado (`en_camino`), sí puede atravesar cultivo para llegar hasta el harvester (necesario, ya que el harvester está en pleno campo), pero se acerca con `_acercarse_a()` en vez de `_ir_hacia()`: recorta el último paso de la ruta para quedar a distancia 1 (ya alcanza para `adyacente_a`) en vez de terminar exactamente sobre la celda del harvester. Sin este recorte, como un harvester detenido cuenta como obstáculo "blando" (ver más abajo), el tractor podía planear una ruta que terminaba pisando la celda exacta del harvester antes de que el chequeo de adyacencia lo detuviera. Descarga (`descargando`), y si se llena o el harvester termina, va a `model.silo` a entregar (`entregado`).
- En todo desplazamiento con terreno completo (recargar, ir al silo, ir al harvester que llamó) usa `self.costos_ruta = {COSECHADO: p.penalizacion_cosechado_tractor}` para que A* prefiera rodear por camino en vez de cruzar cultivo ya cosechado, siempre que el rodeo no salga desproporcionadamente más caro que la penalización.

**`GranjaModel`** (`ap.Model`): crea `Campo`, coloca harvesters/tractores en celdas de camino aleatorias, arma `zonas`/`propietario_zona` y asigna parejas harvester↔tractor. `solicitar_tractor` asigna el tractor libre más cercano. `step()` corre hasta que `cosechado >= total_cultivo`. `update()`/`end()` registran métricas (`cosechado_pct`, `grano_entregado`, `gasolina_total`, `distancia_total`, `combustible_usado`, `giros_totales`, `recargas_totales`).

**`model.silo` y `model.base` son celdas de camino distintas y separadas** (fila 0, columnas 0 y `min(3, columnas-1)`) — no comparten ubicación. Motivos, en orden de descubrimiento:
1. Si comparten una sola celda, un tractor descargando ahí le tapa el paso a los harvesters que solo quieren recargar (y viceversa) — un único punto todo-en-uno se vuelve cuello de botella.
2. Si quedan a distancia 1 (vecinas), un tractor yendo de silo a base y otro yendo de base a silo al mismo tiempo generan un *swap* irresoluble: cada uno planea el único paso directo hacia la celda que el otro ocupa (la meta siempre está exenta del filtro de celdas bloqueadas en `a_estrella`, así el agente pueda aproximarse a un destino ocupado), el sensor de `moverse` lo rechaza cada tick, y ninguno prueba un rodeo porque el camino directo "existe" sobre el papel. Por eso van separadas por varias celdas, no solo vecinas.

Como consecuencia, la ruta de "gas exactamente 0 justo al llegar" (`Tractor.step`) ya no puede asumir que base y silo son el mismo punto: solo entrega la carga si el tractor quedó exactamente sobre `model.silo`, y solo recarga si está sobre `model.base` o `model.silo` (tratando el área del depósito como una sola zona de rescate aunque sean dos celdas).

Sus posiciones son configurables via los parámetros `silo_pos`/`base_pos` (`None` por defecto = la esquina descrita arriba). Cualquier valor que se pase debe seguir respetando las mismas dos condiciones: caer en una celda de camino y quedar separado del otro por varias celdas (p.ej. dos puntos sobre la cruz de camino central, no solo vecinos).

**`a_estrella`** acepta `transitables` (por defecto `TRANSITABLE` = camino+listo+cosechado) para restringir una ruta a un subconjunto de terreno (p.ej. el camino puro que usa el tractor al escoltar), y `costos` (dict terreno→costo por celda, default 1) para *penalizar sin prohibir* cierto tipo de terreno — así el tractor puede seguir cruzando cultivo cosechado cuando es indispensable, pero prefiere rodear por camino cuando el desvío no sale mucho más caro.

**`PARAMETROS`**: diccionario con la configuración por defecto (shape 40×40, 3 harvesters, 2 tractores, etc.) — se copia y sobreescribe (`dict(PARAMETROS)`) en el notebook para cada corrida distinta, nunca se muta directamente.

## Convenciones

- Todo el código (nombres de variables, funciones, docstrings, comentarios) está en español. Mantener esa convención al editar `granja.py`.
- Estados de agentes son strings literales comparados directamente (no Enum) — si se agrega un estado nuevo, actualizar también los diccionarios de color en el notebook (`COLOR_HARVESTER`/`COLOR_TRACTOR`) y cualquier lugar que enumere estados válidos.
- Las celdas objetivo reservadas (`model.reservadas`) deben liberarse (`discard`) en todo camino que abandone o complete un objetivo — al cosechar, al perder gasolina, al no poder trazar ruta — para no dejar celdas fantasma bloqueadas. Usar `Harvester._soltar_objetivo()` en vez de `self.objetivo = None` a pelo: hacerlo a pelo fue justo el bug que dejaba una reserva fantasma cuando la única celda de cultivo restante en todo el mapa era el propio objetivo del harvester (nadie más la liberaba nunca).
- Un agente nunca debe poder quedar `sin_gasolina` de forma permanente por descuido de contabilidad: `Harvester.step`/`Tractor.step` recargan primero si `gasolina <= 0` pero `ubicacion == model.base` (el último tanque puede alcanzar justo para llegar) antes de declarar `sin_gasolina`; `solicitar_tractor` exige `not t.necesitaGasolina()` (no solo `> 0`) para no asignar un tractor casi vacío a un viaje que no puede completar; y `Tractor` en `en_camino` aborta a recargar si `necesitaGasolina()` antes de comprometerse a cruzar el campo.
- Los notebooks importan el modelo (`from granja import GranjaModel, PARAMETROS`) en vez de duplicar lógica; cambios de comportamiento van en `granja.py`.

## Verificación de cambios

No hay suite de tests. Para validar cambios en `granja.py`, correr una simulación corta desde el notebook o con un script ad-hoc, p. ej.:

```python
from granja import GranjaModel, PARAMETROS
m = GranjaModel(dict(PARAMETROS, shape=(20, 20), steps=500))
r = m.run()
print(r.reporters)
```

Confirmar que `cosechado_pct` llega a 100 y que no quedan agentes en `sin_gasolina` de forma permanente ni loops de `reservadas` sin liberar.

Cualquier cambio a la lógica de movimiento, zonas o gasolina debe revalidarse con una corrida de muchos seeds (no solo uno), porque los bloqueos mutuos suelen depender del layout aleatorio de obstáculos/posiciones iniciales y no aparecen siempre:

```python
from granja import GranjaModel, PARAMETROS
fallos = []
for seed in range(1, 51):
    m = GranjaModel(dict(PARAMETROS, shape=(24, 24), steps=1500, seed=seed))
    r = m.run(display=False)
    if r.reporters['cosechado_pct'][0] < 100:
        fallos.append(seed)
print(fallos)  # debe quedar vacio
```

Si un seed falla, no asumir que es lento: correr ese mismo seed con `steps` mucho más grande (10-20x) para distinguir "necesitaba más pasos" (el % sigue subiendo) de un bloqueo real (el % y el estado de todos los agentes quedan exactamente congelados de un tick a otro).

**Límite conocido, no arreglado**: el umbral de recarga (`umbral_gasolina`) es una fracción fija del tanque, no una estimación de distancia real a la base. Con tanques muy chicos y radios de operación grandes relativos a ese umbral (p.ej. `gasolina=100`, `umbral=0.3` en un mapa 26×26), un harvester puede cruzar el umbral estando ya más lejos de lo que ese remanente alcanza para volver, y quedar `sin_gasolina` a mitad de camino — no es un bloqueo mutuo, es que el presupuesto de combustible no alcanza físicamente. Mantener `gasolina_max * umbral_gasolina` con margen holgado sobre la distancia máxima esperada desde `model.base`.
