# granja_multiagente

Simulación multiagente de una cosecha agrícola con [agentpy](https://agentpy.readthedocs.io/) (0.1.5). Sobre un grid (`ap.Grid`) se mueven `Harvester`s (cosechadoras) y `Tractor`s (acarreo) que cosechan un campo, evitan obstáculos y colisiones, y gestionan gasolina y descarga en un silo.

## Archivos

- `granja.py` — todo el modelo: terreno, A*, agentes, `GranjaModel`, y `PARAMETROS` por defecto. Es la única fuente de verdad del comportamiento; los notebooks solo importan desde aquí.
- `granja_v2.ipynb` — notebook de análisis/visualización: corre el modelo con distintos parámetros, grafica métricas (`cosechado_pct`, `grano_entregado`, `gasolina_total`) y genera animaciones (`ap.animate`) guardadas como GIF numerado en `GIFS/` (no pisa corridas anteriores).
- `GIFS/` — salidas de animación generadas por el notebook.
- `.venv/` — entorno virtual local (no tocar/versionar).

No hay `requirements.txt`; dependencias inferidas del entorno: `agentpy`, `numpy`, `matplotlib`, `ipython` (para el notebook).

## Modelo (`granja.py`)

**Terreno (`Campo`, subclase de `ap.Grid`)**: matriz `terreno` con valores `CAMINO`, `LISTO` (cultivo), `COSECHADO`, `OBSTACULO`. Caminos en el perímetro y en cruz central (ancho `ancho_camino`); obstáculos aleatorios (`pct_obstaculos`) sobre celdas de cultivo.

**Pathfinding**: `a_estrella(terreno, inicio, meta, bloqueadas)` — A* estándar en grid 4-direcciones, evita `OBSTACULO` y celdas `bloqueadas` (usado para no chocar con otros agentes).

**`Maquina`** (base de `Harvester`/`Tractor`, subclase de `ap.Agent`): gasolina, dirección, ruta A*, conteo de giros/distancia/recargas. `moverse()` avanza hasta `velocidad` celdas/paso; si la siguiente celda de la ruta está ocupada por otro agente, aborta la ruta (replanea el próximo paso) — es el mecanismo anticolisión.

**`Harvester`**: máquina de estados (`operando`, `esperando_tractor`, `vertiendo`, `recargando`, `sin_gasolina`).
- Zonas: `GranjaModel.setup` particiona el campo en `n_harvesters` rectángulos compactos (`particionar_rectangulos`, bisección tipo guillotina — bloques grandes y contiguos, no franjas de columnas) y asigna cada uno al harvester más cercano (`model.zonas` + `model.propietario_zona`).
- `buscar_objetivo()`: dentro de su zona, recorre en **serpentina** (fila por fila, alternando sentido) las celdas `LISTO` no reservadas, para cubrir toda la zona sin dejar huecos sueltos que obliguen a volver después pisando cultivo ya cosechado. `model.reservadas` sigue siendo compartido entre todos.
- `reclamarZona()`: cuando su zona se queda sin cultivo, reclama la zona más cercana que no tenga dueño activo y aún tenga cultivo (transfiere la propiedad en `model.propietario_zona`), evitando invadir la zona de un hermano que sigue trabajando ahí. Si no hay ninguna reclamable, cae al comportamiento anterior de ayudar en cualquier celda libre del mapa.
- Al llenarse, se detiene y llama a un tractor (`llamarTractor` → `model.solicitar_tractor`); vierte carga cuando el tractor llega.
- Si la gasolina baja del umbral (`umbral_gasolina`) y no está a mitad de una entrega, se desvía a `model.base` a recargar.

**`Tractor`**: máquina de estados (`libre`, `escoltando`, `en_camino`, `descargando`, `al_silo`, `recargando`, `sin_gasolina`).
- Cuando está libre, `seguirHarvester()` lo acerca al harvester asignado (`harvester_seguido`) pero **sin salirse de las casillas de camino**: calcula la celda de camino más cercana a la posición del harvester (`Campo.camino_cercano`) y solo se mueve por camino (`definirRuta(..., transitables=(CAMINO,))`), nunca pisa cultivo mientras espera.
- Al ser llamado (`en_camino`), sí puede atravesar cultivo para llegar hasta el harvester (necesario, ya que el harvester está en pleno campo); descarga (`descargando`), y si se llena o el harvester termina, va a `model.silo` a entregar (`entregado`).

**`GranjaModel`** (`ap.Model`): crea `Campo`, coloca harvesters/tractores en celdas de camino aleatorias, arma `zonas`/`propietario_zona` y asigna parejas harvester↔tractor. `silo` y `base` (gasolinera) comparten ubicación. `solicitar_tractor` asigna el tractor libre más cercano. `step()` corre hasta que `cosechado >= total_cultivo`. `update()`/`end()` registran métricas (`cosechado_pct`, `grano_entregado`, `gasolina_total`, `distancia_total`, `combustible_usado`, `giros_totales`, `recargas_totales`).

**`a_estrella`** acepta un parámetro `transitables` (por defecto `TRANSITABLE` = camino+listo+cosechado) para restringir una ruta a un subconjunto de terreno, como el camino puro que usa el tractor al escoltar.

**`PARAMETROS`**: diccionario con la configuración por defecto (shape 40×40, 3 harvesters, 2 tractores, etc.) — se copia y sobreescribe (`dict(PARAMETROS)`) en el notebook para cada corrida distinta, nunca se muta directamente.

## Convenciones

- Todo el código (nombres de variables, funciones, docstrings, comentarios) está en español. Mantener esa convención al editar `granja.py`.
- Estados de agentes son strings literales comparados directamente (no Enum) — si se agrega un estado nuevo, actualizar también los diccionarios de color en el notebook (`COLOR_HARVESTER`/`COLOR_TRACTOR`) y cualquier lugar que enumere estados válidos.
- Las celdas objetivo reservadas (`model.reservadas`) deben liberarse (`discard`) en todo camino que abandone o complete un objetivo — al cosechar, al perder gasolina, al no poder trazar ruta — para no dejar celdas fantasma bloqueadas.
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
