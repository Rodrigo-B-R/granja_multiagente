"""Q-learning para los agentes de granja_multiagente.

Este paquete es independiente de `granja.py`: define el espacio de estados,
las acciones y las tablas Q de cada tipo de agente, pero no modifica el
modelo. La integracion (que un Harvester/Tractor consulte su tabla Q en vez
de la logica de umbral fija) se conecta despues, sin tocar esta carpeta.

Modulos:
- `utils`: discretizacion de variables continuas en bins, distancia Manhattan.
- `harvester_rl`: estado/acciones/tabla Q para la decision del Harvester de
  seguir cosechando vs desviarse a recargar gasolina.
- `tractor_rl`: estado/acciones/tabla Q para la decision del Tractor de
  esperar, ir a recargar, o atender una llamada de un harvester.
- `q_table`: construccion, guardado y carga de tablas Q genericas
  (diccionario estado -> valores por accion).
"""
