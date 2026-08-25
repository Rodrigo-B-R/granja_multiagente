"""Constantes de recompensa compartidas por Harvester y Tractor.

La recompensa de una decision (seguir vs recargar, esperar vs recargar) no
se calcula en un solo tick: se arma sumando, a lo largo de todos los pasos
que pasan hasta la proxima decision (el agente puede quedar "encerrado" en
`recargando`, `esperando_tractor`, `vertiendo`, etc. varios pasos sin volver
a elegir nada):

    recompensa = eventos_buenos - COSTO_PASO * pasos_transcurridos
                 - (PENALIZACION_VARADO si en el camino quedo sin_gasolina)

`eventos_buenos` acumula R_CELDA_COSECHADA por cada celda que el harvester
cosecha, o R_GRANO_ENTREGADO por unidad de carga que el tractor entrega en
el silo. El costo por paso castiga la demora (fomenta no desviarse sin
necesidad); la penalizacion de varado es deliberadamente grande porque ese
desenlace inmoviliza al agente el resto de la corrida.

COSTO_PASO subio de 0.05 a 0.1 (se probo 0.25 primero: mas alto todavia,
un viaje completo a recargar -40-80 pasos de ida y vuelta a la base- ya
pesaba 10-20 puntos, una fraccion grande de PENALIZACION_VARADO, y el
tractor aprendia a evitar recargar incluso cuando de verdad lo necesitaba
-mas corridas sin terminar, no menos-. Con 0.05 el problema era el opuesto:
un viaje de mas costaba 1-2, casi gratis frente a los 100 de quedar varado,
asi que recargaba por las dudas todo el tiempo. 0.1 deja ese mismo viaje en
4-8: ya pesa en la decision sin acercarse a la escala de la penalizacion de
varado.
"""

R_CELDA_COSECHADA = 1.0
R_GRANO_ENTREGADO = 0.1
COSTO_PASO = 0.1
PENALIZACION_VARADO = 70.0
