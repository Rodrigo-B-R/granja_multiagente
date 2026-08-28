"""Constantes de recompensa compartidas por Harvester y Tractor.

La recompensa de una decision (seguir vs recargar, esperar vs recargar) no
se calcula en un solo tick: se arma sumando, a lo largo de todos los pasos
que pasan hasta la proxima decision (el agente puede quedar "encerrado" en
`recargando`, `esperando_tractor`, `vertiendo`, etc. varios pasos sin volver
a elegir nada):

    recompensa = eventos_buenos - COSTO_PASO * pasos_transcurridos
                 - (PENALIZACION_VARADO_* si en el camino quedo sin_gasolina)

`eventos_buenos` acumula R_CELDA_COSECHADA por cada celda que el harvester
cosecha, R_GRANO_ENTREGADO por unidad de carga que el tractor entrega en
el silo, o R_TANQUE_LLENADO cuando el harvester completa una recarga (ver
mas abajo). El costo por paso castiga la demora (fomenta no desviarse sin
necesidad); la penalizacion de varado es deliberadamente grande porque ese
desenlace inmoviliza al agente el resto de la corrida.

COSTO_PASO subio de 0.05 a 0.1 (se probo 0.25 primero: mas alto todavia,
un viaje completo a recargar -40-80 pasos de ida y vuelta a la base- ya
pesaba 10-20 puntos, una fraccion grande de la penalizacion de varado, y el
tractor aprendia a evitar recargar incluso cuando de verdad lo necesitaba
-mas corridas sin terminar, no menos-. Con 0.05 el problema era el opuesto:
un viaje de mas costaba 1-2, casi gratis frente a los 100 de quedar varado,
asi que recargaba por las dudas todo el tiempo. 0.1 deja ese mismo viaje en
4-8: ya pesa en la decision sin acercarse a la escala de la penalizacion de
varado.

PENALIZACION_VARADO se separo por tipo de agente porque, con tanques
generosos y mapas chicos como los del entrenamiento del tractor, quedarse
sin gasolina es un evento raro durante el entrenamiento -- el castigo,
por grande que sea, casi nunca llega a observarse, y la tabla Q termina
reflejando solo el costo cierto del viaje de ida y vuelta, nunca el riesgo
que ese viaje evita. R_TANQUE_LLENADO compensa eso directamente: es un
bono que se suma cuando el harvester efectivamente completa la recarga
(vuelve a la base y llena el tanque), asi el viaje no queda como un costo
hundido puro -- solo el varado en si (PENALIZACION_VARADO_HARVESTER) y el
COSTO_PASO_RECARGA acumulado siguen desalentando recargar quedandose sin
necesitarlo todavia.

COSTO_PASO_RECARGA es un costo por paso mas bajo que COSTO_PASO, y se
aplica solo mientras la excursion en curso es la de ir a recargar (no a
la de seguir cosechando/escoltando). Se separo porque, con un unico
COSTO_PASO compartido, se observo en la tabla entrenada que
IR_A_RECARGAR practicamente nunca superaba a la alternativa incluso en
estados de gasolina critica cerca de la base (2/24 en las pruebas que
motivaron esto) -- el costo del viaje competia en la misma escala que el
costo de simplemente seguir operando, y encima el varado real (la unica
señal que de verdad justifica el viaje) es un evento raro y con
propagacion lenta via bootstrap, asi que el costo del viaje pesaba mas en
la practica que el riesgo que evita. Bajarlo especificamente para la
excursion de recarga (sin tocar el costo de seguir cosechando) inclina la
balanza hacia recargar sin inflar R_TANQUE_LLENADO ni debilitar el costo
por demora del resto de las decisiones.
"""

R_CELDA_COSECHADA = 1.0
R_GRANO_ENTREGADO = 0.1
R_TANQUE_LLENADO = 15.0
COSTO_PASO = 0.1
COSTO_PASO_RECARGA = 0.06
PENALIZACION_VARADO_TRACTOR = 70.0
PENALIZACION_VARADO_HARVESTER = 150.0
