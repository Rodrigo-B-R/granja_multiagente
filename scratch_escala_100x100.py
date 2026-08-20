import json
import os
import time

from granja import GranjaModel, PARAMETROS

RUTA_JSON = 'resultados_100x100.json'

tractores_lista = list(range(1, 51, 2))
if 50 not in tractores_lista:
    tractores_lista.append(50)

if os.path.exists(RUTA_JSON):
    with open(RUTA_JSON, encoding='utf-8') as f:
        resultados = json.load(f)
else:
    resultados = []

ya_hechos = {f['n_tractores'] for f in resultados}
pendientes = [t for t in tractores_lista if t not in ya_hechos]
print(f"ya hechos: {sorted(ya_hechos)}")
print(f"pendientes: {pendientes}")

for n_t in pendientes:
    n_h = max(1, round(n_t / 0.75))

    params = dict(PARAMETROS)
    params['shape'] = (100, 100)
    params['n_harvesters'] = n_h
    params['n_tractores'] = n_t
    params['seed'] = 1
    params['steps'] = 45000

    t0 = time.time()
    m = GranjaModel(params)
    r = m.run(display=False)
    dt = time.time() - t0

    fila = {
        'n_harvesters': n_h,
        'n_tractores': n_t,
        'pasos': int(r.reporters['pasos'][0]),
        'completo': bool(r.reporters['cosechado_pct'][0] >= 100),
        'tiempo_real_s': dt,
    }
    resultados.append(fila)
    resultados.sort(key=lambda f: f['n_tractores'])

    ruta_tmp = RUTA_JSON + '.tmp'
    with open(ruta_tmp, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2)
    os.replace(ruta_tmp, RUTA_JSON)  # escritura atomica: si se corta a mitad, el .json previo queda intacto

    print(f"harvesters={n_h:3d} tractores={n_t:3d}  pasos={fila['pasos']:6d}  "
          f"completo={fila['completo']}  tiempo_real={dt:6.1f}s  "
          f"[{len(resultados)}/{len(tractores_lista)}]", flush=True)

print('listo, todos los puntos completados' if len(resultados) == len(tractores_lista) else 'quedan pendientes, volver a correr')
