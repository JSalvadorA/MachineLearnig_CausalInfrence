# Explicacion simple de los algoritmos ML (A, B, C)

Este documento explica, en lenguaje no tecnico, que hace cada algoritmo y como usa los datasets A, B y C.

## 1) Que es cada dataset

- Dataset A (A_unidades): una fila por unidad de uso (codudu, codcon). Resume todo el historial 2021-2023 en promedios y totales.
a) ¿Por qué en Dataset A se elimina nuanio y nummes?
Porque el script agrega a nivel de unidad de uso (GROUP BY codudu, codcon).
Al agrupar así, cada unidad queda en una sola fila con promedios y totales del período 2021–2023, por lo que nuanio y nummes pierden sentido en esa tabla y no se incluyen.

Si se quieren conservar, la agregación debe ser mensual, por ejemplo:

GROUP BY codudu, codcon, nuanio, nummes

- Dataset B (B_conexiones): una fila por conexion/predio (codcon). Resume todo el historial por predio.
- Dataset C (C_distritos): una fila por distrito y mes. Sirve para analisis temporal.

Importante: A y B son perfiles historicos agregados (no tienen una fila por mes).

## 2) Regresion (XGBoost)

Objetivo: predecir el consumo_promedio de una unidad o conexion usando otras variables.

En simple: el modelo aprende patrones como "unidades con X caracteristicas suelen consumir Y".
el modelo aprende relaciones típicas entre las variables de entrada y el consumo promedio.
Aquí tienes ejemplos hipotéticos (no son resultados reales, solo ilustrativos):

Ejemplo 1 (categoría y distrito):
“Unidades en el distrito X y categoría Comercial suelen tener consumos promedio más altos que las de categoría Doméstica”.

Ejemplo 2 (estacionalidad):
“Si consumo_verano es alto y consumo_invierno es bajo, el consumo promedio anual tiende a ser medio‑alto”.

Ejemplo 3 (calidad de servicio):
“Cuando score_calidad es alto (más horas/día y días/semana), el consumo promedio tiende a subir, porque hay más disponibilidad del servicio”.

Ejemplo 4 (subsidio):
“Unidades con situdu=1 (subsidiado) suelen tener consumos promedio más bajos que las no subsidiadas”.

En resumen: el modelo detecta combinaciones de características que típicamente van asociadas con consumos más altos o más bajos, y las usa para predecir consumo_promedio.

Salida:
- RMSE y MAE: que tan lejos estan las predicciones en promedio.
- R2: si el modelo aporta mas que usar un promedio simple.
- Importancia de variables: que variables pesan mas en la decision.

Uso tipico: estimar consumo esperado segun perfil del usuario.

## 3) Clustering (K-Means)

Objetivo: agrupar unidades similares segun su comportamiento.

En simple: el algoritmo crea grupos de usuarios parecidos entre si (segmentos).

Salida:
- k optimo (numero de grupos) segun silhouette.
- Perfil de cada cluster (promedios de cada variable).

Uso tipico: segmentar usuarios para analisis o politicas diferenciadas.

## 4) Anomalias (Isolation Forest)

Objetivo: detectar usuarios muy raros o extremos.

En simple: el algoritmo busca registros que estan lejos del comportamiento normal.

Salida:
- porcentaje de anomalias (definido por contamination).
- top de anomalias con sus valores.
- comparacion normal vs anomalo.

Uso tipico: sospechas de consumo anormal, fugas, errores de medicion.

## 5) Series temporales (solo Dataset C)

Objetivo: ver tendencias y estacionalidad por distrito y mes.

En simple: analiza si el consumo sube o baja con el tiempo y si hay patrones por estacion.

Salida:
- tendencia general
- estacionalidad
- indicadores de estabilidad de la serie

Uso tipico: planeamiento y proyecciones mensuales.

## Nota sobre muestreo

Los scripts de modelos usan limites por defecto (500k o 100k). Esto es una muestra y no representa todo el universo.
Para un informe serio, se recomienda muestreo representativo.

Esto es “quién crea qué” en tu stack `new/`, según `CLAUDE.md` (raíz), `new/scripts/CLAUDE.md`, `new/sql/CLAUDE.md` y `new/docker-compose.yml`.

**1) Las tablas NO las crea `02_load_to_postgres.py` (solo las llena)**
- `new/scripts/02_load_to_postgres.py` hace DML:
  - valida que existan `schema dwh` y `dwh.fact_consumo` (`check_database_ready()`)
  - hace `TRUNCATE` de `dwh.staging_consumo` y `dwh.fact_consumo`
  - carga datos (COPY a staging, luego INSERT a fact)
  - registra el proceso en `dwh.load_control` (inserta `in_progress` y al final actualiza a `completed`)
- Importante: **no hay ningún `CREATE TABLE`** en `new/scripts/02_load_to_postgres.py`. Si la tabla existe, fue creada antes.

**2) ¿Entonces de dónde salen las tablas del schema `dwh`?**
Salen del arranque inicial de Postgres por Docker:

- En `new/docker-compose.yml` estás montando:
  - `new/sql/01_init.sql` → `/docker-entrypoint-initdb.d/01_init.sql`
  - `new/sql/02_tables.sql` → `/docker-entrypoint-initdb.d/02_tables.sql`

Cuando el contenedor arranca **por primera vez** con el volumen vacío (`dwh_sunass_new_data`), el entrypoint de Postgres ejecuta esos `.sql`. Ahí se crean:

- `dwh.fact_consumo` (tabla principal) → en `new/sql/02_tables.sql`
- `dwh.staging_consumo` (UNLOGGED, misma estructura) → en `new/sql/02_tables.sql`
- `dwh.load_control` (tabla de control de cargas) → en `new/sql/02_tables.sql`
- además `CREATE SCHEMA dwh`, extensión `pg_stat_statements`, y `search_path` → en `new/sql/01_init.sql`

Por eso ves esas 3 tablas “extra”: **las creó el DDL de inicialización**, no el script de carga.

**3) ¿Y `dwh.fact_consumo_gold`?**
Esa tabla **no está** definida en `new/sql/01_init.sql` ni en `new/sql/02_tables.sql` (o sea: no la crea Docker en el init).

En el repo:
- `new/CLAUDE.md` la menciona como **“Propuesta `fact_consumo_gold`”** (capa GOLD para features).
- `new/cc/tips.md` tiene SQL para **crear índices** y hacer queries **asumiendo que esa tabla ya existe**.

Conclusión: `dwh.fact_consumo_gold` tuvo que ser creada **manualmente en la BD (Navicat/psql)** o por algún proceso fuera de esos init scripts (no está versionado aquí el `CREATE TABLE ... fact_consumo_gold`).

**4) ¿Qué genera `new/data/consumo_2021_2023.parquet`?**
Ese archivo lo genera **solo**:
- `new/scripts/01_convert_to_parquet.py`

Qué hace:
- lee los CSV definidos en `new/.env` (`CSV_FILES` en `DATA_SOURCE`)
- elimina columnas PII (`PII_COLUMNS`)
- concatena y reordena columnas al orden `EXPECTED_COLUMNS` (ver `new/scripts/config.py`)
- escribe el Parquet consolidado en `new/data/consumo_2021_2023.parquet`

O sea: el Parquet es el **staging intermedio en disco** (CSV → Parquet), y luego `new/scripts/02_load_to_postgres.py` lo usa para cargar a Postgres.

**5) Tu screenshot cuadra con esto**
En `dwh.load_control` se ven varias ejecuciones:
- una carga de 10,000 (`completed`)
- otra de 100,000 (`completed`)
- una que quedó `in_progress` (típico de un colapso/kill antes de actualizar a completed)
- y una carga completa con `109161469` filas (`completed`)

Eso lo escribe/actualiza `new/scripts/02_load_to_postgres.py`, pero la tabla `load_control` como tal la creó `new/sql/02_tables.sql`.

**Nota clave sobre el colapso**
En el código actual de `new/scripts/02_load_to_postgres.py` se usa `pl.read_parquet(...)`, que **carga todo el Parquet en RAM** antes de aplicar chunks; con 109M filas eso puede tumbar una máquina de 16 GB. En `new/scripts/CLAUDE.md` ya está documentada la solución (usar `scan_parquet()`/lazy), pero el script (tal como está) no evita ese pico.