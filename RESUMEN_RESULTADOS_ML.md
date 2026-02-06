# Resumen claro de resultados (presentables)

Actualizado: 2026-02-03

Fuente: JSON y Parquet en `new/resultados/`. Se reporta el tamano de muestra usado por cada algoritmo y los hallazgos clave.

## Dataset A_unidades

### Prediccion (Regresion XGBoost)
- Muestra usada (JSON): 487.011 filas (train 389.608 / test 97.403)
- Metricas: RMSE 99.4816, MAE 2.1987, R2 -0.1922
- Top features: consumo_verano (0.629), distrito (0.100), categoria (0.094), consumo_invierno (0.087), registros_totales (0.068)

### Segmentacion (K-Means)
- Muestra usada (JSON): 88.217 filas
- Parquet asignaciones: asignaciones_20260202_120328.parquet (88.217 filas)
- Mejor k: 2 (silhouette 0.9891)
- Tamanos de cluster: 0: 88.210, 1: 7

### Anomalias (Isolation Forest)
- Muestra usada (JSON): 440.441 filas
- Parquet anomalias: anomalias_20260202_120912.parquet (4.405 filas)
- Anomalias: 4.405 (1.00%)
- Comparacion (muestra): consumo_promedio: normal 14.46 vs anom 212.66, consumo_std: normal 5.28 vs anom 127.65, tarifa_efectiva: normal 2.17 vs anom 4.76

## Dataset B_conexiones

### Prediccion (Regresion XGBoost)
- Muestra usada (JSON): 1.784.555 filas (train 1.427.644 / test 356.911)
- Metricas: RMSE 132.5615, MAE 12.1046, R2 0.0932
- Top features: horas_dia_prom (0.325), tipo_predio (0.209), categoria_principal (0.204), distrito (0.185), registros_totales (0.056)

### Segmentacion (K-Means)
- Muestra usada (JSON): 91.033 filas
- Parquet asignaciones: asignaciones_20260202_145445.parquet (91.033 filas)
- Mejor k: 8 (silhouette 0.6129)
- Tamanos de cluster: 0: 11.810, 1: 39.903, 2: 9.054, 3: 4, 4: 30.194, 5: 2, 6: 30, 7: 36

### Anomalias (Isolation Forest)
- Muestra usada (JSON): 1.624.051 filas
- Parquet anomalias: anomalias_20260202_145643.parquet (16.241 filas)
- Anomalias: 16.241 (1.00%)
- Comparacion (muestra): consumo_promedio: normal 14.71 vs anom 347.27, consumo_std: normal 6.48 vs anom 175.09, tarifa_efectiva: normal 2.14 vs anom 5.25

## Dataset C_distritos

### Prediccion (Regresion XGBoost)
- Muestra usada (JSON): 1.847 filas (train 1.477 / test 370)
- Metricas: RMSE 0.8687, MAE 0.5491, R2 0.9427
- Top features: nomdis (0.251), pct_comercial (0.194), score_calidad (0.169), num_unidades (0.129), pct_subsidiado (0.111)

### Segmentacion (K-Means)
- Muestra usada (JSON): 1.845 filas
- Parquet asignaciones: asignaciones_20260202_132317.parquet (1.845 filas)
- Mejor k: 3 (silhouette 0.3503)
- Tamanos de cluster: 0: 390, 1: 1.025, 2: 430

### Anomalias (Isolation Forest)
- Muestra usada (JSON): 1.845 filas
- Parquet anomalias: anomalias_20260202_132357.parquet (19 filas)
- Anomalias: 19 (1.03%)
- Comparacion (muestra): consumo_promedio: normal 14.92 vs anom 23.10, demanda_total_m3: normal 871879.72 vs anom 250483.86, tarifa_promedio: normal 4.15 vs anom 3.56

### Estacionalidad (Series temporales)
- Serie total: 36 puntos (meses)
- Media total: 44355876.77 | Amplitud estacional: 6336738.84
- Tendencia 2021-2023: 3.29%
- Pico estacional: mes 2 | Valle: mes 8
