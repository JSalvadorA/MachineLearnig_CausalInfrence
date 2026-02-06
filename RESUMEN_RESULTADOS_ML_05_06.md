# Resumen de resultados (LightGBM + MiniBatchKMeans)

Actualizado: 2026-02-03

Alcance: resultados reales de algoritmos 05 (LightGBM) y 06 (MiniBatchKMeans).
Fuente: JSON y Parquet en `new/resultados/`.

## Dataset A_unidades

### Prediccion (LightGBM)
- Muestra usada (JSON): 3.170.789 filas (train 2.536.631 / test 634.158)
- Metricas: RMSE 95.0761, MAE 1.9613, R2 0.2414
- Top features (importancia): horas_dia_prom (4328.00), distrito (3363.00), registros_totales (3079.00), consumo_verano (2719.00), consumo_invierno (1901.00)
- Analisis: R2 moderado: hay senal, falta enriquecer features

### Segmentacion (MiniBatchKMeans)
- Muestra usada (JSON): 2.867.405 filas | modo: streaming | batch_size: 10.000
- Parquet asignaciones: asignaciones_20260203_002035.parquet (2.867.405 filas)
- Mejor k: 3 (silhouette 0.7941)
- Tamano de clusters: 0: 2.214.404 (77.2%), 1: 371.362 (13.0%), 2: 281.639 (9.8%)
- Analisis: separacion alta entre clusters

## Dataset B_conexiones

### Prediccion (LightGBM)
- Muestra usada (JSON): 1.784.555 filas (train 1.427.644 / test 356.911)
- Metricas: RMSE 133.0356, MAE 12.2150, R2 0.0867
- Top features (importancia): horas_dia_prom (5093.00), registros_totales (4962.00), distrito (3028.00), score_calidad (2972.00), categoria_principal (1170.00)
- Analisis: R2 bajo: senal debil con features actuales

### Segmentacion (MiniBatchKMeans)
- Muestra usada (JSON): 1.625.223 filas | modo: streaming | batch_size: 10.000
- Parquet asignaciones: asignaciones_20260203_001548.parquet (1.625.223 filas)
- Mejor k: 6 (silhouette 0.5970)
- Tamano de clusters: 0: 701.106 (43.1%), 1: 519.193 (31.9%), 2: 155.937 (9.6%), 5: 140.876 (8.7%), 3: 98.385 (6.1%), 4: 9.726 (0.6%)
- Analisis: separacion moderada

## Dataset C_distritos

### Prediccion (LightGBM)
- Muestra usada (JSON): 1.847 filas (train 1.477 / test 370)
- Metricas: RMSE 0.7133, MAE 0.4691, R2 0.9614
- Top features (importancia): mes_absoluto (4146.00), pct_domestico (2892.00), pct_comercial (2650.00), num_unidades (2395.00), pct_subsidiado (2314.00)
- Analisis: R2 alto: buen poder predictivo

### Segmentacion (MiniBatchKMeans)
- Muestra usada (JSON): 1.845 filas | modo: streaming | batch_size: 10.000
- Parquet asignaciones: asignaciones_20260203_001620.parquet (1.845 filas)
- Mejor k: 3 (silhouette 0.3503)
- Tamano de clusters: 1: 1.103 (59.8%), 2: 469 (25.4%), 0: 273 (14.8%)
- Analisis: separacion baja a media
