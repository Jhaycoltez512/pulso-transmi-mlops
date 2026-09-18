# Exploración inicial de datos — Pulso TransMi

Generado desde la API el 2026-09-18T14:05:54.436755-05:00.

## Corte analizado

- Dataset: `pulso-transmi-starter-v1`
- Rango declarado: 2026-07-26T00:00:00-05:00 a 2026-09-08T23:45:00-05:00
- Frecuencia declarada: 15 minutos
- Estaciones: 12
- Observaciones descargadas: 51840
- Filas de contexto descargadas: 4320

## Calidad

- Filas duplicadas de observaciones: 0
- Combinaciones estación-fecha esperadas pero ausentes: 0
- Intervalos temporales incompletos dentro de las estaciones: 0
- Valores nulos en observaciones: 0
- Demandas negativas: 0
- Instantes de observación sin contexto: 0
- Instantes de contexto sin observaciones: 0

## Interpretación de outliers

Los outliers se marcan por estación usando la regla IQR (menor que Q1 − 1.5×IQR
o mayor que Q3 + 1.5×IQR). Son candidatos a inspección, no errores confirmados:
en demanda de transporte pueden corresponder a picos reales.

## Archivos

- `tables/`: controles de calidad, correlaciones, perfil temporal y outliers.
- `figures/`: distribución, continuidad, perfil horario y correlaciones.
- `maps/stations_demand_map.html`: mapa interactivo; el tamaño y color del marcador
  representan la demanda media por estación. `figures/stations_geographic_demand_map.png`
  es su versión estática.
