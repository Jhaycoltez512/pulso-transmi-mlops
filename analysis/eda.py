"""Generate a basic, reproducible EDA from the Pulso TransMi API.

Run with: python analysis/eda.py
Outputs are written to eda_results/.
"""

from __future__ import annotations

from pathlib import Path

import folium
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pulso_transmi import PulsoTransmiClient


OUTPUT = Path("eda_results")
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
MAPS = OUTPUT / "maps"
FREQUENCY = "15min"


def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=160, bbox_inches="tight")
    plt.close()


def write_report(meta: dict, quality: pd.DataFrame, stations: pd.DataFrame) -> None:
    summary = quality.set_index("metric")["value"].to_dict()
    report = f"""# Exploración inicial de datos — Pulso TransMi

Generado desde la API el {pd.Timestamp.now(tz="America/Bogota").isoformat()}.

## Corte analizado

- Dataset: `{meta['dataset']}`
- Rango declarado: {meta['history_start']} a {meta['history_end']}
- Frecuencia declarada: {meta['frequency_minutes']} minutos
- Estaciones: {len(stations)}
- Observaciones descargadas: {summary['observation_rows']}
- Filas de contexto descargadas: {summary['context_rows']}

## Calidad

- Filas duplicadas de observaciones: {summary['duplicate_observation_keys']}
- Combinaciones estación-fecha esperadas pero ausentes: {summary['missing_station_periods']}
- Intervalos temporales incompletos dentro de las estaciones: {summary['station_time_gaps']}
- Valores nulos en observaciones: {summary['observation_null_values']}
- Demandas negativas: {summary['negative_demand_values']}
- Instantes de observación sin contexto: {summary['observation_timestamps_without_context']}
- Instantes de contexto sin observaciones: {summary['context_timestamps_without_observations']}

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
"""
    (OUTPUT / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    for directory in (TABLES, FIGURES, MAPS):
        directory.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")
    with PulsoTransmiClient() as client:
        metadata = client.meta()
        dataset = metadata["dataset"]
        stations = client.stations()
        observations = client.observations_dataframe(page_size=5000)
        context = client.context_dataframe(page_size=5000)

    observations = observations.sort_values(["station_id", "observed_at"]).reset_index(drop=True)
    context = context.sort_values("observed_at").reset_index(drop=True)
    observations.to_csv(TABLES / "observations.csv", index=False)
    context.to_csv(TABLES / "context.csv", index=False)
    stations.to_csv(TABLES / "stations.csv", index=False)

    expected_times = pd.date_range(
        observations["observed_at"].min(), observations["observed_at"].max(), freq=FREQUENCY
    )
    expected_grid = pd.MultiIndex.from_product(
        [stations["station_id"], expected_times], names=["station_id", "observed_at"]
    )
    actual_grid = pd.MultiIndex.from_frame(observations[["station_id", "observed_at"]])
    missing_grid = expected_grid.difference(actual_grid)
    observation_timestamps = pd.Index(observations["observed_at"].unique())
    context_timestamps = pd.Index(context["observed_at"].unique())

    station_quality = (
        observations.groupby("station_id", as_index=False)
        .agg(rows=("demand", "size"), null_demand=("demand", lambda x: x.isna().sum()),
             min_demand=("demand", "min"), max_demand=("demand", "max"),
             mean_demand=("demand", "mean"), median_demand=("demand", "median"))
        .merge(stations[["station_id", "station_name", "corridor"]], on="station_id", how="left")
    )
    station_quality["missing_periods"] = station_quality["station_id"].map(
        pd.Series(missing_grid.get_level_values("station_id")).value_counts()
    ).fillna(0).astype(int)
    station_quality["time_gaps"] = (
        observations.groupby("station_id")["observed_at"].diff().dt.total_seconds().gt(15 * 60)
        .groupby(observations["station_id"]).sum().values
    )

    quality = pd.DataFrame([
        ("observation_rows", len(observations)),
        ("context_rows", len(context)),
        ("station_count", len(stations)),
        ("duplicate_observation_keys", observations.duplicated(["station_id", "observed_at"]).sum()),
        ("duplicate_context_timestamps", context.duplicated("observed_at").sum()),
        ("missing_station_periods", len(missing_grid)),
        ("station_time_gaps", int(station_quality["time_gaps"].sum())),
        ("observation_null_values", int(observations.isna().sum().sum())),
        ("context_null_values", int(context.isna().sum().sum())),
        ("negative_demand_values", int((observations["demand"] < 0).sum())),
        ("observation_timestamps_without_context", len(observation_timestamps.difference(context_timestamps))),
        ("context_timestamps_without_observations", len(context_timestamps.difference(observation_timestamps))),
    ], columns=["metric", "value"])
    quality.to_csv(TABLES / "data_quality_summary.csv", index=False)
    station_quality.to_csv(TABLES / "station_quality.csv", index=False)

    quartiles = observations.groupby("station_id")["demand"].quantile([0.25, 0.75]).unstack()
    quartiles.columns = ["q1", "q3"]
    quartiles["iqr"] = quartiles.q3 - quartiles.q1
    quartiles["lower_bound"] = (quartiles.q1 - 1.5 * quartiles.iqr).clip(lower=0)
    quartiles["upper_bound"] = quartiles.q3 + 1.5 * quartiles.iqr
    outliers = observations.join(quartiles[["lower_bound", "upper_bound"]], on="station_id")
    outliers["is_iqr_outlier"] = (outliers.demand < outliers.lower_bound) | (outliers.demand > outliers.upper_bound)
    outlier_summary = (
        outliers.groupby("station_id", as_index=False)
        .agg(outlier_count=("is_iqr_outlier", "sum"), rows=("demand", "size"),
             lower_bound=("lower_bound", "first"), upper_bound=("upper_bound", "first"))
    )
    outlier_summary["outlier_rate_pct"] = 100 * outlier_summary.outlier_count / outlier_summary.rows
    outlier_summary.merge(stations[["station_id", "station_name"]], on="station_id").to_csv(
        TABLES / "outliers_iqr_by_station.csv", index=False
    )

    joined = observations.merge(context, on="observed_at", how="left")
    numeric_columns = ["demand", "rain_mm", "rain_forecast", "temperature_c", "temperature_forecast", "event_intensity"]
    correlations = joined[numeric_columns].corr(method="pearson")
    correlations.to_csv(TABLES / "pearson_correlations.csv")
    hourly = joined.assign(hour=joined.observed_at.dt.hour, weekday=joined.observed_at.dt.day_name())
    hourly_profile = hourly.groupby("hour", as_index=False).agg(mean_demand=("demand", "mean"), median_demand=("demand", "median"))
    hourly_profile.to_csv(TABLES / "hourly_demand_profile.csv", index=False)

    plt.figure(figsize=(13, 6))
    sns.boxplot(data=observations, x="station_id", y="demand", hue="station_id", legend=False, palette="tab20")
    plt.title("Distribución de demanda por estación (outliers IQR visibles)")
    plt.xlabel("Estación")
    plt.ylabel("Demanda")
    plt.xticks(rotation=45, ha="right")
    save_figure("demand_distribution_by_station.png")

    plt.figure(figsize=(13, 5))
    sns.barplot(data=station_quality, x="station_id", y="missing_periods", color="#d95f02")
    plt.title("Períodos faltantes por estación frente a la grilla de 15 minutos")
    plt.xlabel("Estación")
    plt.ylabel("Períodos faltantes")
    plt.xticks(rotation=45, ha="right")
    save_figure("missing_periods_by_station.png")

    plt.figure(figsize=(9, 7))
    sns.heatmap(correlations, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="vlag", square=True)
    plt.title("Correlaciones de Pearson: demanda y contexto")
    save_figure("demand_context_correlations.png")

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=hourly_profile, x="hour", y="mean_demand", marker="o", label="Media")
    sns.lineplot(data=hourly_profile, x="hour", y="median_demand", marker="o", label="Mediana")
    plt.xticks(range(24))
    plt.title("Perfil horario agregado de demanda")
    plt.xlabel("Hora del día (America/Bogota)")
    plt.ylabel("Demanda")
    save_figure("hourly_demand_profile.png")

    mapped = stations.merge(station_quality[["station_id", "mean_demand", "max_demand", "missing_periods"]], on="station_id")
    center = [mapped.latitude.mean(), mapped.longitude.mean()]
    demand_map = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap")
    for row in mapped.itertuples():
        color = "#d73027" if row.mean_demand >= mapped.mean_demand.median() else "#4575b4"
        popup = (
            f"<b>{row.station_name}</b><br>ID: {row.station_id}<br>Corredor: {row.corridor}"
            f"<br>Demanda media: {row.mean_demand:.1f}<br>Demanda máxima: {row.max_demand:.0f}"
            f"<br>Períodos faltantes: {row.missing_periods}"
        )
        folium.CircleMarker(
            location=[row.latitude, row.longitude], radius=max(6, row.mean_demand / 12),
            color=color, fill=True, fill_opacity=0.75, popup=popup, tooltip=row.station_name,
        ).add_to(demand_map)
    demand_map.save(MAPS / "stations_demand_map.html")

    plt.figure(figsize=(9, 8))
    scatter = plt.scatter(
        mapped.longitude, mapped.latitude, s=mapped.mean_demand * 1.2,
        c=mapped.mean_demand, cmap="YlOrRd", alpha=0.8, edgecolors="black", linewidths=0.7,
    )
    for row in mapped.itertuples():
        plt.annotate(row.station_id, (row.longitude, row.latitude), xytext=(4, 4), textcoords="offset points", fontsize=8)
    plt.colorbar(scatter, label="Demanda media")
    plt.title("Ubicación geográfica y demanda media por estación")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.grid(alpha=0.25)
    save_figure("stations_geographic_demand_map.png")

    write_report(dataset, quality, stations)
    print(f"EDA complete. Results written to {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
