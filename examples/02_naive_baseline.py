from datetime import timedelta

import numpy as np
import pandas as pd


def accuracy_by_station(frame: pd.DataFrame) -> pd.Series:
    error = (frame["demand"] - frame["prediction"]).abs()
    return 100 * (1 - error.groupby(frame["station_id"]).sum() / frame["demand"].groupby(frame["station_id"]).sum()).clip(lower=0)


def main() -> None:
    observations = pd.read_csv(
        "data/observations.csv",
        dtype={"station_id": "string"},
        parse_dates=["observed_at"],
    ).sort_values(["station_id", "observed_at"])

    observations["prediction"] = observations.groupby("station_id")["demand"].shift(96)
    cutoff = observations["observed_at"].max() - timedelta(days=7)
    validation = observations.loc[observations["observed_at"] > cutoff].dropna().copy()
    scores = accuracy_by_station(validation)

    print(scores.sort_values().round(2))
    print(f"Accuracy promedio: {np.mean(scores):.2f}")


if __name__ == "__main__":
    main()
