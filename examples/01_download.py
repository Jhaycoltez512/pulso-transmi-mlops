from pathlib import Path

from pulso_transmi import PulsoTransmiClient


def main() -> None:
    output = Path("data")
    with PulsoTransmiClient() as client:
        metadata = client.meta()["dataset"]
        print(
            f"Dataset {metadata['dataset']}: {metadata['station_count']} estaciones, "
            f"{metadata['observation_rows']:,} observaciones"
        )
        for filename in ("stations.csv", "observations.csv", "context.csv", "metadata.json"):
            path = client.download(filename, output / filename)
            print(f"descargado: {path}")


if __name__ == "__main__":
    main()
