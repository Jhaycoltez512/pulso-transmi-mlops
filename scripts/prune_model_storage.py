"""One-off cleanup of the Supabase Storage `models` bucket, with the same rule the pipeline
applies after every retrain (scripts/run_forecast_cycle.py:models_to_prune): keep the active
model, the MODELS_TO_KEEP newest and the OLDEST_MODELS_TO_KEEP oldest bundles.

    python scripts/prune_model_storage.py            # list what would be deleted
    python scripts/prune_model_storage.py --apply    # delete it
"""

from __future__ import annotations

import os
import sys

from load_supabase import SupabaseLoader, delete_objects, list_objects, load_dotenv
from run_forecast_cycle import MODEL_BUCKET, models_to_prune


def main() -> None:
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise SystemExit("Configure SUPABASE_URL and SUPABASE_SECRET_KEY.")
    loader = SupabaseLoader(url, key)
    try:
        active = loader.select("model_versions", {"is_active": "eq.true", "select": "version", "limit": 1})
    finally:
        loader.close()
    if not active:
        raise SystemExit("No active model version: refusing to prune without knowing what to keep.")
    objects = list_objects(url, key, MODEL_BUCKET)
    stale = models_to_prune(objects, {f"{active[0]['version']}.joblib"})
    freed = sum(int((obj.get("metadata") or {}).get("size") or 0) for obj in objects if obj["name"] in stale)
    print(f"{len(objects)} objects, active={active[0]['version']}, deleting {len(stale)} (~{freed / 1e6:.0f} MB), keeping {len(objects) - len(stale)}.")
    kept = sorted(obj["name"] for obj in objects if obj["name"] not in stale)
    print("Keeping:\n  " + "\n  ".join(kept))
    print("Deleting:\n  " + "\n  ".join(sorted(stale)))
    if "--apply" not in sys.argv:
        print("Dry run: pass --apply to delete.")
        return
    for start in range(0, len(stale), 100):
        delete_objects(url, key, MODEL_BUCKET, stale[start:start + 100])
    print(f"Remaining: {len(list_objects(url, key, MODEL_BUCKET))} objects.")


if __name__ == "__main__":
    main()
