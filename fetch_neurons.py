"""
neuPrint Fetcher
-----------------
Pulls the three target neuron populations from the real MaleCNS v1.0
connectome via the neuPrint API, and caches the result locally so we
don't re-query every run.

Setup before running:
    export NEUPRINT_TOKEN="your_token_here"

Usage:
    python fetch_neurons.py
"""

import os
import json
from pathlib import Path

from dotenv import load_dotenv
from neuprint import Client, fetch_neurons, fetch_adjacencies, fetch_all_rois, NeuronCriteria as NC

load_dotenv()  # reads .env in the current working directory automatically

DATASET = "male-cns:v1.0"
SERVER = "https://neuprint.janelia.org"
CACHE_DIR = Path(__file__).parent.parent / "logs" / "neuron_cache"

# Named target populations. Type strings need to match neuPrint's actual
# annotation labels -- these are the commonly cited ones from the fly
# connectome literature, but if a query comes back empty we'll need to
# search the type list and adjust the label.
TARGET_POPULATIONS = {
    "reward": "PAM",        # dopaminergic reward-signaling cluster
    "aversive": "PPL1",     # dopaminergic punishment/aversive cluster
    "sensory": "OL",        # optic lobe (visual input) -- broad match, may need narrowing
}


def get_client() -> Client:
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise RuntimeError(
            "NEUPRINT_TOKEN not set. Run: export NEUPRINT_TOKEN='your_token_here'"
        )
    return Client(SERVER, dataset=DATASET, token=token)


def fetch_population_by_type(client: Client, type_substring: str):
    """
    Fetch neurons whose type matches the given substring, plus their
    basic metadata. Use for named cell types (PAM, PPL1) where the
    type field IS the meaningful label.
    """
    criteria = NC(type=f".*{type_substring}.*", regex=True)
    neurons_df, roi_df = fetch_neurons(criteria, client=client)
    return neurons_df


def fetch_population_by_roi(client: Client, roi_name: str):
    """
    Fetch neurons that have synapses within a given brain region (ROI),
    e.g. optic lobe substructures. Use this for sensory input instead
    of type-substring matching -- visual input neurons aren't reliably
    named with "OL" in their type field, but they DO have synapses
    tagged to the optic lobe ROI.
    """
    criteria = NC(rois=[roi_name])
    neurons_df, roi_df = fetch_neurons(criteria, client=client)
    return neurons_df


def discover_optic_lobe_rois(client: Client):
    """
    Print all ROI names containing common optic-lobe substrings so we
    can confirm the exact tags this dataset uses before querying.
    Male-cns typically tags these per hemisphere, e.g. 'ME(R)', 'LO(R)',
    'LOP(R)' (medulla, lobula, lobula plate) plus left-side equivalents.
    """
    all_rois = fetch_all_rois(client=client)
    candidates = [r for r in all_rois if any(
        tag in r for tag in ["ME(", "LO(", "LOP(", "OL", "AME", "LA("]
    )]
    print("Candidate optic-lobe ROI tags found in this dataset:")
    for r in candidates:
        print(f"  - {r}")
    return candidates


def main():
    client = get_client()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: find the real optic-lobe ROI tags this dataset actually uses
    discover_optic_lobe_rois(client)
    print()

    results = {}

    # Type-based populations (reward/aversive) -- unchanged, these worked
    for label, type_substr in [("reward", "PAM"), ("aversive", "PPL1")]:
        print(f"Fetching population '{label}' (type ~ '{type_substr}')...")
        try:
            df = fetch_population_by_type(client, type_substr)
            count = len(df)
            print(f"  -> found {count} neurons")
            keep_cols = [c for c in ["bodyId", "type", "instance", "status", "size"]
                         if c in df.columns]
            records = df[keep_cols].to_dict(orient="records") if count > 0 else []
            results[label] = {"type_query": type_substr, "count": count, "neurons": records}
        except Exception as e:
            print(f"  ERROR fetching '{label}': {e}")
            results[label] = {"type_query": type_substr, "count": 0, "neurons": [], "error": str(e)}

        out_path = CACHE_DIR / f"{label}.json"
        with open(out_path, "w") as f:
            json.dump(results[label], f, indent=2)
        print(f"  -> cached to {out_path}")

    # ROI-based sensory population -- run discover_optic_lobe_rois() output
    # above and adjust SENSORY_ROIS below to match what actually printed.
    SENSORY_ROIS = ["ME(R)", "ME(L)", "LO(R)", "LO(L)", "LOP(R)", "LOP(L)"]
    print(f"\nFetching population 'sensory' (ROIs = {SENSORY_ROIS})...")
    all_sensory = []
    for roi in SENSORY_ROIS:
        try:
            df = fetch_population_by_roi(client, roi)
            print(f"  {roi}: {len(df)} neurons")
            all_sensory.append(df)
        except Exception as e:
            print(f"  {roi}: ERROR -- {e} (this ROI name likely doesn't exist in this dataset, "
                  f"check the discovery list printed above and fix SENSORY_ROIS)")

    if all_sensory:
        import pandas as pd
        combined = pd.concat(all_sensory).drop_duplicates(subset="bodyId")
        count = len(combined)
        keep_cols = [c for c in ["bodyId", "type", "instance", "status", "size"] if c in combined.columns]
        records = combined[keep_cols].to_dict(orient="records")
    else:
        count = 0
        records = []

    results["sensory"] = {"type_query": f"ROIs: {SENSORY_ROIS}", "count": count, "neurons": records}
    out_path = CACHE_DIR / "sensory.json"
    with open(out_path, "w") as f:
        json.dump(results["sensory"], f, indent=2)
    print(f"  -> total unique sensory neurons: {count}, cached to {out_path}")

    # Summary
    print("\n--- Summary ---")
    for label, data in results.items():
        print(f"{label}: {data['count']} neurons")

    return results


if __name__ == "__main__":
    main()