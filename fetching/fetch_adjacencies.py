"""
Adjacency Fetcher
------------------
Pulls the real synaptic weight graph between our three cached
populations (sensory -> reward, sensory -> aversive, and any direct
reward <-> aversive connections) from neuPrint.

Requires neuron_cache/{sensory,reward,aversive}.json to already exist
(run fetch_neurons.py first).

Usage:
    python fetch_adjacencies.py
"""

import os
import json
from pathlib import Path

from dotenv import load_dotenv
from neuprint import Client, fetch_adjacencies, NeuronCriteria as NC  # type: ignore

load_dotenv()

DATASET = "male-cns:v1.0"
SERVER = "https://neuprint.janelia.org"
CACHE_DIR = Path(__file__).parent.parent / "logs" / "neuron_cache"


def get_client() -> Client:
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise RuntimeError("NEUPRINT_TOKEN not set.")
    return Client(SERVER, dataset=DATASET, token=token)


def load_population(label: str) -> list[int]:
    path = CACHE_DIR / f"{label}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run fetch_neurons.py first.")
    with open(path) as f:
        data = json.load(f)
    return [n["bodyId"] for n in data["neurons"]]


def fetch_edges(client: Client, source_ids: list, target_ids: list, batch_size=5000):
    """
    Fetch weighted synaptic connections from source_ids -> target_ids.
    Batches the source list since neuPrint queries can choke on huge
    ID lists in one call (102k sensory neurons is too many for one shot).
    """
    all_edges = []
    for i in range(0, len(source_ids), batch_size):
        batch = source_ids[i:i + batch_size]
        print(f"  querying sources {i}-{i+len(batch)} of {len(source_ids)}...")
        try:
            src_crit = NC(bodyId=batch)
            tgt_crit = NC(bodyId=target_ids)
            neuron_df, conn_df = fetch_adjacencies(src_crit, tgt_crit, client=client)
            if len(conn_df) > 0:
                all_edges.append(conn_df)
        except Exception as e:
            print(f"  WARNING: batch {i} failed -- {e}")
    if all_edges:
        import pandas as pd
        return pd.concat(all_edges, ignore_index=True)
    else:
        import pandas as pd
        return pd.DataFrame(columns=["bodyId_pre", "bodyId_post", "weight"])


def main():
    client = get_client()

    print("Loading cached populations...")
    sensory_ids = load_population("sensory")
    reward_ids = load_population("reward")
    aversive_ids = load_population("aversive")
    print(f"  sensory: {len(sensory_ids)}, reward: {len(reward_ids)}, aversive: {len(aversive_ids)}")

    edges = {}

    print("\nFetching sensory -> reward edges...")
    edges["sensory_to_reward"] = fetch_edges(client, sensory_ids, reward_ids)
    print(f"  -> {len(edges['sensory_to_reward'])} edges found")

    print("\nFetching sensory -> aversive edges...")
    edges["sensory_to_aversive"] = fetch_edges(client, sensory_ids, aversive_ids)
    print(f"  -> {len(edges['sensory_to_aversive'])} edges found")

    print("\nFetching reward <-> aversive edges...")
    edges["reward_to_aversive"] = fetch_edges(client, reward_ids, aversive_ids)
    edges["aversive_to_reward"] = fetch_edges(client, aversive_ids, reward_ids)
    print(f"  -> {len(edges['reward_to_aversive'])} + {len(edges['aversive_to_reward'])} edges found")

    # Cache each edge set
    for name, df in edges.items():
        out_path = CACHE_DIR / f"edges_{name}.json"
        df.to_json(out_path, orient="records", indent=2)
        print(f"  cached {name} -> {out_path}")

    print("\n--- Summary ---")
    for name, df in edges.items():
        print(f"{name}: {len(df)} edges")

    if len(edges["sensory_to_reward"]) == 0 and len(edges["sensory_to_aversive"]) == 0:
        print("\nWARNING: zero direct edges found from sensory to reward/aversive.")
        print("This is actually expected biologically -- sensory neurons rarely")
        print("connect directly to deep dopaminergic clusters, there are usually")
        print("multiple synaptic hops between them. If both are 0, we'll need a")
        print("multi-hop traversal (sensory -> intermediate -> reward/aversive)")
        print("instead of a direct one-hop adjacency. Report back the counts.")

    return edges


if __name__ == "__main__":
    main()