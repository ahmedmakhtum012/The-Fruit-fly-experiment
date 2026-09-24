"""
End-to-End Pipeline Test
--------------------------
Full chain: text stimulus -> connectome sim -> decision table -> 
reply generation -> logged output. Tests that everything wires
together before going live.

Run from project root:
    python pipeline/e2e_test.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add sim and pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "sim"))
sys.path.insert(0, str(Path(__file__).parent))

from connectome_real import ConnectomeSim
from decision_table import DecisionTable
from reply_generator import ReplyGenerator

LOGS_DIR = Path(__file__).parent.parent / "logs"


def log_event(stimulus_text, signals, decision, reply=None):
    """Log one event to a JSON file for later review/documentation."""
    event = {
        "timestamp": datetime.now().isoformat(),
        "stimulus": stimulus_text[:100],  # first 100 chars
        "signals": {
            "reward": round(signals["reward"], 4),
            "aversive": round(signals["aversive"], 4),
            "ambiguity": round(signals["ambiguity"], 4),
        },
        "decision": {
            "action": decision.action.name,
            "confidence": round(decision.confidence, 2),
            "reasoning": decision.reasoning,
        },
        "reply": reply,
    }
    return event


def main():
    print("Initializing pipeline...")
    sim = ConnectomeSim()
    dt = DecisionTable()
    gen = ReplyGenerator()

    test_stimuli = [
        "just adopted the cutest kitten ever look at this fluff",
        "google mapped the entire fruit fly connectome open source now people are training it",
        "my flight got cancelled and I'm stuck at the airport for 8 hours",
        "this new AI model is insane it can do literally everything",
        "doom gameplay with a fruit fly brain controlling the character",
    ]

    events = []

    print(f"\nRunning {len(test_stimuli)} test stimuli through the full pipeline...\n")

    for i, text in enumerate(test_stimuli, 1):
        # Stimulus -> sim
        signals = sim.run(text=text)

        # Signals -> decision
        decision = dt.decide(
            reward=signals["reward"],
            aversive=signals["aversive"],
            ambiguity=signals["ambiguity"],
        )

        # If REPLY action, generate text
        reply = None
        if decision.action.name == "REPLY":
            reply = gen.generate(signals["reward"], signals["aversive"])

        # Log the event
        event = log_event(text, signals, decision, reply)
        events.append(event)

        # Print to console
        print(f"[{i}] {text[:50]!r}")
        print(
            f"    Signals: R={signals['reward']:.3f} A={signals['aversive']:.3f} U={signals['ambiguity']:.3f}"
        )
        print(f"    Decision: {decision.action.name} (confidence: {decision.confidence:.2f})")
        if reply:
            print(f"    Reply: '{reply}'")
        print(f"    Reasoning: {decision.reasoning}")
        print()

    # Write full log to file
    log_path = LOGS_DIR / "e2e_test_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(events, f, indent=2)

    print(f"\n[OK] Pipeline test complete. Full log saved to {log_path}")
    print(f"\nSummary:")
    print(f"  Total stimuli: {len(events)}")
    action_counts = {}
    for e in events:
        action = e["decision"]["action"]
        action_counts[action] = action_counts.get(action, 0) + 1
    for action, count in sorted(action_counts.items()):
        print(f"  {action}: {count}")


if __name__ == "__main__":
    main()