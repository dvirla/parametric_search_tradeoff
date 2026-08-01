"""Build multi-round mocked-search history pools from the 5 single-round templates in
data/frames_cues/search_multi_turn.json, for the round-count ablation (does search
suppression/increase from a mocked-search history change with more prior rounds?).

Each output pool is a list of conversations; each conversation is N single-round
templates (user -> assistant tool_call -> tool tool_response -> assistant text)
concatenated back-to-back, one per ALL C(5,N) combinations of DISTINCT base templates
(never repeats the same topic twice in one history).

Usage:
  uv run python scripts/build_multiround_search_history.py
"""
import itertools
import json

SRC = "data/frames_cues/search_multi_turn.json"
OUT = {
    2: "data/frames_cues/search_multi_turn_2round.json",
    3: "data/frames_cues/search_multi_turn_3round.json",
}


def main():
    templates = json.load(open(SRC))
    for n_rounds, out_path in OUT.items():
        pool = []
        for combo in itertools.combinations(range(len(templates)), n_rounds):
            conv = []
            for idx in combo:
                conv.extend(templates[idx])
            pool.append(conv)
        with open(out_path, "w") as f:
            json.dump(pool, f, indent=2)
        print(f"Wrote {len(pool)} {n_rounds}-round conversations to {out_path}")


if __name__ == "__main__":
    main()
