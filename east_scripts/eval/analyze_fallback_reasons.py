"""
Counts why the LLM chunkers fell back to the word-count heuristic.

Input: the chunks-*.json files for each chunker.
Output: printed fallback rate and reason breakdown, per latency level.
Older files without per-latency reasons fall back to a single-reason view.
"""
import json
from collections import Counter

CHUNKERS = ["llama", "gemma", "nilechat"]
LATENCIES = ["low", "medium", "high"]

for chunker in CHUNKERS:
    with open(f"data/mt_data/train_data/chunks-{chunker}.json", "r", encoding="utf-8") as f:
        entries = json.load(f)

    print(f"{chunker}: {len(entries)} entries")

    if entries and "fallback_reason_by_latency" in entries[0]:
        for latency in LATENCIES:
            fallback_reasons = Counter()
            n_fallback = 0
            for e in entries:
                reason = e["fallback_reason_by_latency"].get(latency)
                if reason is not None:
                    n_fallback += 1
                    fallback_reasons[reason] += 1
            print(f"  {latency}: {n_fallback}/{len(entries)} fallback ({n_fallback / len(entries):.1%})")
            for reason, count in fallback_reasons.most_common():
                print(f"    {reason}: {count} ({count / n_fallback:.1%} of fallbacks)")
    else:
        fallback_entries = [e for e in entries if e["fallback"]]
        reasons = Counter(e.get("fallback_reason", "unknown") for e in fallback_entries)
        print(f"  (legacy single-reason schema, pre-dates per-latency validation) "
              f"{len(fallback_entries)}/{len(entries)} fallback ({len(fallback_entries) / len(entries):.1%})")
        for reason, count in reasons.most_common():
            print(f"    {reason}: {count} ({count / len(fallback_entries):.1%} of fallbacks)")
    print()
