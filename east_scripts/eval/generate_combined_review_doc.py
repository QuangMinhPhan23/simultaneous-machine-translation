"""
Builds one HTML review sheet for human reviewers, covering both chunk
boundaries and translation quality.

Input: chunks-*.json files plus prediction.json files under a results root.
Output: an HTML file with Y/N checklists next to each example.
Translation examples come from five buckets: best/worst COMET, repetition
loops, code-switching, and random. Only runs where the real data lives (Katana).
"""
import argparse
import html
import json
import random
import re

LATIN_RUN = re.compile(r"[A-Za-z]{2,}")


def esc(text):
    """Escape text before it goes into the HTML, so source text cannot break the page."""
    return html.escape(text, quote=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_chunks(path):
    """Load one chunker's output, keyed by (conversation id, turn index).

    That key is what lets the same turn be looked up across all three chunkers."""
    with open(path, "r", encoding="utf-8") as f:
        return {(e["conv_id"], e["turn_idx"]): e for e in json.load(f)}


def word_count(text):
    return len(text.split())


def entry_fallback(entry, latency):
    """True if this chunker had to use the word-count backup at this latency."""
    return entry.get("fallback_by_latency", {}).get(latency, False)


def entry_chunks(entry, latency):
    return entry["chunks"][latency]


def has_repetition_loop(text, min_repeats=4):
    """True if the same word repeats several times in a row, a common generation failure."""
    words = text.split()
    run_word, run_len = None, 0
    for w in words:
        if w == run_word:
            run_len += 1
            if run_len >= min_repeats:
                return True
        else:
            run_word, run_len = w, 1
    return False


def has_code_switch(text):
    """True if the text contains a run of Latin letters, i.e. English inside the Arabic."""
    return bool(LATIN_RUN.search(text))


# ---------- chunk-boundary section ----------

def pick_chunk_examples(n_success, n_fallback, seed, min_words, max_words, latency):
    """Pick turns to show in the chunk-boundary section, at one latency level.

    Only turns that all three chunkers handled are usable, since the point is to compare them
    side by side. Two samples are drawn: turns where every chunker succeeded, and turns where at
    least one fell back, so a reviewer sees both."""
    llama = load_chunks("data/mt_data/train_data/chunks-llama.json")
    gemma = load_chunks("data/mt_data/train_data/chunks-gemma.json")
    nilechat = load_chunks("data/mt_data/train_data/chunks-nilechat.json")
    common_keys = set(llama) & set(gemma) & set(nilechat)
    common_keys = {
        k for k in common_keys
        if latency in llama[k].get("chunks", {})
        and latency in gemma[k].get("chunks", {})
        and latency in nilechat[k].get("chunks", {})
    }

    def source_len_ok(key):
        """Keep turns of a readable length: very short or very long ones are hard to judge."""
        n_words = word_count("".join(entry_chunks(llama[key], latency)["source_chunks"]))
        return min_words <= n_words <= max_words

    def any_fallback(key):
        return (
            entry_fallback(llama[key], latency)
            or entry_fallback(gemma[key], latency)
            or entry_fallback(nilechat[key], latency)
        )

    success_pool = [k for k in common_keys if not any_fallback(k) and source_len_ok(k)]
    fallback_pool = [k for k in common_keys if any_fallback(k) and source_len_ok(k)]

    rng = random.Random(seed)
    success_keys = rng.sample(success_pool, min(n_success, len(success_pool)))
    fallback_keys = rng.sample(fallback_pool, min(n_fallback, len(fallback_pool)))

    return (
        [("all chunkers succeeded", k, llama, gemma, nilechat) for k in success_keys]
        + [("contains >=1 fallback", k, llama, gemma, nilechat) for k in fallback_keys]
    )


def render_chunk_table(examples, latency):
    """Build the chunk-comparison table: one block per turn, four rows inside it.

    The first row is the uncut reference sentence pair, then one row per chunker showing where it
    cut, with the chunks joined by " / ". Each chunker row carries a Y/N checklist."""
    if not examples:
        return "<p>No examples matched the requested filters.</p>"
    rows = []
    for status, key, llama, gemma, nilechat in examples:
        conv_id, turn_idx = key
        entries = [("llama", llama[key]), ("gemma", gemma[key]), ("nilechat", nilechat[key])]
        first_chunks = entry_chunks(entries[0][1], latency)
        ref_en = " ".join(first_chunks["source_chunks"])
        ref_ar = " ".join(first_chunks["target_chunks"])
        n_rows = len(entries) + 1
        rows.append(
            f'<tr><td rowspan="{n_rows}">{esc(f"{conv_id} / {turn_idx}")}<br>'
            f'<span class="badge">{esc(status)}</span></td>'
            f'<td class="chunker">reference</td>'
            f'<td>{esc(ref_en)}</td><td lang="ar" dir="rtl">{esc(ref_ar)}</td>'
            f'<td></td></tr>'
        )
        for name, entry in entries:
            chunk_data = entry_chunks(entry, latency)
            fb = " (fallback)" if entry_fallback(entry, latency) else ""
            en = " / ".join(chunk_data["source_chunks"])
            ar = " / ".join(chunk_data["target_chunks"])
            checklist = (
                '<div class="checklist" dir="ltr">'
                'Chunk reasonable? Y / N &nbsp; Note: __________<br>'
                'Chunk match meaning? Y / N &nbsp; Note: __________'
                '</div>'
            )
            rows.append(
                f'<tr><td class="chunker">{esc(name)}{fb}</td>'
                f'<td>{esc(en)}</td><td lang="ar" dir="rtl">{esc(ar)}</td>'
                f'<td dir="ltr">{checklist}</td></tr>'
            )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Turn</th><th>Chunker</th>'
        '<th>English chunks</th><th>Egyptian Arabic chunks</th><th>Review</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table></div>"
    )


# ---------- translation section ----------

def load_predictions_maybe(path):
    """Load one cell's predictions keyed by index, or None if that cell was never evaluated."""
    try:
        return {p["index"]: p for p in load_json(path)}
    except FileNotFoundError:
        return None


def pick_translation_examples(model_key, variant, latency, results_root,
                               n_good, n_bad, n_repetition, n_codeswitch, n_random, seed):
    """Choose which translations go in the review sheet, from five different buckets.

    Sampling only the worst or only random examples would give a skewed picture, so the picks are
    the lowest and highest COMET, detected repetition loops, code-switched text, and random.
    Each index is used once: a bucket only draws from what earlier buckets did not take."""
    east8b_preds = load_predictions_maybe(f"{results_root}/east8b/{variant}/{latency}/prediction.json")
    nilechat_preds = load_predictions_maybe(f"{results_root}/nilechat/{variant}/{latency}/prediction.json")
    if east8b_preds is None and nilechat_preds is None:
        raise FileNotFoundError(
            f"No prediction.json under {results_root} for variant={variant} latency={latency} "
            "(neither east8b nor nilechat was evaluated on this cell)"
        )

    # --model chooses which system's COMET scores decide "best" and "worst". If only one system
    # was evaluated for this cell, that one is used for both the ranking and the row content.
    primary = east8b_preds if model_key == "east8b" else nilechat_preds
    if primary is None:
        primary = east8b_preds if east8b_preds is not None else nilechat_preds

    if east8b_preds is not None and nilechat_preds is not None:
        common_idx = sorted(set(east8b_preds) & set(nilechat_preds))
    else:
        common_idx = sorted(primary.keys())

    # Rank by COMET, worst first. Sentences with no COMET score cannot be ranked.
    scored = [i for i in common_idx if "COMET" in primary[i]]
    ranked = sorted(scored, key=lambda i: primary[i]["COMET"])

    picked = set()
    picks = []

    worst = ranked[:n_bad]
    picks += [("lowest COMET", i) for i in worst]
    picked.update(worst)

    best = ranked[-n_good:] if n_good else []
    picks += [("highest COMET", i) for i in reversed(best)]
    picked.update(best)

    def any_model_repeats(i):
        """True if either system got stuck repeating a word on this sentence."""
        preds = [p for p in (east8b_preds, nilechat_preds) if p is not None]
        return any(has_repetition_loop(p[i]["prediction"]) for p in preds)

    repetition_pool = [i for i in scored if i not in picked and any_model_repeats(i)]
    rng = random.Random(seed)
    repetition_pick = rng.sample(repetition_pool, min(n_repetition, len(repetition_pool)))
    picks += [("repetition loop detected", i) for i in repetition_pick]
    picked.update(repetition_pick)

    codeswitch_pool = [
        i for i in scored
        if i not in picked and (has_code_switch(primary[i]["reference"])
                                 or has_code_switch(primary[i]["prediction"]))
    ]
    codeswitch_pick = rng.sample(codeswitch_pool, min(n_codeswitch, len(codeswitch_pool)))
    picks += [("code-switched content", i) for i in codeswitch_pick]
    picked.update(codeswitch_pick)

    # The random bucket draws from every remaining index, including unscored ones, so the sheet
    # is not made up only of cases that some filter flagged.
    remaining = [i for i in common_idx if i not in picked]
    random_pick = rng.sample(remaining, min(n_random, len(remaining)))
    picks += [("random", i) for i in random_pick]

    return picks, east8b_preds, nilechat_preds


def render_translation_table(picks, east8b_preds, nilechat_preds):
    """Build the translation table: one row per picked sentence.

    Each row shows the English source, the human Egyptian reference, and both systems' output
    with their scores and a Y/N checklist."""
    if not picks:
        return "<p>No examples available.</p>"

    def score_str(p):
        """Format whichever of BLEU / COMET / AL this prediction happens to carry."""
        parts = []
        if "BLEU" in p:
            parts.append(f'BLEU {p["BLEU"]:.1f}')
        if "COMET" in p:
            parts.append(f'COMET {p["COMET"]:.1f}')
        if "AL" in p:
            parts.append(f'AL {p["AL"]:.1f}')
        return " · ".join(parts)

    checklist = (
        '<div class="checklist" dir="ltr">'
        'Translation correct? Y / N &nbsp; Note: __________<br>'
        'Dialect correct? Y / N &nbsp; Note: __________'
        '</div>'
    )

    def output_cell(preds, idx, model_label):
        """One system's cell for a row, or a placeholder if that system was not evaluated."""
        if preds is None:
            return f"<td>({esc(model_label)} not evaluated for this cell)</td>"
        p = preds[idx]
        return (
            f"<td lang=\"ar\" dir=\"rtl\">{esc(p['prediction'].strip())}<br>"
            f'<span class="score">{esc(model_label)}: {esc(score_str(p))}</span>{checklist}</td>'
        )

    rows = []
    ref_source = east8b_preds if east8b_preds is not None else nilechat_preds
    for label, idx in picks:
        e = ref_source[idx]
        rows.append(
            f'<tr><td>idx {idx}<br><span class="badge">{esc(label)}</span></td>'
            f"<td>{esc(e['source'].strip())}</td>"
            f"<td lang=\"ar\" dir=\"rtl\">{esc(e['reference'].strip())}</td>"
            + output_cell(east8b_preds, idx, "EAST-8B")
            + output_cell(nilechat_preds, idx, "Nile-Chat")
            + "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>idx / bucket</th><th>English source</th>'
        "<th>Reference (Egyptian dialect)</th><th>EAST-8B output + review</th>"
        "<th>Nile-Chat output + review</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Combined Chunk + Translation Review — {model} / {variant} / {latency}</title>
<style>
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{ background: #FFFFFF; color: #000000; font-family: Georgia, "Times New Roman", Times, serif; line-height: 1.5; }}
  [lang="ar"] {{ font-family: "Segoe UI", Tahoma, "Noto Sans Arabic", Arial, sans-serif; line-height: 1.8; }}
  .page {{ max-width: 1000px; margin: 0 auto; padding: 48px 24px 80px; }}
  header.masthead {{ margin-bottom: 40px; }}
  header.masthead .eyebrow {{ font-family: ui-monospace, "Cascadia Code", "SF Mono", Consolas, monospace; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; margin: 0 0 10px; }}
  header.masthead h1 {{ font-size: 1.7rem; font-weight: 700; line-height: 1.2; margin: 0 0 14px; }}
  header.masthead p {{ font-size: 0.98rem; max-width: 84ch; margin: 0 0 8px; }}
  h2 {{ font-size: 1.25rem; font-weight: 700; margin: 36px 0 6px; }}
  p.block-note {{ font-size: 0.92rem; max-width: 84ch; margin: 0 0 18px; }}
  ol.instructions {{ margin: 0 0 18px; padding-left: 22px; max-width: 84ch; }}
  ol.instructions li {{ font-size: 0.92rem; margin-bottom: 10px; }}
  hr.rule {{ border: none; border-top: 1px solid #000; margin: 28px 0; }}
  .table-wrap {{ overflow-x: auto; margin: 10px 0 20px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; }}
  th, td {{ border: 1px solid #000; padding: 6px 10px; text-align: left; vertical-align: top; }}
  th {{ font-weight: 700; font-size: 0.76rem; letter-spacing: 0.02em; text-transform: uppercase; }}
  td.chunker {{ font-family: ui-monospace, "Cascadia Code", "SF Mono", Consolas, monospace; font-size: 0.82rem; white-space: nowrap; }}
  td[lang="ar"] {{ font-size: 0.98rem; }}
  .badge {{ display: inline-block; font-size: 0.7rem; padding: 1px 6px; border: 1px solid #000; border-radius: 3px; margin-top: 4px; }}
  .score {{ display: inline-block; font-size: 0.72rem; color: #444; margin-top: 4px; font-family: ui-monospace, "Cascadia Code", "SF Mono", Consolas, monospace; }}
  .checklist {{ margin-top: 6px; padding-top: 4px; border-top: 1px dashed #999; font-size: 0.78rem; color: #333; }}
  footer.colophon {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid #000; font-size: 0.85rem; }}
  @media print {{ body {{ padding: 0; }} .page {{ max-width: none; padding: 0; }} }}
</style>
</head>
<body>
<div class="page">
  <header class="masthead">
    <p class="eyebrow">EAST · Alexandria · {model} / {variant} / {latency}</p>
    <h1>Combined Chunk + Translation Review</h1>
    <p>Two things to judge: whether each chunker's cut points make sense, and whether each model's
      translation is correct and reads as genuine Egyptian dialect. Examples are pulled from several
      different buckets on purpose (highest score, lowest score, a detected repetition-loop failure,
      code-switched content, and random) so you're seeing different kinds of cases, not just "the worst
      ones" or "the best ones."</p>
  </header>

  <h2>Instructions for reviewers</h2>
  <ol class="instructions">
    <li>For each <strong>chunk</strong> row: <strong>Chunk reasonable?</strong> — does this cut fall
      somewhere a real interpreter could naturally start speaking? <strong>Chunk match meaning?</strong>
      — does this Arabic chunk actually mean the same thing as the English chunk next to it? Circle Y or
      N for each; add a short note if something specific stands out.</li>
    <li>For each <strong>translation</strong> row: <strong>Translation correct?</strong> — does the
      meaning match the reference? <strong>Dialect correct?</strong> — does it read as genuine Egyptian
      dialect, not formal/MSA or stilted? Circle Y or N for each; note briefly if you circle N.</li>
    <li>Please don't skip the "highest COMET" rows even though they're the automatic metric's most
      confident cases — this project has already found cases where a high score hid a real error.</li>
  </ol>

  {chunk_sections}

  <hr class="rule">

  <h2>Translation comparison ({variant} / {latency})</h2>
  <p class="block-note">Source, human reference, and both models' actual output, sampled from five
    different selection criteria (see the bucket label on each row) rather than one random sample.</p>
  {translation_table}

  <footer class="colophon">
    Generated by <code>east_scripts/eval/generate_combined_review_doc.py</code> from
    <code>data/mt_data/train_data/chunks-*.json</code> and
    <code>{results_root}/**/prediction.json</code> on Katana.
  </footer>
</div>
</body>
</html>
"""


def main():
    """Assemble the review sheet: chunk sections first, then the translation table, then write
    the HTML file and print what went into it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nilechat", choices=["east8b", "nilechat"])
    parser.add_argument("--variant", default="chunk-llama")
    parser.add_argument("--latency", default="high")
    parser.add_argument("--chunk_latencies", nargs="+", default=["low", "medium"],
                         choices=["low", "medium", "high"],
                         help="Which granularities to review chunking at (high is usually one whole chunk)")
    parser.add_argument("--results_root", default="results/granularity_comparison_alldomains")
    parser.add_argument("--n_chunk_success", type=int, default=6)
    parser.add_argument("--n_chunk_fallback", type=int, default=4)
    parser.add_argument("--n_translation_good", type=int, default=4)
    parser.add_argument("--n_translation_bad", type=int, default=4)
    parser.add_argument("--n_translation_repetition", type=int, default=3)
    parser.add_argument("--n_translation_codeswitch", type=int, default=3)
    parser.add_argument("--n_translation_random", type=int, default=4)
    parser.add_argument("--min_words", type=int, default=4)
    parser.add_argument("--max_words", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="combined_review.html")
    args = parser.parse_args()

    # Step 1: one chunk-boundary section per requested latency, each with its own heading.
    chunk_sections = []
    total_chunk_examples = 0
    for lat in args.chunk_latencies:
        chunk_examples = pick_chunk_examples(
            args.n_chunk_success, args.n_chunk_fallback, args.seed, args.min_words, args.max_words, lat,
        )
        total_chunk_examples += len(chunk_examples)
        chunk_sections.append(
            f'<h2>Chunk-boundary comparison ({esc(lat)} latency)</h2>\n'
            '<p class="block-note">Same turn, cut independently by each chunker (llama / gemma / nilechat). '
            'Includes both clean successes and cases where at least one chunker fell back to a mechanical '
            'backup, so you can see what a failure actually looks like next to a success.</p>\n'
            + render_chunk_table(chunk_examples, lat)
        )

    # Step 2: the translation examples for the single --variant / --latency cell being reviewed.
    picks, east8b_preds, nilechat_preds = pick_translation_examples(
        args.model, args.variant, args.latency, args.results_root,
        args.n_translation_good, args.n_translation_bad, args.n_translation_repetition,
        args.n_translation_codeswitch, args.n_translation_random, args.seed,
    )

    # Step 3: drop the rendered tables into the page template and write the file.
    out_html = HTML_TEMPLATE.format(
        model=esc(args.model), variant=esc(args.variant), latency=esc(args.latency),
        results_root=esc(args.results_root),
        chunk_sections="\n<hr class=\"rule\">\n".join(chunk_sections),
        translation_table=render_translation_table(picks, east8b_preds, nilechat_preds),
    )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out_html)

    # Step 4: report how many examples each bucket contributed, so an empty bucket (for example
    # no repetition loops found) is visible without opening the page.
    bucket_counts = {}
    for label, _ in picks:
        bucket_counts[label] = bucket_counts.get(label, 0) + 1
    print(f"Wrote {args.out}: {total_chunk_examples} chunk examples across {args.chunk_latencies}, "
          f"{len(picks)} translation examples "
          f"({bucket_counts}).")


if __name__ == "__main__":
    main()
