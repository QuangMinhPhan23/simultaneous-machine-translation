"""
Dialect-register scoring for Arabic text, mainly for filtering training targets.

  ALDiScorer:      0..1 dialectness (AMR-KELEG/Sentence-ALDi), 0 = MSA, 1 = fully dialectal
  DialectIDScorer: CAMeL-Lab NADI country head, returns P(target country label)

Needs transformers + torch; both models download on first use and run much faster on a GPU.
Run with --selftest to check ALDi scores an Egyptian sentence above an MSA one.
"""
import argparse
import json
import sys


def _require_transformers():
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "score_dialectness needs `transformers` and `torch`.\n"
            "  pip install 'transformers>=4.30' torch\n"
            f"(import failed: {e})"
        )


def _pick_device(device):
    import torch
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class ALDiScorer:
    """Continuous 0..1 dialectness (higher = more dialectal, further from MSA)."""

    def __init__(self, model_name="AMR-KELEG/Sentence-ALDi", device=None,
                 batch_size=32, max_length=128):
        _require_transformers()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = _pick_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

    def score(self, texts):
        """texts: list[str] -> list[float] in [0, 1]. Empty/whitespace -> 0.0."""
        out = [None] * len(texts)
        idx = [i for i, t in enumerate(texts) if t and t.strip()]
        for i in [j for j in range(len(texts)) if j not in set(idx)]:
            out[i] = 0.0
        for start in range(0, len(idx), self.batch_size):
            batch_idx = idx[start:start + self.batch_size]
            batch = [texts[i] for i in batch_idx]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits  # [B, 1] regression head
            for i, val in zip(batch_idx, logits[:, 0].tolist()):
                out[i] = min(max(0.0, val), 1.0)
        return out


class DialectIDScorer:
    """CAMeL-Lab NADI country head. .score() returns dicts holding the target label's
    probability and the top label, so callers can filter on either one."""

    def __init__(self, model_name="CAMeL-Lab/bert-base-arabic-camelbert-mix-did-nadi",
                 target_label="Egypt", device=None, batch_size=32, max_length=128):
        _require_transformers()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = _pick_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

        self.id2label = self.model.config.id2label
        label2id = {v: k for k, v in self.id2label.items()}
        if target_label not in label2id:
            raise ValueError(
                f"target_label {target_label!r} not in this model's labels. "
                f"Available: {sorted(label2id)}"
            )
        self.target_label = target_label
        self.target_id = label2id[target_label]

    def score(self, texts):
        out = [None] * len(texts)
        idx = [i for i, t in enumerate(texts) if t and t.strip()]
        for i in [j for j in range(len(texts)) if j not in set(idx)]:
            out[i] = {"p_target": 0.0, "top_label": None, "top_p": 0.0}
        for start in range(0, len(idx), self.batch_size):
            batch_idx = idx[start:start + self.batch_size]
            batch = [texts[i] for i in batch_idx]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probs = self.model(**enc).logits.softmax(dim=-1)
            top_p, top_i = probs.max(dim=-1)
            for i, p_t, tp, ti in zip(batch_idx, probs[:, self.target_id].tolist(),
                                      top_p.tolist(), top_i.tolist()):
                out[i] = {"p_target": p_t, "top_label": self.id2label[ti], "top_p": tp}
        return out


# ---- text extraction from the project's various json shapes ----
_TEXT_FIELDS = ("prediction", "output", "reference", "target", "text", "hypothesis")


def extract_texts(records, field=None):
    """records: list of dicts or list of str -> list[str]. Picks the first field of
    _TEXT_FIELDS that is present, unless `field` is given."""
    if records and isinstance(records[0], str):
        return list(records)
    if field is None:
        for f in _TEXT_FIELDS:
            if records and isinstance(records[0], dict) and f in records[0]:
                field = f
                break
        if field is None:
            raise SystemExit(
                f"Could not find a text field in the first record "
                f"(looked for {_TEXT_FIELDS}). Pass --text_field explicitly."
            )
    return [r.get(field, "") if isinstance(r, dict) else str(r) for r in records]


def _summ(vals):
    xs = sorted(vals)
    n = len(xs)
    if n == 0:
        return "n=0"
    mean = sum(xs) / n
    return f"n={n} mean={mean:.3f} min={xs[0]:.3f} p50={xs[n // 2]:.3f} max={xs[-1]:.3f}"


def _selftest():
    msa = "أريد أن أذهب إلى السوق الآن"   # "I want to go to the market now", MSA
    egy = "عايز أروح السوق دلوقتي"                 # same sentence, Egyptian
    print("Loading ALDi (first run downloads the model)...", file=sys.stderr)
    aldi = ALDiScorer()
    s_msa, s_egy = aldi.score([msa, egy])
    print(f"ALDi  MSA={s_msa:.3f}  EGY={s_egy:.3f}  (expect MSA < EGY)")
    assert s_egy > s_msa, "ALDi selftest FAILED: dialectal sentence did not score higher than MSA"
    print("selftest OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", help="json (list of dicts or list of str) or a .txt (one sentence/line)")
    parser.add_argument("--text_field", default=None,
                         help=f"field to score; auto-detected from {_TEXT_FIELDS} if omitted")
    parser.add_argument("--scorer", choices=["aldi", "did", "both"], default="aldi")
    parser.add_argument("--target_label", default="Egypt", help="DID label to report P() for")
    parser.add_argument("--min_dialectness", type=float, default=None,
                         help="if set, add keep=bool (ALDi score >= this) and print keep/drop counts")
    parser.add_argument("--device", default=None, help="cuda|cpu (auto if omitted)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", default=None, help="write per-record scores to this json")
    parser.add_argument("--selftest", action="store_true",
                         help="score one MSA and one Egyptian sentence, check MSA scores lower, exit")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        sys.exit(0)

    if not args.input:
        parser.error("--input is required (or use --selftest)")

    if args.input.endswith(".txt"):
        with open(args.input, encoding="utf-8") as f:
            records = [ln.rstrip("\n") for ln in f if ln.strip()]
    else:
        with open(args.input, encoding="utf-8") as f:
            records = json.load(f)
    texts = extract_texts(records, args.text_field)

    aldi_scores = did_scores = None
    if args.scorer in ("aldi", "both"):
        aldi_scores = ALDiScorer(device=args.device, batch_size=args.batch_size).score(texts)
        print(f"ALDi dialectness: {_summ(aldi_scores)}", file=sys.stderr)
    if args.scorer in ("did", "both"):
        did = DialectIDScorer(target_label=args.target_label, device=args.device, batch_size=args.batch_size)
        did_scores = did.score(texts)
        p_tgt = [d["p_target"] for d in did_scores]
        n_top = sum(1 for d in did_scores if d["top_label"] == args.target_label)
        print(f"DID P({args.target_label}): {_summ(p_tgt)}; "
              f"top-label=={args.target_label} for {n_top}/{len(did_scores)}", file=sys.stderr)

    if args.min_dialectness is not None and aldi_scores is not None:
        keep = [s >= args.min_dialectness for s in aldi_scores]
        n_keep = sum(keep)
        print(f"keep (ALDi >= {args.min_dialectness}): {n_keep}/{len(keep)} "
              f"({100 * n_keep / max(1, len(keep)):.1f}%); drop {len(keep) - n_keep}", file=sys.stderr)

    if args.output:
        annotated = []
        for i, r in enumerate(records):
            row = dict(r) if isinstance(r, dict) else {"text": r}
            if aldi_scores is not None:
                row["aldi"] = aldi_scores[i]
                if args.min_dialectness is not None:
                    row["keep"] = aldi_scores[i] >= args.min_dialectness
            if did_scores is not None:
                row["did"] = did_scores[i]
            annotated.append(row)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(annotated, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(annotated)} scored records to {args.output}", file=sys.stderr)
