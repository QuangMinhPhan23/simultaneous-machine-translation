"""Method C: build chunks from word alignment instead of asking a model to segment.

Reads the Step 1 parallel files, aligns English words to target words with a multilingual encoder,
works out how much of the target may be written after each English prefix, and cuts there. The
target is only cut, never re-translated. Writes the same chunks-file schema as generate_chunks.py.

Usage:
  python word_order_study/align_chunks.py --language vietnamese \
      --output word_order_study/data/vietnamese/chunks-aligned.json
"""
import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "east_scripts", "data"))

import chunk_prompts
from segment import atomic_units
from syntax_boundaries import choose_cuts, require_parser
from build_arabic_simt_sft_data import LATENCY_CHUNK_WORDS

EXT = {"vietnamese": "vi", "msa": "ar", "korean": "ko", "saudi": "ar", "egyptian": "ar"}

# A middle layer of multilingual BERT aligns words best; the last layer is too specialised.
ALIGN_LAYER = 8


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


class WordAligner:
    """Aligns the words of two sentences using a multilingual encoder.

    A word is embedded by averaging its subword vectors, and a pair is kept only if the two words
    are each other's best match."""

    def __init__(self, model_name):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def _word_vectors(self, words):
        """One vector per input word, averaged over that word's subword pieces."""
        torch = self.torch
        enc = self.tokenizer(words, is_split_into_words=True, return_tensors="pt",
                             truncation=True, max_length=512)
        word_ids = enc.word_ids(0)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = self.model(**enc).hidden_states[ALIGN_LAYER][0]

        sums, counts = {}, Counter()
        for pos, w in enumerate(word_ids):
            if w is None:
                continue
            sums[w] = hidden[pos] if w not in sums else sums[w] + hidden[pos]
            counts[w] += 1
        if not sums:
            return None, []
        kept = sorted(sums)
        vecs = torch.stack([sums[w] / counts[w] for w in kept])
        return torch.nn.functional.normalize(vecs, dim=-1), kept

    def align(self, src_words, tgt_words):
        """Return the list of (source index, target index) pairs that match both ways."""
        torch = self.torch
        src_vecs, src_kept = self._word_vectors(src_words)
        tgt_vecs, tgt_kept = self._word_vectors(tgt_words)
        if src_vecs is None or tgt_vecs is None:
            return []
        sim = src_vecs @ tgt_vecs.T
        # Keep i, j only when source i picks target j and j picks i back.
        best_tgt = sim.argmax(dim=1)
        best_src = sim.argmax(dim=0)
        pairs = []
        for i in range(sim.shape[0]):
            j = int(best_tgt[i])
            if int(best_src[j]) == i:
                pairs.append((src_kept[i], tgt_kept[j]))
        return pairs


def writable_target_prefix(pairs, n_src, n_tgt):
    """For each English prefix length, how many target words may be written already.

    A target word is safe only once every English word aligned to it has been read. Unaligned
    target words never block."""
    needs_src_upto = [-1] * n_tgt
    for i, j in pairs:
        if 0 <= j < n_tgt:
            needs_src_upto[j] = max(needs_src_upto[j], i)

    allowed = [0] * (n_src + 1)
    t = 0
    for i in range(n_src + 1):
        # After reading English words 0..i-1, move past every target word whose English is covered.
        while t < n_tgt and needs_src_upto[t] < i:
            t += 1
        allowed[i] = t
    allowed[n_src] = n_tgt  # the last chunk writes whatever is left
    return allowed


def chunk_by_alignment(src_units, tgt_units, pairs, chunk_size, source_cuts=None):
    """Cut both sides at aligned points, aiming for chunk_size source words per chunk.

    Pass source_cuts to say where the English may be cut; otherwise it is cut every chunk_size words."""
    n_src, n_tgt = len(src_units), len(tgt_units)
    if n_src == 0 or n_tgt == 0:
        return [], []
    allowed = writable_target_prefix(pairs, n_src, n_tgt)
    cut_points = source_cuts if source_cuts else list(range(chunk_size, n_src, chunk_size))

    src_chunks, tgt_chunks = [], []
    s_prev = t_prev = 0
    for s in cut_points:
        if s >= n_src:
            break
        t = allowed[s]
        # Cut only if both sides move forward and some target is left for the final chunk.
        if t > t_prev and t < n_tgt:
            src_chunks.append(" ".join(src_units[s_prev:s]))
            tgt_chunks.append(" ".join(tgt_units[t_prev:t]))
            s_prev, t_prev = s, t
    # Whatever is left becomes the final chunk, so both sides are covered.
    src_chunks.append(" ".join(src_units[s_prev:]))
    tgt_chunks.append(" ".join(tgt_units[t_prev:]))
    return src_chunks, tgt_chunks


def main():
    """Chunk one language by alignment and write a chunks file in the usual schema."""
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, choices=chunk_prompts.LANGUAGES)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="bert-base-multilingual-cased")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--syntactic", action="store_true",
                    help="Method D: cut English at syntactic boundaries instead of every k words")
    ap.add_argument("--max_span", type=int, default=7, help="max words in one read step (--syntactic)")
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.syntactic:
        require_parser()   # stop now instead of quietly writing fixed-width chunks
    ext = EXT[args.language]
    src_lines = read_lines(os.path.join(DATA, args.language, f"{args.split}.en"))
    tgt_lines = read_lines(os.path.join(DATA, args.language, f"{args.split}.{ext}"))
    if len(src_lines) != len(tgt_lines):
        raise SystemExit(f"line counts differ: {len(src_lines)} vs {len(tgt_lines)}")
    n = len(src_lines) if args.max_examples is None else min(args.max_examples, len(src_lines))

    print(f"Loading aligner: {args.model}")
    aligner = WordAligner(args.model)

    entries, stats = [], Counter()
    for i in range(n):
        s, t = src_lines[i].strip(), tgt_lines[i].strip()
        if not s or not t:
            continue
        src_units = atomic_units(s, "english")
        tgt_units = atomic_units(t, args.language)
        try:
            pairs = aligner.align(src_units, tgt_units)
        except Exception as e:
            pairs = []
            stats["align_error"] += 1
        stats["aligned_pairs"] += len(pairs)

        chunks = {}
        for latency, size in LATENCY_CHUNK_WORDS.items():
            cuts = choose_cuts(src_units, size, args.max_span) if args.syntactic else None
            sc, tc = chunk_by_alignment(src_units, tgt_units, pairs, size, cuts)
            chunks[latency] = {"source_chunks": sc, "target_chunks": tc}
            stats[f"chunks_{latency}"] += len(sc)

        entries.append({
            "idx": i, "method": "aligned", "language": args.language,
            "chunks": chunks,
            # Alignment never falls back, but the keys stay so other scripts read this file unchanged.
            "fallback": False,
            "fallback_by_latency": {lat: False for lat in LATENCY_CHUNK_WORDS},
            "fallback_reason_by_latency": {lat: None for lat in LATENCY_CHUNK_WORDS},
        })
        if len(entries) % args.save_every == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
            print(f"  ...{i + 1}/{n} processed", file=sys.stderr)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"\n{args.language}/aligned: {len(entries)} entries -> {args.output}")
    print(f"  mean aligned word pairs per sentence: {stats['aligned_pairs'] / max(1, len(entries)):.1f}")
    for lat in LATENCY_CHUNK_WORDS:
        print(f"  {lat}: {stats[f'chunks_{lat}'] / max(1, len(entries)):.1f} chunks per sentence")
    if stats["align_error"]:
        print(f"  WARNING: {stats['align_error']} sentences failed to align and were left in one chunk")


if __name__ == "__main__":
    main()
