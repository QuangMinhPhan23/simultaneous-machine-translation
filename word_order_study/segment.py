"""Word-aware splitting for the fallback chunker.

Splitting on whitespace is fine for English, Arabic and Korean, but Vietnamese whitespace separates
syllables, so a split can cut through a word ("dan chu" is one word, democracy). atomic_units()
groups Vietnamese syllables back into words with underthesea or pyvi, and warns once and uses plain
whitespace if neither is installed.
"""
import sys

_WARNED = set()

# Only Vietnamese needs grouping, the others already split into words on whitespace.
NEEDS_GROUPING = {"vietnamese"}


def _warn_once(key, message):
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"WARNING: {message}", file=sys.stderr)


def _vietnamese_words(text):
    """Group Vietnamese syllables into words. Joining the units with spaces gives back the input."""
    try:
        from underthesea import word_tokenize
        # underthesea gives a multi-syllable word as one string with spaces inside.
        units = word_tokenize(text)
    except ImportError:
        try:
            from pyvi import ViTokenizer
            # pyvi joins the syllables of a word with underscores, e.g. "dan_chu".
            units = [u.replace("_", " ") for u in ViTokenizer.tokenize(text).split()]
        except ImportError:
            _warn_once("vi", "no Vietnamese segmenter (underthesea or pyvi) installed; "
                             "falling back to syllable splitting, which can cut words in half")
            return text.split()

    # If the segmenter changed the characters, keep the raw text instead.
    if "".join(units).replace(" ", "") != text.replace(" ", ""):
        _warn_once("vi_mismatch", "Vietnamese segmenter changed the text; using whitespace instead")
        return text.split()
    return units


def atomic_units(text, language):
    """Split text into the units a chunk boundary must not cut through."""
    if language in NEEDS_GROUPING:
        return _vietnamese_words(text)
    return text.split()


def _split_into_k(units, k):
    """Cut a unit list into k pieces of nearly equal size and join each piece into a string."""
    n = len(units)
    return [" ".join(units[(i * n) // k:((i + 1) * n) // k]) for i in range(k)]


def chunk_sentence_pair_aware(source, target, chunk_size, language):
    """Word-count chunking that never cuts a target word in half.

    Both sides get the same number of chunks, taken from the longer side and split proportionally."""
    source_units = atomic_units(source, "english")
    target_units = atomic_units(target, language)
    if not source_units or not target_units:
        return [], []
    k = max(1, -(-max(len(source_units), len(target_units)) // chunk_size))
    k = min(k, len(source_units), len(target_units))
    return _split_into_k(source_units, k), _split_into_k(target_units, k)


if __name__ == "__main__":
    # Quick check: "dan chu" (democracy) must stay in one unit.
    sys.stdout.reconfigure(encoding="utf-8")
    vi = "Ở mức này , tôi sợ rằng chính nền dân chủ , không phải hệ thống một đảng của Trung Quốc ."
    en = "At this rate , I 'm afraid it is democracy , not China 's one-party system ."
    for lang_units, label in ((atomic_units(vi, "vietnamese"), "vietnamese"),
                              (atomic_units(vi, "korean"), "plain whitespace")):
        print(f"{label:18}: {len(lang_units)} units")
    sc, tc = chunk_sentence_pair_aware(en, vi, 2, "vietnamese")
    print(f"\nlow-latency chunks: {len(sc)}")
    for s, t in zip(sc, tc):
        print(f"  EN: {s}\n  VI: {t}")
    print("\nreconstructs:", " ".join(tc).split() == vi.split())
