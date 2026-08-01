"""Chunk validator for the word-order study.

Checks that a model's chunks rebuild the sentence, comparing content characters only: after NFC
normalization, whitespace and punctuation are dropped. So a lost comma still passes, but a lost or
invented word fails. A latency that fails falls back to the word-count split."""
import difflib
import unicodedata

from generate_semantic_chunks import LATENCY_JSON_KEYS, extract_json_object
from build_arabic_simt_sft_data import LATENCY_CHUNK_WORDS, align_chunks, chunk_sentence_pair

DEFAULT_MIN_SIMILARITY = 0.97


def _is_punct_or_symbol(ch):
    """True if the character is punctuation or a symbol (Unicode category starting with P or S)."""
    return unicodedata.category(ch)[0] in ("P", "S")


def content_only(text):
    """NFC-normalize, then keep only content characters (drop whitespace + punctuation/symbols)."""
    text = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in text if not ch.isspace() and not _is_punct_or_symbol(ch))


def reconstructs(chunks, original, min_similarity):
    """True if joining the chunks gives back the original sentence, content characters only."""
    orig = content_only(original)
    # A sentence of only punctuation and spaces has no content to compare.
    if not orig:
        return True
    joined = content_only("".join(chunks))
    return difflib.SequenceMatcher(None, joined, orig).ratio() >= min_similarity


def _strip_empty(chunks):
    """Drop anything in the list that is not a string, or is blank."""
    return [c for c in chunks if isinstance(c, str) and c.strip()]


def parse_chunk_response_wo(response_text, source, target, min_similarity=DEFAULT_MIN_SIMILARITY):
    """Same contract as generate_semantic_chunks.parse_chunk_response:
    returns (chunks_by_latency, fallback_by_latency, reason_by_latency)."""
    # No JSON in the reply means nothing is usable, so every latency falls back.
    parsed = extract_json_object(response_text)
    if not isinstance(parsed, dict):
        return (
            {lat: chunk_sentence_pair(source, target, w) for lat, w in LATENCY_CHUNK_WORDS.items()},
            {lat: True for lat in LATENCY_CHUNK_WORDS},
            {lat: "json_parse_failed" for lat in LATENCY_CHUNK_WORDS},
        )

    # Check each latency on its own, so one bad latency does not throw away the other two.
    chunks_by_latency, fallback_by_latency, reason_by_latency = {}, {}, {}
    for latency, json_key in LATENCY_JSON_KEYS.items():
        chunk_words = LATENCY_CHUNK_WORDS[latency]
        entry = parsed.get(json_key)
        reason = None
        sc = tc = None
        if not isinstance(entry, dict):
            reason = "missing_latency_key"
        else:
            sc, tc = entry.get("source_chunks"), entry.get("target_chunks")
            if not isinstance(sc, list) or not isinstance(tc, list):
                reason = "missing_or_wrong_type_keys"
            else:
                sc, tc = _strip_empty(sc), _strip_empty(tc)
                if not sc or not tc:
                    reason = "empty_chunk_list"
                elif not reconstructs(sc, source, min_similarity) or not reconstructs(tc, target, min_similarity):
                    reason = "reconstruction_mismatch"

        # Passed: keep the model's chunks, forced to the same count on both sides. Failed: split by words.
        if reason is None:
            chunks_by_latency[latency] = align_chunks(sc, tc)
        else:
            chunks_by_latency[latency] = chunk_sentence_pair(source, target, chunk_words)
        fallback_by_latency[latency] = reason is not None
        reason_by_latency[latency] = reason

    return chunks_by_latency, fallback_by_latency, reason_by_latency


def parse_single_latency(response_text, source, target, min_similarity=DEFAULT_MIN_SIMILARITY):
    """Validate a single-latency response and return aligned (source_chunks, target_chunks).

    Same checks as above. Returns None if the answer is unusable, so the caller can try again."""
    parsed = extract_json_object(response_text)
    if not isinstance(parsed, dict):
        return None
    sc, tc = parsed.get("source_chunks"), parsed.get("target_chunks")
    if not isinstance(sc, list) or not isinstance(tc, list):
        return None
    sc, tc = _strip_empty(sc), _strip_empty(tc)
    if not sc or not tc:
        return None
    if not reconstructs(sc, source, min_similarity) or not reconstructs(tc, target, min_similarity):
        return None
    return align_chunks(sc, tc)
