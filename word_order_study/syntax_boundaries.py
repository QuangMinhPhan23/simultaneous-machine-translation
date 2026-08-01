"""Method D: pick English chunk boundaries at syntactic joints instead of every k words.

candidate_boundaries() uses spaCy to find positions where a phrase ends, and choose_cuts() takes
the ones closest to the wanted chunk size, forcing a cut after max_span words when none is found.
Without a spaCy model every position is a candidate, which gives fixed-width chunks.
"""
import sys

_NLP = None
_WARNED = False


def _nlp():
    """Load spaCy once. Returns None if no English model is installed."""
    global _NLP, _WARNED
    if _NLP is None:
        try:
            import spacy
            for name in ("en_core_web_trf", "en_core_web_sm"):
                try:
                    _NLP = spacy.load(name, disable=["ner", "lemmatizer"])
                    break
                except OSError:
                    continue
        except ImportError:
            _NLP = None
        if _NLP is None and not _WARNED:
            _WARNED = True
            print("WARNING: no spaCy English model; falling back to fixed-width source chunks",
                  file=sys.stderr)
    return _NLP


def require_parser():
    """Stop the run if spaCy is missing, instead of quietly producing fixed-width chunks."""
    if _nlp() is None:
        raise SystemExit(
            "ERROR: --syntactic needs a spaCy English model, and none is installed.\n"
            "  pip install spacy && python -m spacy download en_core_web_sm")


def candidate_boundaries(words):
    """Sorted word counts where a read step could end. The sentence end is always included."""
    nlp = _nlp()
    if nlp is None:
        return list(range(1, len(words) + 1))

    # Parse the words as given, so positions match our word list exactly.
    from spacy.tokens import Doc
    doc = Doc(nlp.vocab, words=words)
    for _, proc in nlp.pipeline:
        doc = proc(doc)

    bounds = set()
    for nc in doc.noun_chunks:                       # end of a noun phrase
        bounds.add(nc.end)
    for tok in doc:
        if tok.pos_ in ("ADP", "SCONJ") and tok.i > 0:
            bounds.add(tok.i)                        # a new phrase starts here, so cut before it
        if tok.pos_ == "PUNCT":
            bounds.add(tok.i + 1)                    # after the punctuation
        if tok.pos_ in ("VERB", "AUX") and tok.i > 0:
            # before a verb, but only once a subject has appeared
            if any(t.dep_ in ("nsubj", "nsubjpass") and t.i < tok.i for t in doc):
                bounds.add(tok.i)
    bounds.add(len(words))
    return sorted(b for b in bounds if 0 < b <= len(words))


def choose_cuts(words, chunk_size, max_span=7):
    """Pick the boundaries to use, aiming for chunk_size words per step.

    Takes the candidate closest to the wanted length, or cuts at max_span if there is none."""
    cands = candidate_boundaries(words)
    cuts, prev = [], 0
    while prev < len(words):
        target = prev + chunk_size
        limit = prev + max_span
        usable = [c for c in cands if prev < c <= limit]
        if usable:
            pick = min(usable, key=lambda c: (abs(c - target), c))
        else:
            pick = min(limit, len(words))
        cuts.append(pick)
        prev = pick
    return cuts
