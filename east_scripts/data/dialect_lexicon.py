"""
Hand-curated function-word substitution tables used to build DPO negatives:
  1. dialect -> MSA injection (Stage 4), simulating the MSA-leakage failure mode.
  2. EG <-> LB marker swap (Stage 2), turning a correct reference into a wrong-dialect one.

Only Egyptian and Lebanese are covered. Substitution is whole-word with no morphology, so
neighbouring words may end up disagreeing in gender or number.
"""
import re

# dialectal_word -> MSA word.
EG_TO_MSA = {
    "عايز": "أريد",
    "عايزة": "أريد",
    "عاوز": "أريد",
    "مش": "ليس",
    "ده": "هذا",
    "دي": "هذه",
    "كده": "هكذا",
    "ليه": "لماذا",
    "فين": "أين",
    "إزاي": "كيف",
    "ازاي": "كيف",
    "دلوقتي": "الآن",
    "خالص": "إطلاقا",
    "علشان": "لأن",
    "بتاع": "الخاص بـ",
    "كويس": "جيد",
    "قوي": "جدا",
}

LB_TO_MSA = {
    "بدي": "أريد",
    "بدك": "تريد",
    "مو": "ليس",
    "هيدا": "هذا",
    "هيدي": "هذه",
    "هيك": "هكذا",
    "ليش": "لماذا",
    "وين": "أين",
    "هلق": "الآن",
    "كتير": "جدا",
    "منيح": "جيد",
}

# EG <-> LB swap, built from the two tables above via the MSA gloss they share.
# Only words that both dialects have an entry for can be swapped.
_SHARED_GLOSSES = set(EG_TO_MSA.values()) & set(LB_TO_MSA.values())
_EG_BY_GLOSS = {gloss: word for word, gloss in EG_TO_MSA.items()}
_LB_BY_GLOSS = {gloss: word for word, gloss in LB_TO_MSA.items()}
EG_TO_LB = {_EG_BY_GLOSS[g]: _LB_BY_GLOSS[g] for g in _SHARED_GLOSSES}
LB_TO_EG = {v: k for k, v in EG_TO_LB.items()}

DIALECT_TO_MSA_TABLES = {"EG": EG_TO_MSA, "LB": LB_TO_MSA}
DIALECT_SWAP_TABLES = {("EG", "LB"): EG_TO_LB, ("LB", "EG"): LB_TO_EG}


def _substitute(text, table):
    """Whole-word substitution, longest keys first so a longer entry is not shadowed by a
    shorter one sharing its prefix. Returns (new_text, n_substitutions) so callers can drop
    pairs where nothing matched."""
    if not table:
        return text, 0
    n_hits = 0
    for src in sorted(table, key=len, reverse=True):
        # The lookarounds require whitespace (or a string edge) on both sides, so the word is
        # only replaced when it stands alone and not inside a longer word.
        pattern = r"(?<!\S)" + re.escape(src) + r"(?!\S)"
        text, n = re.subn(pattern, table[src], text)
        n_hits += n
    return text, n_hits


def inject_msa(text, dialect):
    """Stage 4 perturbation: replace dialectal function words with their MSA
    equivalent, simulating the MSA-leakage failure mode."""
    table = DIALECT_TO_MSA_TABLES.get(dialect)
    if table is None:
        raise ValueError(f"No MSA substitution table for dialect {dialect!r}")
    return _substitute(text, table)


def count_msa_markers(text, dialect):
    """Counts the distinct MSA function words in this text, by whole-word overlap with the
    MSA side of the dialect table. The table is small, so a 0 does not prove the sentence
    is MSA-free: use ALDi (score_dialectness.py) when recall matters."""
    table = DIALECT_TO_MSA_TABLES.get(dialect)
    if table is None:
        raise ValueError(f"No MSA substitution table for dialect {dialect!r}")
    msa_words = set(table.values())
    return len(set(text.split()) & msa_words)


def swap_dialect(text, src_dialect, dst_dialect):
    """Stage 2 negative: render src_dialect text with dst_dialect markers,
    without touching content words, giving a same-content, wrong-dialect pair."""
    table = DIALECT_SWAP_TABLES.get((src_dialect, dst_dialect))
    if table is None:
        raise ValueError(f"No swap table for {src_dialect} -> {dst_dialect}")
    return _substitute(text, table)
