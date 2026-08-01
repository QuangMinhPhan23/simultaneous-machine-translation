"""Prompt registry for the word-order study: Method A (generic) vs Method B (language-specific).

Both methods share one prompt scaffold; Method B only adds a per-language word-order guidance block.
build_prompt() asks for all three latencies in one answer, build_prompt_single() for one latency.
"""
import json

LANGUAGES = ["vietnamese", "msa", "korean", "saudi", "egyptian"]

TGT_LANG_NAME = {
    "vietnamese": "Vietnamese",
    "msa": "Modern Standard Arabic (MSA)",
    "korean": "Korean",
    "saudi": "Saudi Arabic",
    "egyptian": "Egyptian Arabic",
}

# Method B word-order guidance, one block per language.
GUIDANCE = {
    "vietnamese": (
        "WORD-ORDER GUIDANCE (Vietnamese): Vietnamese has the same Subject-Verb-Object order as "
        "English, so translation proceeds with minimal reordering. Each English chunk should map "
        "directly to a Vietnamese chunk in the same order. Use short 2-3 word phrases at low latency, "
        "clause-level segments at medium, and full clauses or the whole sentence at high."
    ),
    "msa": (
        "WORD-ORDER GUIDANCE (Modern Standard Arabic): MSA typically uses Verb-Subject-Object order, "
        "unlike English SVO — the Arabic verb often precedes the subject. Ensure each English chunk "
        "carries enough context to yield a grammatically correct Arabic phrase, and keep the English "
        "subject and verb together in one chunk when possible, since the Arabic output reorders them. "
        "Use 2-4 word chunks at low latency, clause-level at medium, full clauses at high."
    ),
    "korean": (
        "WORD-ORDER GUIDANCE (Korean): Korean is Subject-Object-Verb — the verb comes at the END of "
        "the clause — and marks nouns with particles (은/는, 이/가, 을/를). The interpreter cannot "
        "produce the verb until enough of the English sentence is heard, so keep related noun phrases "
        "together and make each chunk carry enough meaning for a partial Korean output. Use 2-3 word "
        "chunks keeping noun phrases intact at low latency, chunks covering a subject-verb or "
        "verb-object pair at medium, and full clauses at high."
    ),
    "saudi": (
        "WORD-ORDER GUIDANCE (Saudi Arabic): Saudi Arabic often uses Verb-Subject-Object order, like "
        "MSA but with colloquial vocabulary and grammar. The interpreter may need the English verb "
        "before starting the Arabic output, so keep English subject-verb pairs together when possible. "
        "Use Saudi colloquial vocabulary and informal register, NOT MSA. Use 2-4 word chunks at low "
        "latency, clause-level at medium, full clauses at high."
    ),
    "egyptian": (
        "WORD-ORDER GUIDANCE (Egyptian Arabic): Egyptian Arabic generally follows Subject-Verb-Object "
        "order (closer to English than MSA), with colloquial vocabulary and grammar. Keep English "
        "subject-verb pairs together when helpful. Use Egyptian colloquial vocabulary and informal "
        "register, NOT MSA. Use 2-4 word chunks at low latency, clause-level at medium, full clauses "
        "at high."
    ),
}

# Shared English worked example.
_EX_EN = {
    "low": ["Good morning,", "I need", "ten kilograms of tomatoes", "for the restaurant."],
    "medium": ["Good morning, I need", "ten kilograms of tomatoes for the restaurant."],
    "high": ["Good morning, I need ten kilograms of tomatoes for the restaurant."],
}
_EX_EN_FULL = "Good morning, I need ten kilograms of tomatoes for the restaurant."

# Per-language target worked example. Joining a latency's chunks gives back the full string.
_EX_TGT = {
    "vietnamese": {
        "full": "Chào buổi sáng, tôi cần mười ki-lô cà chua cho nhà hàng.",
        "low": ["Chào buổi sáng,", "tôi cần", "mười ki-lô cà chua", "cho nhà hàng."],
        "medium": ["Chào buổi sáng, tôi cần", "mười ki-lô cà chua cho nhà hàng."],
        "high": ["Chào buổi sáng, tôi cần mười ki-lô cà chua cho nhà hàng."],
    },
    "msa": {
        "full": "صباح الخير، أحتاج عشرة كيلوغرامات من الطماطم للمطعم.",
        "low": ["صباح الخير،", "أحتاج", "عشرة كيلوغرامات من الطماطم", "للمطعم."],
        "medium": ["صباح الخير، أحتاج", "عشرة كيلوغرامات من الطماطم للمطعم."],
        "high": ["صباح الخير، أحتاج عشرة كيلوغرامات من الطماطم للمطعم."],
    },
    "saudi": {
        "full": "صباح الخير، أبي عشر كيلو طماط للمطعم.",
        "low": ["صباح الخير،", "أبي", "عشر كيلو طماط", "للمطعم."],
        "medium": ["صباح الخير، أبي", "عشر كيلو طماط للمطعم."],
        "high": ["صباح الخير، أبي عشر كيلو طماط للمطعم."],
    },
    "egyptian": {
        "full": "صباح الخير، عايز عشرة كيلو طماطم للمطعم.",
        "low": ["صباح الخير،", "عايز", "عشرة كيلو طماطم", "للمطعم."],
        "medium": ["صباح الخير، عايز", "عشرة كيلو طماطم للمطعم."],
        "high": ["صباح الخير، عايز عشرة كيلو طماطم للمطعم."],
    },
    "korean": {
        # Korean moves "for the restaurant" before the object, to show SOV reordering.
        "full": "안녕하세요, 식당에 쓸 토마토 십 킬로그램이 필요해요.",
        "low": ["안녕하세요,", "식당에 쓸", "토마토 십 킬로그램이", "필요해요."],
        "medium": ["안녕하세요, 식당에 쓸", "토마토 십 킬로그램이 필요해요."],
        "high": ["안녕하세요, 식당에 쓸 토마토 십 킬로그램이 필요해요."],
    },
}

# The JSON shape the model must answer with: all three latencies in one object.
_SCHEMA_LINE = ('{"low_latency": {"source_chunks": ["...", "..."], "target_chunks": ["...", "..."]}, '
                '"medium_latency": {"source_chunks": ["...", "..."], "target_chunks": ["...", "..."]}, '
                '"high_latency": {"source_chunks": ["...", "..."], "target_chunks": ["...", "..."]}}')

# Placeholders for the sentence pair. They are filled in with str.replace, not str.format, so the
# braces in the JSON example are left alone.
SRC_SENTINEL, TGT_SENTINEL = "__SOURCE__", "__TARGET__"


def _example_json(language):
    """Render the worked example as the JSON the model should imitate, for all three latencies."""
    tgt = _EX_TGT[language]
    obj = {
        "low_latency": {"source_chunks": _EX_EN["low"], "target_chunks": tgt["low"]},
        "medium_latency": {"source_chunks": _EX_EN["medium"], "target_chunks": tgt["medium"]},
        "high_latency": {"source_chunks": _EX_EN["high"], "target_chunks": tgt["high"]},
    }
    return json.dumps(obj, ensure_ascii=False, indent=2)


def get_template(method, language):
    """Return the prompt with everything baked in except SRC_SENTINEL / TGT_SENTINEL."""
    assert method in ("generic", "specific"), method
    assert language in LANGUAGES, language
    tgt_lang = TGT_LANG_NAME[language]
    guidance = (GUIDANCE[language] + "\n\n") if method == "specific" else ""
    ex_tgt = _EX_TGT[language]["full"]
    ex_json = _example_json(language)
    return f"""You are helping prepare training data for a simultaneous interpreter.
{guidance}Given an English sentence and its {tgt_lang} translation, split BOTH sentences into aligned chunks at THREE different granularities, one per latency level:
- low latency: fragment into brief, coherent phrases that each convey one complete thought -- the MOST chunks
- medium latency: longer chunks, roughly clause-length -- fewer chunks than low latency
- high latency: the longest chunks, covering full clauses or the whole sentence -- the FEWEST chunks (often just one)

For EACH of the three granularities independently:
- each chunk is a point where a real-time interpreter could plausibly start translating without waiting for more of the source sentence
- concatenating the chunks in order EXACTLY reconstructs each original sentence, character for character -- copy substrings verbatim, INCLUDING every comma, period, and space; do NOT paraphrase, correct, translate, add, remove, or reorder anything, and do NOT fix spelling, spacing, or grammar even if the text looks wrong
- never output an empty chunk, and never drop the final punctuation
- the English chunks and the {tgt_lang} chunks must have the SAME NUMBER of chunks, in the same order (chunk i of the source aligns with chunk i of the target)
- the chunk COUNT must not increase from low to medium to high latency (low >= medium >= high)

Example:
English: {_EX_EN_FULL}
{tgt_lang}: {ex_tgt}
{ex_json}

Now do the same for:
English: {SRC_SENTINEL}
{tgt_lang}: {TGT_SENTINEL}

Respond with ONLY a JSON object of this exact form, no other text:
{_SCHEMA_LINE}"""


def build_prompt(method, language, source, target):
    """Final prompt string ready to send to the model."""
    return get_template(method, language).replace(SRC_SENTINEL, source).replace(TGT_SENTINEL, target)


# --- Single-latency prompt, used by the --retry_fallbacks pass ---------------------------------------
# Same rules and worked example as above, but one latency per call, so the answer is much shorter.

_LAT_DESC = {
    "low": "brief, coherent phrases that each convey one complete thought (the FINEST granularity -- the MOST chunks)",
    "medium": "roughly clause-length segments (MEDIUM granularity -- fewer, larger chunks)",
    "high": "the longest chunks, covering full clauses or the whole sentence (the COARSEST -- often just one chunk)",
}
_SCHEMA_SINGLE = '{"source_chunks": ["...", "..."], "target_chunks": ["...", "..."]}'


def _example_json_single(language, latency):
    """Same worked example as _example_json, cut down to one latency."""
    obj = {"source_chunks": _EX_EN[latency], "target_chunks": _EX_TGT[language][latency]}
    return json.dumps(obj, ensure_ascii=False, indent=2)


def get_template_single(method, language, latency):
    """Return the single-latency prompt with everything baked in except the two sentinels."""
    assert method in ("generic", "specific"), method
    assert language in LANGUAGES, language
    assert latency in ("low", "medium", "high"), latency
    tgt_lang = TGT_LANG_NAME[language]
    guidance = (GUIDANCE[language] + "\n\n") if method == "specific" else ""
    ex_tgt_full = _EX_TGT[language]["full"]
    ex_json = _example_json_single(language, latency)
    return f"""You are helping prepare training data for a simultaneous interpreter.
{guidance}Given an English sentence and its {tgt_lang} translation, split BOTH into aligned chunks for {latency.upper()} latency: {_LAT_DESC[latency]}.

Rules:
- each chunk is a point where a real-time interpreter could plausibly start translating without waiting for more of the source sentence
- concatenating the chunks in order EXACTLY reconstructs each original sentence, character for character -- copy substrings verbatim, INCLUDING every comma, period, and space; do NOT paraphrase, correct, translate, add, remove, or reorder anything, and do NOT fix spelling, spacing, or grammar even if the text looks wrong
- never output an empty chunk, and never drop the final punctuation
- the English chunks and the {tgt_lang} chunks must have the SAME NUMBER of chunks, in the same order (chunk i of the source aligns with chunk i of the target)

Example ({latency} latency):
English: {_EX_EN_FULL}
{tgt_lang}: {ex_tgt_full}
{ex_json}

Now do the same for:
English: {SRC_SENTINEL}
{tgt_lang}: {TGT_SENTINEL}

Respond with ONLY a JSON object of this exact form, no other text:
{_SCHEMA_SINGLE}"""


def build_prompt_single(method, language, source, target, latency):
    """Single-latency prompt string ready to send to the model."""
    return (get_template_single(method, language, latency)
            .replace(SRC_SENTINEL, source).replace(TGT_SENTINEL, target))
