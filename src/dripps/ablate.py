"""Text perturbations for the encoder's causal ablation (T3).

The APC span is not annotated, so it is recovered from the auxiliary ``tendo``
and the participle that follows it. Every function returns ``None`` when the
span is ambiguous or the perturbation does not apply, and the caller drops that
row rather than guessing. Pure Python: the main finite verb is found by the
caller with a POS tagger and passed in as a character span.
"""

from __future__ import annotations

import re

from .schema import APC_AUXILIARY

Span = tuple[int, int]

_TENDO = re.compile(APC_AUXILIARY["pt"], re.I)
_TOKEN = re.compile(r"\w+|[^\w\s]")
_PARTICIPLE = re.compile(
    r"(?:\w*(?:ad|id|íd)[oa]s?|(?:feit|dit|post|vist|abert|escrit|cobert|tid|pag|ganh|gast|mort"
    r"|pres|eleit|aceit|expuls|salv|prescrit|subscrit|vind)[oa]s?)$", re.I)
# may stand between "tendo" and its participle; anything else there (a subject, an object) means
# the span is ambiguous and the row is dropped rather than guessed
_BETWEEN = {"sido", "estado", "já", "ainda", "sempre", "também", "nunca", "mal", "bem", "só",
            "apenas", "logo", "recentemente", "anteriormente", "posteriormente", "actualmente",
            "atualmente", "inicialmente", "mesmo", "novamente", "entretanto", "então", "até", "depois",
            "aliás", "inclusive", "agora"}

# A closed lexicon of explicit temporal adverbials; deliberately small and checkable.
_ADVERBIAL = re.compile(
    r"\b(?:\d+\s+(?:dias|semanas|meses|anos)\s+(?:depois|antes)|há\s+\d+\s+\w+|"
    r"depois|antes|logo|já|ainda|agora|hoje|ontem|amanhã|então|entretanto|seguida|"
    r"durante|enquanto|após|anteriormente|posteriormente|recentemente|actualmente|atualmente|"
    r"no\s+ano\s+passado|na\s+semana\s+passada|desde)\b", re.I)


def tendo_spans(text: str) -> list[Span]:
    return [m.span() for m in _TENDO.finditer(text)]


def apc_span(text: str) -> tuple[Span, Span] | None:
    """((start, end) of the APC verb group, (start, end) of the participle), or None."""
    tendo = list(_TENDO.finditer(text))
    if len(tendo) != 1:
        return None
    light = None  # copular "tendo sido campeão": the light verb itself is the participle
    for i, m in enumerate(_TOKEN.finditer(text, tendo[0].end())):
        w = m.group().lower()
        if i == 4 or not w.isalnum():
            break
        if w in ("sido", "estado") and light is None:
            light = m
        if w in _BETWEEN:
            continue
        if _PARTICIPLE.match(w):
            return (tendo[0].start(), m.end()), m.span()
        break
    return ((tendo[0].start(), light.end()), light.span()) if light else None


def mask_spans(text: str, spans: list[Span], mask: str) -> str:
    for s, e in sorted(spans, reverse=True):
        text = text[:s] + mask + text[e:]
    return text


def main_verb_span(doc, apc: tuple[Span, Span]) -> Span | None:
    """Finite verb of the root clause, from a spaCy ``Doc``; None if it sits inside the APC."""
    roots = [t for t in doc if t.dep_ == "ROOT"]
    if len(roots) != 1:
        return None
    root = roots[0]
    cands = [root] + [c for c in root.children if c.dep_ in ("aux", "aux:pass", "cop")]
    fin = [t for t in cands if "Fin" in t.morph.get("VerbForm")]
    if len(fin) != 1:
        return None
    t = fin[0]
    span = (t.idx, t.idx + len(t))
    if apc[0][0] <= span[0] < apc[0][1]:
        return None
    return span


def adverbial_spans(text: str) -> list[Span]:
    return [m.span() for m in _ADVERBIAL.finditer(text)]


def _balanced(s: str) -> bool:
    return s.count("(") == s.count(")") and sum(s.count(c) for c in '"“”«»') % 2 == 0


def move_apc_initial(text: str, lower_first: bool = False) -> str | None:
    """'Main, tendo X.' -> 'Tendo X, main.' for a final APC with no internal comma.

    ``lower_first`` lower-cases the main clause's first letter (the caller says
    when that word is not a proper noun), so the move does not also change casing.
    Returns None if a quote or bracket would be split across the two clauses.
    """
    tendo = list(_TENDO.finditer(text))
    if len(tendo) != 1 or not text[: tendo[0].start()].rstrip().endswith(","):
        return None
    tail = re.search(r"[.!?…]*\s*$", text).group()
    clause = text[tendo[0].start(): len(text) - len(tail)]
    main = text[: tendo[0].start()].rstrip().rstrip(",").rstrip()
    if "," in clause or not (_balanced(clause) and _balanced(main)):
        return None
    if lower_first:
        main = main[:1].lower() + main[1:]
    return f"{clause[0].upper()}{clause[1:]}, {main}{tail}"


def mask_random(text: str, forbidden: list[Span], k: int, mask: str, rng) -> str | None:
    """Mask ``k`` random words outside ``forbidden``: the control for 'any masking hurts'."""
    words = [m.span() for m in re.finditer(r"\w+", text)
             if not any(s <= m.start() < e for s, e in forbidden)]
    if len(words) < k:
        return None
    pick = rng.choice(len(words), size=k, replace=False)
    return mask_spans(text, [words[i] for i in pick], mask)
