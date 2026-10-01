"""Canonical schema for the DRIPPS corpus.

Single source of truth for column names, label inventories, and the two
linguistic mappings the analysis depends on:

  * aspectual class -> Moens & Steedman (1988) event-nucleus primitives
  * tense label     -> language-neutral TAM bundle

Every label attested in the released corpus is listed explicitly. Unknown
labels are an error, not a silently-dropped row: see ``io.load``.
"""

from __future__ import annotations

# --- columns -----------------------------------------------------------------

#: Single source of truth: models and fold assignment must not desynchronise.
SEED = 20260920

ID = "ID"
SENTENCE = "Sentence"
TARGET = "TR"

#: Annotated cue columns that may legitimately feed a TR model.
CUE_COLUMNS = ("Position", "CNT", "TMC", "ATMC", "ATSC")

#: Columns that must NEVER feed a TR model -- see ``leakage.py``.
#: ``SR-SC`` encodes temporal order definitionally (``before``/``after``);
#: ``DR`` is partly defined by it (``asynchrony`` never yields ``Simul``).
LEAKY_COLUMNS = ("DR", "SR-SC")

RAW_COLUMNS = (ID, "DR", "SR-SC", "CNT", "Position", TARGET, "TMC", "ATMC", "ATSC", SENTENCE)

#: ISO 24617-8 DR-core relations attested in the corpus. Validated even though
#: they are excluded from modelling: a re-export with a new or typo'd label must
#: fail loudly rather than quietly change the circular upper bound.
DR_LABELS = (
    "asynchrony", "cause", "expansion", "concession", "elaboration",
    "conjunction", "exemplification", "synchrony", "contrast", "manner",
)

#: Literal string the exporter writes for an absent semantic role.
NULL_TOKEN = "null"

#: Semantic role of the subordinate clause. ``SYM`` marks the symmetric relations.
SR_LABELS = (
    "after", "reason", "expander", "result", "before", "e-raiser",
    "specific", "SYM", "instance", NULL_TOKEN, "narrative", "means",
)

# --- varieties ---------------------------------------------------------------

#: ID prefix -> (variety code, language, is_portuguese)
VARIETY_BY_PREFIX = {
    "PTEU": ("EP", "pt", True),
    "PTBR": ("BP", "pt", True),
    "PTAO": ("AP", "pt", True),
    "PTMZ": ("MP", "pt", True),
    # second batch (Dados_Violeta.xlsx): same varieties, "V" marks the batch
    "PTEUV": ("EP", "pt", True),
    "PTAOV": ("AP", "pt", True),
    "PTMZV": ("MP", "pt", True),
    "ENBE": ("BE", "en", False),
}

#: The four Portuguese varieties, in the order used in tables and figures.
PT_VARIETIES = ("EP", "BP", "AP", "MP")

#: British English is a contrastive reference, not a fifth data point: it is
#: 95% anterior, so its cue weights are unidentifiable (see docs/DATA_AUDIT.md).
REFERENCE_VARIETY = "BE"

# --- target ------------------------------------------------------------------

TR_LABELS = ("Ant", "Post", "Simul")

# --- aspectual classes -> Moens & Steedman (1988) primitives -----------------

POSITION_LABELS = ("Initial", "Medial", "Final")

#: class -> (dynamic, durative, telic). ``Pon`` (Point) is attested twice, MP only.
ASPECT_PRIMITIVES = {
    "St":   (False, True,  False),   # State
    "Pro":  (True,  True,  False),   # Process
    "Culm": (True,  False, True),    # Culmination
    "CP":   (True,  True,  True),    # Culminated process
    "Pon":  (True,  False, False),   # Point
}

ASPECT_LABELS = tuple(ASPECT_PRIMITIVES)

# --- tense -> TAM bundle -----------------------------------------------------

#: Six orthogonal, language-neutral dimensions. A 31-level categorical is not
#: estimable at n~200/variety and does not transfer across languages (the EN and
#: PT label sets are disjoint); these six are both estimable and comparable.
TAM_FIELDS = ("tense", "perfective", "progressive", "perfect", "finite", "irrealis")

TENSE_VALUES = ("past", "present", "future", "nonfinite")


def _tam(tense, perfective=False, progressive=False, perfect=False, finite=True, irrealis=False):
    return dict(zip(TAM_FIELDS, (tense, perfective, progressive, perfect, finite, irrealis)))


#: Every ``TMC`` value attested in the corpus. Portuguese labels follow the
#: Portuguese grammatical tradition; English labels the standard descriptive set.
TAM_BUNDLE = {
    # -- Portuguese: indicative -----------------------------------------------
    "PP":             _tam("past",    perfective=True),                 # pretérito perfeito simples
    "PIMP-Ind":       _tam("past"),                                     # pretérito imperfeito
    "PMP":            _tam("past",    perfective=True, perfect=True),   # mais-que-perfeito
    "PPC-Ind":        _tam("present", perfect=True),                    # pretérito perfeito composto
    "Pres-Ind":       _tam("present"),                                  # presente do indicativo
    "Pres":           _tam("present"),                                  # presente / EN simple present
    "Fut":            _tam("future"),                                   # futuro / EN future
    "Fut-C":          _tam("future",  perfect=True),                    # futuro composto
    # -- Portuguese: irrealis --------------------------------------------------
    "Pres-Conj":      _tam("present", irrealis=True),
    "PIMP-Conj":      _tam("past",    irrealis=True),
    "PPC-Conj":       _tam("present", perfect=True, irrealis=True),
    "Cond":           _tam("future",  irrealis=True),                   # condicional / EN would
    "Cond-C":         _tam("future",  perfect=True, irrealis=True),
    # -- Portuguese: progressive periphrases -----------------------------------
    "PresPro":        _tam("present", progressive=True),                # está a fazer (EP)
    "PresPro-G":      _tam("present", progressive=True),                # está fazendo (BP)
    "PstPro":         _tam("past",    progressive=True),                # estava a fazer (Violeta batch)
    "ir(Pres)+Ger":   _tam("present", progressive=True),                # vai fazendo
    "ir(PP)+Ger":     _tam("past",    perfective=True, progressive=True),
    # -- Portuguese: prospective periphrases -----------------------------------
    "ir(Pres)+INF-S": _tam("future"),                                   # vai fazer
    "ir(Fut)+INF-S":  _tam("future"),                                   # irá fazer
    # -- Portuguese: non-finite ------------------------------------------------
    "INF-S":          _tam("nonfinite", finite=False),                  # infinitivo simples
    "INF-C":          _tam("nonfinite", finite=False, perfect=True),    # infinitivo composto
    "GS":             _tam("nonfinite", finite=False, progressive=True),  # gerúndio simples
    "Ger-S":          _tam("nonfinite", finite=False, progressive=True),  # gerúndio simples (BP label)
    # -- English ---------------------------------------------------------------
    "Pst":            _tam("past",    perfective=True),
    "PstImp":         _tam("past"),
    "PresP":          _tam("present", perfect=True),
    "PstP":           _tam("past",    perfective=True, perfect=True),
    "PresCont":       _tam("present", progressive=True),
    "PstCont":        _tam("past",    progressive=True),
    "PresP-Cont":     _tam("present", perfect=True, progressive=True),
    "PstP-Cont":      _tam("past",    perfect=True, progressive=True),
}

TMC_LABELS = tuple(TAM_BUNDLE)

# --- connector ---------------------------------------------------------------

#: Attested connectors. Near-absent outside BP (17.6%) and BE (5%), so the
#: modelled feature is presence/absence, not the lexical item.
CONNECTOR_LABELS = ("mesmo", "porém", "despite", "after", "although")

#: Auxiliary introducing the perfect participial clause, by language. The APC
#: span is not annotated in the corpus, so it is recovered from this token.
#: Defined once here because a duplicated copy of this pattern previously went
#: stale in the audit script while the test kept passing.
APC_AUXILIARY = {"pt": r"\btendo\b", "en": r"\bhaving\b"}
