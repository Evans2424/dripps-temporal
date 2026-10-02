# The DRIPPS annotation scheme

Reference for the linguistic categories the models consume. Source: Silvano,
Cordeiro, Leal & Pais, *DRIPPS: a Corpus with Discourse Relations in Perfect
Participial Sentences*, LDK 2023 (ACL Anthology `2023.ldk-1.51`), plus the label
inventory observed in the released export.

## The construction

An **adverbial perfect participial clause** (APC; in the Portuguese grammatical
tradition, *oração gerundiva adverbial com gerúndio composto*) is a subordinate
clause formed with the auxiliary **`ter` in the gerund (`tendo`)** — or English
**`having`** — followed by a past participle:

> *A PSP anunciou a detenção de sete pessoas, **tendo sido apreendidas** mais de
> quatro mil doses de droga.* (EP)
>
> ***Having served his country**, he became a great believer in the need for change.* (BE)

The construction is **almost always devoid of a discourse connector** (95.1% of
the corpus has none), so the temporal and discourse reading has to be inferred
from other signals. That is what makes it a good probe for cue hierarchies: the
inference *must* run through tense, aspect and position.

## Fields

| Field | Meaning | Role here |
|---|---|---|
| `ID` | `<variety><n>`; prefix encodes the variety | grouping |
| `DR` | Discourse relation, ISO 24617-8 DR-core | **excluded — leaky** |
| `SR-SC` | Semantic role of the subordinate clause | **excluded — leaky** |
| `CNT` | Connector, lexical item or empty | cue |
| `Position` | Position of SC relative to MC | cue |
| `TR` | **Temporal relation between SC and MC** | **target** |
| `TMC` | Tense of the main clause | cue |
| `ATMC` | Aspectual class of the main clause | cue |
| `ATSC` | Aspectual class of the subordinate clause | cue |
| `Sentence` | Raw sentence | neural models only |

MC = main clause, SC = subordinate (participial) clause.

## Target: `TR`

- **`Ant`** (anteriority) — the SC situation precedes the MC situation.
- **`Post`** (posteriority) — the SC situation follows the MC situation.
- **`Simul`** (simultaneity) — the two overlap.

Note the participle is morphologically *perfect*, which in the descriptive
literature is associated with anteriority (Lobo 2003, for initial clauses). The
corpus shows this is only true for English: the Portuguese varieties are
predominantly **posterior**, which is the phenomenon the paper is about.

## Why `DR` and `SR-SC` are excluded

They are not merely correlated with `TR`; they are partly **defined** by it.
`SR-SC` takes the values `before` / `after` for the `asynchrony` relation, which
*is* temporal order — in the corpus, `before` is anterior in 100% of cases and
`after` posterior in 100% of cases. Likewise `DR == "asynchrony"` never
co-occurs with `Simul`; `elaboration` (57/57), `exemplification` (16/16) and
`synchrony` (7/7) are always `Simul`, and `conjunction` (21/23) and `manner` (5/6) nearly so.

A model given these columns reaches macro-F1 0.750 while explaining nothing
(fitted on the 793 Portuguese rows that have `SR-SC`; the honest cues score 0.736 on the same rows).
That number is reported once, as an explicit circular upper bound, and the
columns are otherwise blocked by `dripps.leakage.assert_no_leakage`.

For reference, the ISO 24617-8 relations and their argument roles:

| Relation | Arg1 role | Arg2 role | |
|---|---|---|---|
| Cause | result | reason | asymmetric |
| Expansion | narrative | expander | asymmetric |
| Asynchrony | before | after | asymmetric |
| Concession | expectation-raiser | expectation-denier | asymmetric |
| Elaboration | broad | specific | asymmetric |
| Exemplification | set | instance | asymmetric |
| Manner | achievement | means | asymmetric |
| Conjunction / Contrast / Synchrony | — | — | symmetric (`SYM`) |

## Aspectual classes → Moens & Steedman (1988)

`ATMC` and `ATSC` use the event-nucleus taxonomy. Each class is a unique
combination of three primitives, so the model uses the primitives directly —
fewer parameters, and it states the hypothesis under test in its own terms.

| Label | Class | dynamic | durative | telic | Example |
|---|---|---|---|---|---|
| `St` | State | − | + | − | *ser presidente* |
| `Pro` | Process | + | + | − | *correr* |
| `Culm` | Culmination | + | − | + | *chegar* |
| `CP` | Culminated process | + | + | + | *construir uma casa* |
| `Pon` | Point | + | − | − | *tossir* |

`Pon` is attested **twice**, both in MP. `ATSC` never takes it.

**The hypothesis under test** (Silvano et al. 2021): anteriority and posteriority
readings relate to **telic** situations, simultaneity to **durative** ones, *in
both clauses*. The last clause is an interaction claim, so `both_telic` and
`both_durative` are built explicitly rather than left for a model to find.

## Tense → TAM bundles

`TMC` has **33 distinct labels**, and the Portuguese and English inventories are
nearly disjoint (`PP`/`Pres-Ind`/`PIMP-Ind`… vs `Pst`/`Pres`/`PresP`…). A
33-level categorical is neither estimable at n≈200–350 per variety nor comparable
across languages, so each label maps to six orthogonal dimensions:

`tense` (past / present / future / nonfinite, *past* is the reference level),
`perfective`, `progressive`, `perfect`, `finite`, `irrealis`.

Selected Portuguese labels:

| Label | Portuguese name | Bundle |
|---|---|---|
| `PP` | pretérito perfeito simples | past, perfective |
| `PIMP-Ind` | pretérito imperfeito | past |
| `PMP` | pretérito mais-que-perfeito | past, perfective, perfect |
| `PPC-Ind` | pretérito perfeito composto | present, perfect |
| `Pres-Ind` | presente do indicativo | present |
| `Pres-Conj` | presente do conjuntivo | present, irrealis |
| `Cond` | condicional | future, irrealis |
| `PresPro` / `PresPro-G` | *está a fazer* (EP) / *está fazendo* (BP) | present, progressive |
| `ir(Pres)+INF-S` | *vai fazer* | future |
| `INF-C` | infinitivo composto | nonfinite, perfect |
| `GS` / `Ger-S` | gerúndio simples | nonfinite, progressive |

**`PPC-Ind` deserves care.** European Portuguese's *pretérito perfeito composto*
(*tenho feito*) is **not** the English present perfect: it denotes an iterative
or durative eventuality extending to the utterance time. It is therefore mapped
as `present + perfect`, not as a past. The prediction is that it should force a
simultaneity reading — and in EP all 6 occurrences are `Simul`.

English labels map straightforwardly: `Pst` → past·perfective, `Pres` → present,
`PresP` → present·perfect, `PstP` → past·perfective·perfect, `PresCont` →
present·progressive, `PstCont` → past·progressive, and the continuous perfects
combine `perfect` with `progressive`.

`Pres`, `Fut`, `Cond` and `GS` are the only labels shared by both languages; the
bundle resolves them identically, so the ambiguity is harmless.

## Position

`Initial` / `Medial` / `Final`, modelled as `final` vs `non-final` (86.4% of the
corpus is final). Lobo (2003) proposed that initial APCs carry anteriority and
final ones posteriority; Silvano et al. (2021) report the corpus does not
support a clean split. The models here find non-final position to be a real but
third-ranked cue, pointing toward anteriority.

## Connector

Present in only 4.4% of the corpus, and unevenly: **BP 17.6%** (almost all
*mesmo*) against 0.4–2.4% in EP/AP/MP and 5% in BE (*despite*, *after*, *although*).
Modelled as presence/absence, not as a lexical item. Conclusions about EP/AP/MP
connectors are "too rare to estimate here" (4, 6 and 1 clauses), never "connectors do not matter".

## Later annotation batches

### Second batch (Violeta)

150 Portuguese sentences (50 each EP, AP, MP; none for BP) arrived as `Dados_Violeta.xlsx`, one sheet per
variety, and are converted by `experiments/00_ingest.py` into `data/raw/dripps_violeta.csv`. IDs carry a `V`
(`PTEUV1`, `PTAOV1`, `PTMZV1`), so the batch is readable from the ID and cannot collide with the original export.
`io.load` exposes it as `batch == "violeta"`. None of the 150 sentences occurs in the original export, and none
contains more than one `tendo`, so each is one row.

What differs from the original export, and how it was handled:

- **No `SR-SC`.** The field is empty. It is never a model feature; the leakage statistics, the circular upper
  bound and the Leakage tab use only the 793 Portuguese rows that have it (the later batches have none).
- **`DR` capitalised** (`Asynchrony`); lower-cased on ingest.
- **`Futuro Perfeito`** (1 row) is the future perfect and is mapped to `Fut-C`.
- **`PstPro`** (1 row, MZ: *estavam a usar*) is a new tense label, coded past · progressive, the same bundle as
  the English `PstCont`. This is a coding decision made at ingest, not one the annotator stated.
- **Almost all final** (141 of 150) and almost no connectors (6, all *mesmo*), like the original EP, AP and MP.
- **No agreement figure.** Nothing in the workbook says how reliable these annotations are against the original
  ones, so pooled results should be read with that in mind.

### Third batch (Abergaria)

95 European Portuguese sentences from `DADOS_João_Abergaria_PE.xlsx` (one sheet, newswire from five
newspapers) are converted by the same `experiments/00_ingest.py` into `data/raw/dripps_abergaria.csv`. IDs carry a
`J` (`PTEUJ1`…`PTEUJ95`) and `io.load` exposes `batch == "abergaria"`. The sheet had 98 rows; three were dropped
and are reported by the script:

- **Two duplicates of the Violeta EP batch** (`EX11` = `PTEUV21`, `PUB03` = `PTEUV15`). The overlap is the only
  place two annotators coded the same sentence: `PUB03` agrees on every column, `EX11` differs on `TR`
  (`Ant` here, `Simul` in the Violeta batch). Two sentences are far too few for an agreement figure.
- **One row repeated inside the sheet** (`OBS07`, identical annotation).

A cross-batch test asserts no sentence appears in two batches.

What differs from the other files, and how it was handled:

- **Different column layout and tense spellings** (`Pres - Ind`, `PIMP - Ind`, …), normalised at ingest. Judgement
  calls, made without the annotator: `PMQP C - Ind` → `PMP` (3 rows), `Inf - opor` → `INF-S`, `quer (Pres)` → `Pres`
  (modal main verb), passive `ser (PP) + Part. P` → `PP`, and a bare `Part. P` (1 row, a participial main clause)
  → `PartP`, a new non-finite bundle.
- **No `DR` and no `SR-SC`;** both are left empty and never reach a model.
- **Connector "N/A" on every row,** read as *no connector*. If it means "not annotated", connector estimates are
  biased toward zero for these rows.
- **Position** has more non-final clauses (12 of 95) than the Violeta batch, and includes the only two non-final
  `Post` rows in Portuguese (`PTEUJ17`, `PTEUJ90`).
- **No agreement figure** (see above).
