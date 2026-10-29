"""The project summary shown in the viewer and the app: what was run, what it showed, what is running, what is next.

The scores, intervals and contrasts are read from results/tables, so they follow a re-run. The wording is
hand-written and some counts quoted in it (0 of 47 initial Post, 26 of 33 medial Ant) come from docs/LINGUISTICS.md,
so re-read both when tables change. Edit ``summary`` when a running experiment finishes.
"""

from __future__ import annotations

from . import results


#: One caveat for both pages; the encoder sentence must follow results/tables/t1_encoder.csv.
NOTE = ("The cue hierarchy and the model ladder are the firm results. Text alone sits below the cues-only model: plain "
        "xlm-roberta-base and plain BERTimbau have paired intervals against B3 that exclude 0. Encoders that start from "
        "the cues or see them only match B3, and no model has been shown to beat it or to add to the cues. Annotator "
        "agreement is not available for the later batches.")


def summary(n_rows: int) -> dict:
    """Rows for the three tables: ``done`` and ``running`` (name, setup, result, conclusion, direction), ``planned`` (name, setup, direction)."""
    f = lambda x, d=3: f"{x:.{d}f}".replace("-0." + "0" * d, "0." + "0" * d)  # noqa: E731  (no "-0.000")
    base = results.csv("baselines.csv").set_index("model")
    cue = results.csv("cue_importance.csv").set_index("block")
    b6 = results.csv("b6_tests.csv").set_index("term")
    t1 = results.csv("t1_encoder.csv").set_index("model")
    t3 = results.csv("t3_ablation.csv").set_index("perturbation")
    iv = lambda r, k="macro_f1": f"{f(r[k])} [{f(r.ci_lo)}, {f(r.ci_hi)}]"  # noqa: E731
    def need(k):  # a t1_encoder.csv written on a machine without every cache lacks rows: say how to get them
        if k not in t1.index:
            raise results.MissingTable(f"t1_encoder.csv has no row {k!r} -- run `{results.PRODUCED_BY['t1_encoder.csv']}` "
                                       "(or experiments/09_merge_encoder.py on the lab tables) first")
        return t1.loc[k]

    enc = {v: need(f"encoder {v}") for v in ("plain", "marked", "plain_feat", "marked_feat")}
    d = need("encoder marked_feat - B3 logit (paired)")
    dm = need("encoder marked_feat - encoder marked (paired)")
    verb = t3.loc["mask_main_verb - ctrl_main_verb (d_P_true)"]
    part = t3.loc["mask_participle - ctrl_participle (d_P_true)"]
    move = t3.loc["move_apc_initial"]
    done = [
        ("Corpus and leakage control",
         f"{n_rows} Portuguese clauses (EP, BP, AP, MP); 3 relations (Ant, Post, Simul). Discourse relation and semantic role "
         "excluded as definitional.",
         f"Using them gives macro-F1 {f(base.loc['circular (DR+SR-SC)'].macro_f1)}, against "
         f"{f(base.loc['circular control: B3 on same rows'].macro_f1)} for the cues on the same rows.",
         "The annotated cues are a legitimate, non-circular input.",
         f"The rhetorical relation adds little beyond aspect and tense (+{f(base.loc['circular (DR+SR-SC)'].macro_f1 - base.loc['circular control: B3 on same rows'].macro_f1)} macro-F1 on the rows that have it): the coded cues already carry most of what it would tell."),
        ("Baseline ladder B0 to B5 (5-fold, sentence-grouped, 10 repeats)",
         "Majority, one rule, additive logit (B3), tree, random forest, XGBoost.",
         f"Majority {f(base.loc['B0 majority'].macro_f1)}; one rule {f(base.loc['B1 one-rule'].macro_f1)}; "
         f"B3 {iv(base.loc['B3 logit'])}; forest {f(base.loc['B5 forest'].macro_f1)}; "
         f"XGBoost {f(base.loc['B5 xgboost'].macro_f1)}.",
         "An additive model matches the non-linear ones: no cue interactions to find.",
         "Cues combine additively. Telicity or durativity shared by both clauses is not needed to explain the data (interaction block ablation about 0), which puts the 'same configuration in both clauses' claim in doubt."),
        ("RQ1: cue hierarchy (refit ablation, group bootstrap)",
         "Drop in macro-F1 when a cue block is removed and the model refit.",
         f"Main-clause aspect {f(cue.loc['aspect_mc'].importance_ablation)}, tense {f(cue.loc['tense_mc'].importance_ablation)}, "
         f"position {f(cue.loc['position'].importance_ablation)}; subordinate aspect "
         f"{f(cue.loc['aspect_sc'].importance_ablation)} [{f(cue.loc['aspect_sc'].ci_lo_ablation)}, "
         f"{f(cue.loc['aspect_sc'].ci_hi_ablation)}].",
         "The reading is carried by the main clause, not by the participial clause.",
         "A completed (telic, perfective) main event followed by a final participial clause reads as narrative continuation (Post): the participle marks its own completion, not order. Initial clauses are never Post (0 of 47, 41 Ant); medial ones are mostly Ant (26 of 33)."),
        ("RQ2: variety differences (pooled hierarchical model, B6)",
         "Variety intercepts and cue slopes, tested by bootstrap likelihood-ratio.",
         f"Base rates differ (deviance drop {f(b6.loc['intercepts (base rates)'].dev_drop, 1)}, "
         f"p = {f(b6.loc['intercepts (base rates)'].p_boot)}); slopes: omnibus p = {f(b6.loc['slopes, all cues'].p_boot)}, "
         "no single block survives Holm correction.",
         "Varieties differ in base rates; different cue weights are not shown.",
         "Same cue grammar, different mix of contexts: EP, AP and MP lean Post, BP leans Ant, with more fronted clauses and connectors (mesmo). Contact accounts are speculative on newswire data."),
        ("T1a: encoder on the sentence text alone (xlm-roberta-base, tuned in-fold)",
         "Plain text, and text with the participial group and main verb marked. Same folds as B3.",
         f"B3 {iv(base.loc['B3 logit'])}. Plain {iv(enc['plain'])}; marked {iv(enc['marked'])}.",
         "Marking the clauses closes most of the gap to B3, but the text alone does not exceed it.",
         "The reading sits at the two verbs, where aspect and tense live: telling the model where they are recovers most of the lost signal."),
        ("T1b: the annotated cues concatenated with the text vector (_feat)",
         "The 17 cue columns join the encoder output before the classifier; cue learning rate tuned in-fold.",
         f"Plain + cues {iv(enc['plain_feat'])}; marked + cues {iv(enc['marked_feat'])}; "
         f"marked + cues minus B3 {f(d.macro_f1)} [{f(d.ci_lo)}, {f(d.ci_hi)}]; "
         f"minus marked alone {f(dm.macro_f1)} [{f(dm.ci_lo)}, {f(dm.ci_hi)}].",
         "Giving the encoder the cues does not measurably beat B3 or the text-only encoder; averaging with B3 at the pre-set weight (0.5) gives no gain either.",
         "Beyond the five coded cues, little is learnable at this sample size. Adverbials, lexical semantics and world knowledge are sparse in 1,038 rows, which is not evidence that they do not matter."),
        ("T3: masking and displacement (plain encoder)",
         "Mask participle, main verb or adverbials against a random-word control; move the participial clause to the front.",
         f"P(true reading) change, net of control: main verb {f(verb.d_P_true)} [{f(verb.d_P_true_lo)}, {f(verb.d_P_true_hi)}]; "
         f"participle {f(part.d_P_true)} [{f(part.d_P_true_lo)}, {f(part.d_P_true_hi)}]. "
         f"Moving the clause forward shifts Post to Ant (Ant {f(move.d_P_Ant, 2)}, Post {f(move.d_P_Post, 2)}).",
         "The encoder leans on the main verb more than on the participle, in line with RQ1. Displacement also changes a cue (position).",
         "A model reading raw text also weights the main verb above the participle and reproduces the fronting effect (Post to Ant). That converges with the annotated analysis, but position is itself a coded cue, so the shift is partly circular."),
    ]
    ci = lambda r: f"[{f(r.ci_lo)}, {f(r.ci_hi)}]"  # noqa: E731
    pv = lambda a, b=None: need(f"encoder {a} - " + (f"encoder {b}" if b else "B3 logit") + " (paired)")  # noqa: E731
    x = lambda v: need(f"encoder {v}")  # noqa: E731
    fuse = lambda v: need(f"B3 + encoder {v}, w=0.25 (post hoc) - B3 logit (paired)")  # noqa: E731
    done += [
        ("T1, round 3: other ways to use the cues (xlm-roberta-base)",
         "Cues as a residual on B3's log-odds (_res), as Portuguese words before the sentence (_txt), or as an auxiliary "
         "prediction target (_aux); each plain and marked. Exploratory; no multiplicity correction.",
         f"Residual: plain {f(x('plain_res').macro_f1)}, marked {f(x('marked_res').macro_f1)}; auxiliary: plain "
         f"{f(x('plain_aux').macro_f1)}, marked {f(x('marked_aux').macro_f1)}; words: plain {f(x('plain_txt').macro_f1)}, "
         f"marked {f(x('marked_txt').macro_f1)}. Plain residual minus B3 {f(pv('plain_res').macro_f1)} {ci(pv('plain_res'))}; "
         f"marked auxiliary minus B3 {f(pv('marked_aux').macro_f1)} {ci(pv('marked_aux'))}.",
         "No way of using the cues makes the text beat B3. Residual and auxiliary versions match it; cues as words is the "
         f"weakest, with plain auxiliary, and least stable (seed sd {f(x('plain_txt').sd_seed)}). Starting from B3 also removes the instability of "
         f"plain text (seed sd {f(x('plain').sd_seed)} to {f(x('plain_res').sd_seed)}).",
         "The text does not add what the coded labels leave out: asked to start from the cue model, the encoder returns its "
         "answer. A light post hoc average with B3 (w = 0.25) gains 0.01 to 0.02 for every encoder, including BERTimbau, which "
         f"is below B3 alone (marked auxiliary {f(fuse('marked_aux').macro_f1)} {ci(fuse('marked_aux'))}; plain BERTimbau "
         f"{f(fuse('plain@bertimbau').macro_f1)} {ci(fuse('plain@bertimbau'))}). That looks like the ordinary gain from blending "
         "two partly independent predictors, not something specific to the cues; not pre-set, not corrected for the number of variants."),
        ("T1, round 3: Portuguese checkpoint (BERTimbau)",
         "Plain and marked on BERTimbau, tuned in-fold, same folds; compared with xlm-roberta-base by paired bootstrap.",
         f"Plain {iv(x('plain@bertimbau'))}; marked {iv(x('marked@bertimbau'))}. Minus B3: plain "
         f"{f(pv('plain@bertimbau').macro_f1)} {ci(pv('plain@bertimbau'))}, marked {f(pv('marked@bertimbau').macro_f1)} "
         f"{ci(pv('marked@bertimbau'))}. Minus xlm-roberta-base: plain {f(pv('plain@bertimbau', 'plain').macro_f1)} "
         f"{ci(pv('plain@bertimbau', 'plain'))}, marked {f(pv('marked@bertimbau', 'marked').macro_f1)} "
         f"{ci(pv('marked@bertimbau', 'marked'))}.",
         "A Portuguese checkpoint does not close the gap: plain is below B3 (interval excludes 0), marked is about as far below with an interval that includes 0, and neither is above xlm-roberta-base. "
         "It is steadier than plain xlm-roberta-base across seeds, and marking the verbs adds little "
         f"({f(pv('marked@bertimbau', 'plain@bertimbau').macro_f1)}, against {f(pv('marked', 'plain').macro_f1)} for xlm-roberta-base).",
         "Portuguese pretraining alone does not recover what the cues give, so the gap is not simply a matter of the model "
         "knowing Portuguese better. Caveat: BERTimbau is pretrained on Brazilian text and most of the corpus is "
         "European, so Albertina (PT-PT) is the fairer test."),
    ]
    running = [
        ("T1, round 3: Albertina (PT-PT) and mDeBERTa",
         "Plain and marked, tuned in-fold, same folds. DeBERTa models run in fp32 (bf16 attention masking overflows).",
         "Partly run; tables not final.",
         "Does a European-Portuguese model, or a different architecture, close the gap?",
         "A gain for Albertina only would point to European-Portuguese usage; none would leave annotation size and the "
         "ceiling of the five cues as the explanation."),
    ]
    planned = [
        ("Larger encoders", "XLM-R large, BERTimbau large, Albertina 900M, only if Albertina or mDeBERTa show a gain (BERTimbau did not).",
         "Is capacity, not pretraining language, the limit for text-only models?"),
        ("Generative models", "4 to 5 open-weight LLMs (7B to 70B), zero-shot, few-shot, cues in the prompt, LoRA; same folds and perturbations.",
         "Do models with broad world knowledge use lexical event cues (adverbials such as de seguida) that the inventory leaves out?"),
        ("Decision-making models", "Two open reproductions of a Jev-style mechanism, scored on the same task.",
         "Does an explicit decision process weight main-clause cues the way the corpus does?"),
        ("Comparison with theory", "Test the aspect and tense predictions per model family; agreement between families; variety question.",
         "Which claims survive across families: telicity to Ant or Post, durativity to Simul, main-clause dominance?"),
        ("Validation", "Hand-check the span sample; fresh annotated batch as a held-out confirmation; annotator agreement.",
         "Are the later batches' lower scores and the two medial Post cases a matter of annotation or of the language?"),
        ("Paper", "Computational linguistics venue; code and tables released.",
         "State the position and main-clause accounts as hypotheses: the numbers support the pattern, not the explanation."),
    ]
    return {"note": NOTE, "done": [dict(zip(("name", "setup", "result", "conclusion", "direction"), r)) for r in done],
            "running": [dict(zip(("name", "setup", "result", "conclusion", "direction"), r)) for r in running],
            "planned": [dict(zip(("name", "setup", "direction"), r)) for r in planned]}
