"""Model architectures, one tab each: a diagram, the formal specification, the settings used and the references.

Settings are copied from src/dripps/models.py, encoder.py, decision.py and experiments/07_hierarchical.py; check them
there if a model changes. Diagrams are graphviz, in the same style as the B4 tree page."""

import streamlit as st

import data  # noqa: F401  (puts src/ on sys.path; this view can be imported before any other)
from dripps import decision

HEAD = ('digraph G {{ rankdir={rd}; nodesep=0.35; ranksep=0.45; bgcolor="transparent";'
        'node [shape=box, style="rounded,filled", fillcolor="#F4F6F7", color="#5D6D7E", fontname="Helvetica", fontsize=11];'
        'edge [color="#5D6D7E", fontname="Helvetica", fontsize=9];')
OUT = 'node [shape=box, style="rounded,filled", fillcolor="#16232A", fontcolor="white"];'
DATA = 'node [shape=note, style="filled", fillcolor="#FDF2E9", color="#B9770E", fontcolor="#16232A"];'
OP = 'node [shape=box, style="rounded,filled", fillcolor="#F4F6F7", color="#5D6D7E", fontcolor="#16232A"];'
OPT = 'node [shape=box, style="rounded,dashed,filled", fillcolor="white", color="#7F8C8D", fontcolor="#16232A"];'


def dot(body: str, rd: str = "TB") -> str:
    return HEAD.format(rd=rd) + body + "}"


def spec(rows: list[tuple[str, str]]):
    st.markdown("| setting | value |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows))


B3 = dot(DATA + 'x [label="17 annotated cues  x ∈ {0,1}¹⁷\\naspect (MC, SC) · tense · position\\nconnector · interactions"];'
         + OP + 's [label="standardise\\n(z-score, fitted on the training fold)"];'
         'l [label="3 linear scores  zₖ = bₖ + wₖᵀx̃\\none weight vector per reading"];'
         'sm [label="softmax"];'
         + OUT + 'y [label="P(Ant) · P(Post) · P(Simul)"];'
         'x -> s -> l -> sm -> y;'
         + OPT + 'p [label="L2 penalty  ‖W‖² / (2C),  C = 1"]; p -> l [style=dashed, arrowhead=none];')

TREE = dot(DATA + 'x [label="17 annotated cues"];'  # the fitted depth-3 tree, as in results/tables/tree_rules.txt
           + OP + 'node [shape=diamond, fillcolor="#EAF2F8"];'
           'd [label="main clause durative?"]; p1 [label="participial clause fronted?"]; t [label="present tense?"];'
           'f [label="main clause perfective?"]; p2 [label="participial clause fronted?"]; dy [label="main clause dynamic?"];'
           + OUT + 'l1 [label="Ant"]; l2 [label="Post"]; l3 [label="Ant"]; l4 [label="Simul"]; l5 [label="Ant"];'
           'l6 [label="Ant"]; l7 [label="Simul"];'
           'x -> d; d -> p1 [label="no"]; d -> t [label="yes"]; p1 -> f [label="no"]; p1 -> l3 [label="yes"];'
           'f -> l1 [label="no"]; f -> l2 [label="yes"]; t -> p2 [label="no"]; t -> dy [label="yes"];'
           'p2 -> l4 [label="no"]; p2 -> l5 [label="yes"]; dy -> l6 [label="no"]; dy -> l7 [label="yes"];'
           + OPT + 'fo [label="B5 forest: 1,000 such trees on bootstrap samples,\\nrandom feature subsets; average of leaf probabilities"];'
           'xg [label="B5 XGBoost: 400 depth-3 trees fitted in sequence,\\neach to the gradient of the loss; sum of leaf scores → softmax"];'
           'l2 -> fo [style=invis]; l6 -> xg [style=invis];')

ENC = dot(DATA + 't [label="sentence text\\n(marked: [[ tendo + participle ]] … << main verb >>)"];'
          'x [label="17 annotated cues\\n(_feat only)"];'
          + OP + 'tok [label="sub-word tokenizer (fast)"];'
          'enc [label="Transformer encoder, all weights fine-tuned\\nxlm-roberta-base · BERTimbau · Albertina · mDeBERTa\\n12 layers · d = 768"];'
          'pool [label="pooling: h(<s>)  ⊕  h(participle)  ⊕  h(main verb)\\n(plain: h(<s>) only; a missing span → h(<s>))"];'
          'head [label="text head: dropout 0.1 → linear (3d → 3)"];'
          + OPT + 'fh [label="cue head: linear 17 → 3, no bias\\nweights start at 0; own learning rate feat_lr"];'
          'sc [label="standardise with B3\'s scaler"];'
          'off [label="_res: B3 log-odds offset\\n(head starts at 0 = starts at B3)"];'
          'aux [label="_aux: auxiliary heads predict each cue\\n(loss weight 0.5; not used at test time)"];'
          + OP + 'sum [label="+", shape=circle, width=0.35, fixedsize=true];'
          'sm [label="softmax"];' + OUT + 'y [label="P(Ant) · P(Post) · P(Simul)"];'
          't -> tok -> enc -> pool -> head -> sum -> sm -> y;'
          'x -> sc -> fh -> sum [style=dashed]; off -> sum [style=dashed]; pool -> aux [style=dashed];')

T5 = dot(DATA + 's [label="sentence  +  question  +  3 readings as options"];'
         + OP + 'rot [label="3 cyclic option orders\\n(A,B,C) = (Ant,Post,Simul) · (Post,Simul,Ant) · (Simul,Ant,Post)"];'
         'subgraph cluster_q { label="per order: one forward pass, no text generated"; fontname="Helvetica"; fontsize=10; color="#AAB7B8";'
         'p [label="chat prompt, thinking off, ends in \\"Answer:\\""];'
         'lm [label="stock causal LLM (Qwen 4B · 9B · 27B, open weights, pinned commit)\\nor Jev 1.14 through TypeSafe\'s API (Choice question)"];'
         'lt [label="logits of the tokens  A  B  C  → softmax (fp32)\\nJev: returns the 3 option probabilities directly"]; p -> lm -> lt; }'
         'gm [label="geometric mean over the 3 orders, renormalised\\n(cancels an additive slot bias exactly)"];'
         + OUT + 'raw [label="raw P(reading)  — zero-shot, no labels"];'
         + OPT + 'cal [label="Dirichlet calibration on the training fold\\nsoftmax(W log p + b), 12 parameters"];'
         'fu [label="B3 + model:  ½ P_B3  +  ½ P_cal  (w = 0.5, fixed in advance)"];'
         's -> rot -> p; lt -> gm -> raw; raw -> cal [style=dashed]; cal -> fu [style=dashed];')

B6 = dot(DATA + 'x [label="17 cues  +  variety (EP · BP · AP · MP, one-hot)"];'
         + OP + 'm0 [label="M0: cues only  (= B3 on the 4 varieties)"];'
         'm1 [label="M1: + variety intercepts  (base rates differ)"];'
         'm2 [label="M2: + variety × cue slopes\\nL2-penalised deviations = partial pooling"];'
         'cv [label="penalty C chosen by inner 5-fold CV\\ngrid 0.003 … 3"];'
         'lr [label="deviance drop M0→M1, M1→M2 (per cue block)\\nparametric-bootstrap p-values, Holm over 6 blocks"];'
         + OUT + 'y [label="do varieties differ in base rates? in cue weights?"];'
         'x -> m0 -> m1 -> m2 -> lr -> y; cv -> m2 [style=dashed];')


def render():
    st.title("Model architectures")
    st.caption("How each model family is built, for presentation. Boxes: data (note shape), operations (rounded), "
               "outputs (dark); dashed = optional component or variant. Settings match the code.")
    tabs = st.tabs(["B3 logit", "Decision trees (B4, B5)", "Encoders (T1)", "LLM decision pipeline (T5)",
                    "Varieties model (B6)"])

    with tabs[0]:
        st.subheader("B3 · L2-penalised multinomial logistic regression")
        st.graphviz_chart(B3)
        st.latex(r"P(y = k \mid \mathbf{x}) = \frac{\exp(b_k + \mathbf{w}_k^\top \tilde{\mathbf{x}})}"
                 r"{\sum_{j} \exp(b_j + \mathbf{w}_j^\top \tilde{\mathbf{x}})},\quad "
                 r"\tilde{\mathbf{x}} = \frac{\mathbf{x} - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}}},\quad "
                 r"\min_{W,b}\; -\!\sum_i \log P(y_i \mid \mathbf{x}_i) + \tfrac{1}{2C}\lVert W \rVert^2")
        spec([("inputs", "17 binary cue columns in 6 blocks (`features.CUE_BLOCKS`)"),
              ("parameters", "3 × (17 + 1) = 54"), ("penalty", "L2, C = 1"), ("solver", "lbfgs, max_iter 5000"),
              ("why it is the reference", "additive and readable: a signed weight per cue and reading; matches the non-linear models")])
        st.caption("Pedregosa et al. 2011 (scikit-learn).")

    with tabs[1]:
        st.subheader("B4 · CART decision tree, and the B5 ensembles")
        st.graphviz_chart(TREE)
        st.markdown("The tree splits the clauses by one yes/no cue at a time, choosing at each node the split that most "
                    "reduces Gini impurity, and predicts the majority reading of each leaf. The diagram is the fitted depth-3 "
                    "tree (also on the *B4 decision tree* page): durativity of the main clause first, then position and "
                    "tense, the top cues of the hierarchy, found without being told. The "
                    "ensembles ask whether interactions between cues add anything beyond B3. They don't: XGBoost 0.695, "
                    "forest 0.692, B3 0.695.")
        st.latex(r"\text{Gini}(t) = 1 - \sum_k p_{k,t}^2,\qquad "
                 r"\text{split}^* = \arg\max_{s}\; \text{Gini}(t) - \tfrac{n_L}{n_t}\text{Gini}(t_L) - \tfrac{n_R}{n_t}\text{Gini}(t_R)")
        spec([("B4 tree", "CART, max_depth 4 (scored), 3 (shown); min_samples_leaf 20"),
              ("B5 forest", "1,000 trees, min_samples_leaf 5, bootstrap rows, random feature subsets"),
              ("B5 XGBoost", "400 trees, depth 3, learning rate 0.05, subsample 0.8, colsample 0.8, λ = 1"),
              ("explanation", "path-dependent TreeSHAP (Lundberg et al. 2020), as a rank check only")])
        st.caption("Breiman et al. 1984 (CART); Breiman 2001 (random forests); Chen & Guestrin 2016 (XGBoost).")

    with tabs[2]:
        st.subheader("Fine-tuned Transformer encoders: `plain`, `marked` and their variants")
        st.graphviz_chart(ENC)
        st.latex(r"\mathbf{h} = [\,h_{\langle s\rangle};\, h_{\text{part}};\, h_{\text{verb}}\,] \in \mathbb{R}^{3d},\quad "
                 r"\mathbf{z} = W_t\,\mathrm{drop}(\mathbf{h}) + b_t \;\;[+\; W_f \tilde{\mathbf{x}}]_{\_feat} "
                 r"\;\;[+\; \log \tfrac{P_{B3}}{\pi}]_{\_res},\quad P = \mathrm{softmax}(\mathbf{z})")
        st.markdown("| variant | text | uses the cues as… |\n|---|---|---|\n"
                    "| `plain` | raw sentence, h(&lt;s&gt;) only | — |\n"
                    "| `marked` | verbs marked, 3 pooled vectors | — |\n"
                    "| `_feat` | as above | extra input, added to the logits (cue head) |\n"
                    "| `_res` | as above | starting point: the head learns a correction to B3's log-odds |\n"
                    "| `_txt` | cues written as Portuguese words before the sentence | words |\n"
                    "| `_aux` | as above | auxiliary targets during training only |")
        spec([("checkpoints", "xlm-roberta-base; BERTimbau (PT-BR); Albertina 100M (PT-PT); mDeBERTa-v3 (fp32)"),
              ("optimiser", "AdamW, weight decay 0.01; linear schedule, 10% warm-up; gradient clip 1"),
              ("training", "batch 16, max length 256, bf16 autocast (fp32 for DeBERTa models)"),
              ("loss", "cross-entropy weighted by inverse class frequency (+ 0.5 × auxiliary loss for `_aux`)"),
              ("tuning, inside each training fold", "lr 3e-5 / 5e-5 × 6 / 12 epochs (× feat_lr 1e-4 / 1e-3 / 1e-2 for `_feat`)"),
              ("evaluation", "B3's folds, 5 seeds, group bootstrap; fusion with B3 at w = 0.5 for cue-free variants")])
        st.caption("Conneau et al. 2020 (XLM-R); Souza et al. 2020 (BERTimbau); Rodrigues et al. 2023 (Albertina); "
                   "He et al. 2023 (DeBERTaV3); Loshchilov & Hutter 2019 (AdamW); Clark et al. 2019 (product of experts, `_res`).")

    with tabs[3]:
        st.subheader("T5 · zero-shot decisions from an LLM's option probabilities")
        st.graphviz_chart(T5)
        st.latex(r"p^{(o)}_k = \mathrm{softmax}_{k}\big(\ell^{(o)}_{\text{slot}_o(k)}\big),\quad "
                 r"\hat p_k \propto \exp\Big(\tfrac{1}{3}\sum_{o=1}^{3} \log p^{(o)}_k\Big),\quad "
                 r"p^{\text{cal}} = \mathrm{softmax}(W \log \hat p + b)")
        st.markdown("The model never writes text. The prompt ends where the next token must be an option letter; the "
                    "logits of ` A`, ` B`, ` C` are read and soft-maxed. A slot preference adds a constant to the "
                    "log-probability of whatever sits in that slot, and each reading visits every slot once, so the "
                    "geometric mean cancels it. Only the optional calibration uses labels, and only on the training "
                    "fold. For Jev, the same sentence, question and readings go to TypeSafe's API as a Choice question, "
                    "with options named `before` / `after` / `same_time`.")
        st.code(decision.render_user(decision.build_prompt(
            "Orlando Janeiro desfilou pelas artérias da cidade, tendo apostado no contacto interpessoal.")), language=None)
        spec([("models", "Qwen3.5-4B, 9B; Qwen3.6-27B-FP8; Qwen3.8-27B-FP8 (pre-set headline); Jev 1.14.0"),
              ("inference", "one forward pass per prompt × 3 orders; batch 8, left padding, fp32 softmax"),
              ("texts", "plain, marked, and 7 perturbations (masks, random-word controls, fronting)"),
              ("calibration", "multinomial logit on log-probabilities (Dirichlet calibration), 12 parameters, training folds"),
              ("fusion", "½ B3 + ½ calibrated model; fixed before the run")])
        st.caption("Zheng et al. 2024 (option-order bias); Zhao et al. 2021 (calibrate before use); Kull et al. 2019 "
                   "(Dirichlet calibration). See docs/CALIBRATION.md.")

    with tabs[4]:
        st.subheader("B6 · pooled hierarchical model over the four varieties (RQ2)")
        st.graphviz_chart(B6)
        st.latex(r"z_{k}(\mathbf{x}, v) = b_k + \beta_{k,v} + (\mathbf{w}_k + \boldsymbol{\delta}_{k,v})^\top \tilde{\mathbf{x}},"
                 r"\qquad \text{penalty on } \boldsymbol{\delta} \text{ only (partial pooling)}")
        spec([("varieties", "EP, BP, AP, MP (n = 1,038; British English excluded, 95% Ant)"),
              ("penalty", "L2 on slope deviations; main effects scaled so the penalty barely touches them"),
              ("C", "chosen by inner 5-fold CV from 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3 (nested)"),
              ("tests", "deviance drops, parametric bootstrap (300 draws), Holm over the 6 cue blocks")])
        st.caption("Pedregosa et al. 2011.")
