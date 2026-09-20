"""Build the self-contained results viewer at results/viewer.html.

Reads everything from results/tables/ and docs/, inlines it as JSON, and writes
one HTML file with no external data fetches -- it has to render from the file
alone once published.

Run after 02-05; it reads what they write and names the missing target if a
table is absent.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dripps import features, interpret, io, models, schema  # noqa: E402

TABLES = ROOT / "results/tables"
TEMPLATE = ROOT / "experiments/assets/viewer_template.html"
OUT = ROOT / "results/viewer.html"

PRODUCED_BY = {
    "baselines.csv": "make baselines",
    "cue_importance.csv": "make hierarchy",
    "logit_coefficients.csv": "make hierarchy",
    "marginal_association.csv": "make hierarchy",
    "oner_rule.json": "make hierarchy",
    "tree_rules.txt": "make hierarchy",
    "variety_cue_importance.csv": "make varieties",
    "variety_cue_ranks.csv": "make varieties",
    "variety_performance.csv": "make varieties",
    "shap_summary.csv": "make explain",
    "shap_blocks.csv": "make explain",
    "shap_forest.csv": "make explain",
    "shap_xgboost.csv": "make explain",
}


def _need(name: str) -> Path:
    path = TABLES / name
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(ROOT)} -- run `{PRODUCED_BY[name]}` first"
        )
    return path


def _records(name: str) -> list[dict]:
    return pd.read_csv(_need(name)).to_dict(orient="records")


# --- markdown, only the subset docs/methods.md uses -------------------------

def _inline(text: str) -> str:
    text = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    # [key] and [key1; key2] are citation keys into references.bib
    def cite(m):
        keys = [k.strip() for k in m.group(1).split(";")]
        links = ", ".join(f'<a class="cite" href="#ref-{k}">{k}</a>' for k in keys)
        return f"[{links}]"
    return re.sub(r"\[([A-Za-z0-9_\-;, ]+)\]", cite, text)


def _markdown(md: str) -> str:
    out, rows, para = [], [], []

    def flush_para():
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()

    def flush_table():
        if not rows:
            return
        head, body = rows[0], rows[2:]  # rows[1] is the |---| separator
        th = "".join(f"<th>{_inline(c)}</th>" for c in head)
        tb = "".join(
            "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
            for r in body
        )
        out.append(
            f'<div class="scroll-x"><table><thead><tr>{th}</tr></thead>'
            f"<tbody>{tb}</tbody></table></div>"
        )
        rows.clear()

    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            flush_para()
            rows.append([c.strip() for c in stripped.strip("|").split("|")])
            continue
        flush_table()
        if not stripped:
            flush_para()
        elif stripped.startswith("#"):
            flush_para()
            level = len(stripped) - len(stripped.lstrip("#"))
            out.append(f"<h{level}>{_inline(stripped[level:].strip())}</h{level}>")
        else:
            para.append(stripped)
    flush_para()
    flush_table()
    return "\n".join(out)


def _references(bib: str) -> list[dict]:
    """Enough BibTeX parsing for our own file: key, and the fields we display."""
    entries = []
    for block in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.S):
        key, body = block.group(1).strip(), block.group(2)
        fields = {
            m.group(1).lower(): " ".join(m.group(2).split())
            for m in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}(?=,\s*\n|\s*\n?$)", body, re.S)
        }
        clean = lambda s: re.sub(r"[{}\\]|\$", "", s or "")  # noqa: E731
        entries.append({
            "key": key,
            "author": clean(fields.get("author", "")).replace(" and ", "; "),
            "title": clean(fields.get("title", "")),
            "venue": clean(fields.get("journal") or fields.get("booktitle", "")),
            "year": fields.get("year", ""),
            "volume": fields.get("volume", ""),
            "pages": clean(fields.get("pages", "")),
            "doi": fields.get("doi", ""),
            "url": fields.get("url", ""),
        })
    return sorted(entries, key=lambda e: (e["author"], e["year"]))


# --- payload ----------------------------------------------------------------

def _shap_cube(slug: str, ids: list[str]) -> dict:
    """SHAP long form back into nested arrays: [class][row][feature]."""
    long = pd.read_csv(_need(f"shap_{slug}.csv"))
    classes = sorted(long["class"].unique())
    feats = list(dict.fromkeys(long["feature"]))
    wide = (
        long.pivot_table(index=["class", "id"], columns="feature", values="shap")
        .reindex(columns=feats)
    )
    cube = [
        wide.loc[c].reindex(ids).round(4).to_numpy().tolist()
        for c in classes
    ]
    return {"classes": classes, "features": feats, "values": cube}


def build_payload() -> dict:
    full = io.load()
    df = io.analysis_frame(full)
    X, y = features.build(df), features.target(df)
    ids = [str(i) for i in df[schema.ID]]

    tree = models.b4_tree(max_depth=3).fit(X, y)
    structure = interpret.tree_structure(
        tree, X.columns, tree.classes_, X=X, ids=ids
    )

    variety_imp = pd.read_csv(_need("variety_cue_importance.csv"), index_col=0)
    variety_ranks = pd.read_csv(_need("variety_cue_ranks.csv"), index_col=0)

    return {
        "meta": {
            "generated": date.today().isoformat(),
            "n_rows": int(len(df)),
            "n_total": int(len(full)),
            "n_features": int(X.shape[1]),
            "classes": list(schema.TR_LABELS),
            "varieties": list(schema.PT_VARIETIES),
            "reference_variety": schema.REFERENCE_VARIETY,
            "seed": int(schema.SEED),
        },
        "blocks": {k: list(v) for k, v in features.CUE_BLOCKS.items()},
        "derived": {k: list(v) for k, v in features.DERIVED.items()},
        "cue_importance": _records("cue_importance.csv"),
        "baselines": _records("baselines.csv"),
        "marginal": _records("marginal_association.csv"),
        "oner": json.loads(_need("oner_rule.json").read_text(encoding="utf-8")),
        "coefficients": {
            "features": list(pd.read_csv(_need("logit_coefficients.csv"), index_col=0).index),
            "classes": list(pd.read_csv(_need("logit_coefficients.csv"), index_col=0).columns),
            "values": pd.read_csv(_need("logit_coefficients.csv"), index_col=0)
                        .round(4).to_numpy().tolist(),
        },
        "variety": {
            "blocks": list(variety_imp.index),
            "names": list(variety_imp.columns),
            "importance": variety_imp.round(4).to_numpy().tolist(),
            "ranks": variety_ranks.to_numpy().tolist(),
            "performance": _records("variety_performance.csv"),
        },
        "tree": structure,
        "tree_rules": _need("tree_rules.txt").read_text(encoding="utf-8"),
        "shap": {
            "summary": _records("shap_summary.csv"),
            "blocks": _records("shap_blocks.csv"),
            "ids": ids,
            "models": {
                "B5 forest": _shap_cube("forest", ids),
                "B5 xgboost": _shap_cube("xgboost", ids),
            },
        },
        "sentences": {
            str(r[schema.ID]): {
                "text": str(r[schema.SENTENCE]),
                "variety": str(r["variety"]),
                "tr": str(r[schema.TARGET]),
            }
            for _, r in df.iterrows()
        },
        "methods": _markdown((ROOT / "docs/methods.md").read_text(encoding="utf-8")),
        "references": _references((ROOT / "docs/references.bib").read_text(encoding="utf-8")),
    }


def _clean(obj):
    """NaN -> None, so the page gets JSON null instead of a bare NaN token.

    The interaction block has no permutation importance by construction, and
    pandas reads that blank as NaN; ``json`` would emit the non-standard literal
    ``NaN``, which ``JSON.parse`` rejects.
    """
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def render(payload: dict, template: str) -> str:
    blob = json.dumps(_clean(payload), ensure_ascii=False, allow_nan=False)
    # corpus text is arbitrary; a literal </script would close the host element
    blob = blob.replace("</", "<\\/")
    return template.replace("__PAYLOAD__", blob)


def main() -> None:
    payload = build_payload()
    html = render(payload, TEMPLATE.read_text(encoding="utf-8"))
    OUT.write_text(html, encoding="utf-8")
    size = len(html.encode()) / 1024
    print(f"wrote {OUT.relative_to(ROOT)} ({size:.0f} KB, "
          f"{len(payload['sentences'])} sentences, "
          f"{len(payload['tree']['nodes'])} tree nodes)")


if __name__ == "__main__":
    main()
