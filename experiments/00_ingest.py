"""Convert the later annotation batches to the pipeline's export format.

* data/raw/archive/Dados_Violeta.xlsx -> data/raw/dripps_violeta.csv (PT-EU/AO/MZ)
* data/raw/DADOS_João_Abergaria_PE.xlsx -> data/raw/dripps_abergaria.csv (EP only):
  one sheet, a different column layout, its own tense spellings, no DR/SR-SC, and
  a connector column that is "N/A" throughout. Rows whose sentence already exists in
  an earlier file, or repeats inside the sheet with the same annotation, are dropped
  (reported, with whether the two annotators agreed).


Writes data/raw/dripps_violeta.csv (committed, so nothing downstream needs an
xlsx reader). New rows get a ``V`` in the ID prefix (PTEUV1, PTAOV1, PTMZV1) so
they never collide with the original export and the batch is readable from the
ID. The workbook has no SR-SC column, so that field is left empty; ``io.load``
accepts the empty value and the leakage checks skip those rows.
Reads the xlsx with the stdlib: it is plain OOXML and one script needs no
new dependency.
"""

from __future__ import annotations

import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dripps import schema  # noqa: E402

SRC = ROOT / "data/raw/archive/Dados_Violeta.xlsx"
OUT = ROOT / "data/raw/dripps_violeta.csv"
M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
PREFIX = {"Data PT-EU": "PTEUV", "Data PT-AO": "PTAOV", "Data PT-MZ": "PTMZV"}
# workbook column -> export column (B..J; A is a running number, K holds stray notes)
COLS = {"B": None, "C": schema.SENTENCE, "D": "Position", "E": schema.TARGET,
        "F": "TMC", "G": "ATMC", "H": "ATSC", "I": "DR", "J": "CNT"}
# spelling variants in the workbook; anything else unknown fails in io.load
ALIASES = {"TMC": {"Futuro Perfeito": "Fut-C"}}


def _sheet_paths(z: zipfile.ZipFile) -> dict[str, str]:
    """Sheet name -> part path, resolved through workbook.xml.rels (tab order is not file order)."""
    rid = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    return {s.get("name"): "xl/" + rels[s.get(rid)].lstrip("/").removeprefix("xl/")
            for s in ET.fromstring(z.read("xl/workbook.xml")).iter(f"{M}sheet")}


def _warn_blank_mesmo(rows: list[dict], label: str) -> None:
    """A concessive 'mesmo tendo' with no connector is probably an annotation gap; report, never edit."""
    bad = [r[schema.ID] for r in rows if re.search(r"\bmesmo\s+tendo\b", r[schema.SENTENCE], re.I) and not r["CNT"]]
    if bad:
        print(f"  warning ({label}): 'mesmo tendo' with empty CNT: {bad}")


def _cells(z: zipfile.ZipFile, name: str, strings: list[str]):
    for row in ET.fromstring(z.read(name)).iter(f"{M}row"):
        out = {}
        for c in row.iter(f"{M}c"):
            v = c.find(f"{M}v")
            if v is None or not v.text:
                continue
            out[re.match(r"[A-Z]+", c.get("r")).group()] = (
                strings[int(v.text)] if c.get("t") == "s" else v.text
            )
        yield out


ABERGARIA_SRC = ROOT / "data/raw/DADOS_João_Abergaria_PE.xlsx"
ABERGARIA_OUT = ROOT / "data/raw/dripps_abergaria.csv"
# sheet column -> export column (A is empty; K onwards unused); C is the newspaper
A_COLS = {"D": schema.SENTENCE, "E": "Position", "F": schema.TARGET, "G": "TMC", "H": "ATMC", "I": "ATSC", "J": "CNT"}
# this annotator's tense spellings; "Pres - Ind" style spacing is normalised first.
# Judgement calls: PMQP C -> PMP (composite pluperfect), "Inf - opor" -> INF-S,
# "quer (Pres)" -> Pres (modal main verb), passive "ser (PP) + Part. P" -> PP,
# bare "Part. P" (participial main clause) -> PartP, a new non-finite bundle.
A_TMC = {"Pres-Ind": "Pres-Ind", "PIMP-Ind": "PIMP-Ind", "PPC-Ind": "PPC-Ind", "PMQP C-Ind": "PMP",
         "Inf-opor": "INF-S", "Inf-C": "INF-C", "quer (Pres)": "Pres", "ser (PP) + Part. P": "PP",
         "Part. P": "PartP", "ir (Pres) + INF-S": "ir(Pres)+INF-S"}


def abergaria(z: zipfile.ZipFile) -> None:
    from dripps import io  # noqa: E402  (after sys.path is set)

    strings = ["".join(t.text or "" for t in si.iter(f"{M}t"))
               for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(f"{M}si")]
    seen = pd.concat([io.load(io.ORIGINAL_RAW), io.load(io.VIOLETA_RAW)])
    prior_rows = {}  # a multi-APC sentence has several earlier rows; keep them all
    for s, i, r in zip(seen.Sentence, seen.ID, seen[["TR", "TMC", "ATMC", "ATSC", "Position"]].to_dict("records")):
        prior_rows.setdefault(io.group_key(s), []).append((i, r))
    lines, kept, dropped, n, kept_recs = [";".join(schema.RAW_COLUMNS) + ";"], {}, [], 0, []
    for cells in _cells(z, next(iter(_sheet_paths(z).values())), strings):
        sent = " ".join(cells.get("D", "").split())
        if not sent or cells.get("B") in (None, "Abbreviation"):
            continue
        rec = {col: " ".join(cells.get(l, "").split()) for l, col in A_COLS.items()}
        rec["TMC"] = re.sub(r"\s*-\s*", "-", rec["TMC"])
        rec["TMC"] = A_TMC.get(rec["TMC"], rec["TMC"])
        rec["CNT"] = "" if rec["CNT"] == "N/A" else rec["CNT"]
        key = io.group_key(sent)
        ann = {k: rec[k] for k in ("TR", "TMC", "ATMC", "ATSC", "Position")}
        if key in prior_rows:  # already in an earlier file
            match = next((i for i, r in prior_rows[key] if r == ann), None)
            other, prior = match or prior_rows[key][0][0], prior_rows[key][0][1]
            dropped.append((cells["B"], other, "agree" if match else f"differ on {[k for k in ann if prior[k] != ann[k]]}"))
            continue
        if ann in kept.get(key, []):  # same sentence, same annotation, twice in the sheet
            dropped.append((cells["B"], "this sheet", "identical row"))
            continue
        kept.setdefault(key, []).append(ann)
        n += 1
        kept_recs.append(rec)
        rec.update({schema.ID: f"PTEUJ{n}", "DR": "", "SR-SC": ""})
        lines.append(";".join(rec[c] for c in schema.RAW_COLUMNS) + ";")
    for orig, other, verdict in dropped:
        print(f"  dropped {orig} (duplicate of {other}; annotations {verdict})")
    ABERGARIA_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _warn_blank_mesmo(kept_recs, "Abergaria")
    print(f"{ABERGARIA_SRC.name}: {n} rows -> PTEUJ1..PTEUJ{n}; wrote {ABERGARIA_OUT.relative_to(ROOT)}")


def main() -> None:
    z = zipfile.ZipFile(SRC)
    strings = ["".join(t.text or "" for t in si.iter(f"{M}t"))
               for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(f"{M}si")]
    names = _sheet_paths(z)
    lines, recs = [";".join(schema.RAW_COLUMNS) + ";"], []
    for sheet, prefix in PREFIX.items():
        n = 0
        for cells in _cells(z, names[sheet], strings):
            if not cells.get("C", "").strip() or cells.get("C") == "Sentences":
                continue  # header rows and the stray notes below the data
            n += 1
            if "A" in cells and cells["A"].isdigit() and int(cells["A"]) != n:
                raise SystemExit(f"{sheet}: row number {cells['A']} != running count {n}; ids would not trace to the workbook")
            rec = {col: " ".join(cells.get(letter, "").split()) for letter, col in COLS.items() if col}
            rec["DR"] = rec["DR"].lower()
            rec["TMC"] = ALIASES["TMC"].get(rec["TMC"], rec["TMC"])
            rec[schema.ID], rec["SR-SC"] = f"{prefix}{n}", ""
            lines.append(";".join(rec[c] for c in schema.RAW_COLUMNS) + ";")
            recs.append(rec)
        print(f"{sheet}: {n} rows -> {prefix}1..{prefix}{n}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _warn_blank_mesmo(recs, "Violeta")
    print(f"wrote {OUT.relative_to(ROOT)}")
    if ABERGARIA_SRC.exists():  # the committed CSV is enough on a clone without the workbook
        abergaria(zipfile.ZipFile(ABERGARIA_SRC))


if __name__ == "__main__":
    main()
