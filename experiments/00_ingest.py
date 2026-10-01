"""Convert data/raw/archive/Dados_Violeta.xlsx to the pipeline's export format.

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


def main() -> None:
    z = zipfile.ZipFile(SRC)
    strings = ["".join(t.text or "" for t in si.iter(f"{M}t"))
               for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(f"{M}si")]
    names = {s.get("name"): f"xl/worksheets/sheet{i}.xml" for i, s in
             enumerate(ET.fromstring(z.read("xl/workbook.xml")).iter(f"{M}sheet"), 1)}
    lines = [";".join(schema.RAW_COLUMNS) + ";"]
    for sheet, prefix in PREFIX.items():
        n = 0
        for cells in _cells(z, names[sheet], strings):
            if not cells.get("C", "").strip() or cells.get("C") == "Sentences":
                continue  # header rows and the stray notes below the data
            n += 1
            rec = {col: " ".join(cells.get(letter, "").split()) for letter, col in COLS.items() if col}
            rec["DR"] = rec["DR"].lower()
            rec["TMC"] = ALIASES["TMC"].get(rec["TMC"], rec["TMC"])
            rec[schema.ID], rec["SR-SC"] = f"{prefix}{n}", ""
            lines.append(";".join(rec[c] for c in schema.RAW_COLUMNS) + ";")
        print(f"{sheet}: {n} rows -> {prefix}1..{prefix}{n}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
