# Why `lineno.sty` is in this folder

TeX Live 2026 ships `lineno` v5.7 (2026/02/16), which misplaces the review-mode line numbers of `acl.sty`: the
numbers land inside the text or in the gap between the columns. This folder carries `lineno` v5.4 (2025/01/29),
unchanged from the TeX Live 2024 final archive:

- ftp.math.utah.edu/pub/tex/historic/systems/texlive/2024/tlnet-final/archive/lineno.tar.xz

LaTeX loads this local copy before the system one. The ACL style files are not modified.

Delete this copy once `lineno` is fixed upstream, or when compiling on Overleaf.
