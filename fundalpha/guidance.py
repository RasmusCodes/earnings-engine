"""Management outlook (next-quarter revenue and gross-margin guidance) extracted from earnings releases.

Point-in-time by construction: a guidance record carries the acceptance timestamp of the 8-K it came from, and
the only number used to validate a parse is the revenue of the quarter *reported in the same release* (the base
for sequential-percentage guidance and for unit inference). Actual results of the guided quarter are never used
to accept or reject a parse.

Formats handled (all normalised to low / high / mid):
  range     "revenue ... approximately $1,365 to $1,415 million"          table "Revenues | $9.3B - $10.1B"
  pm_abs    "net revenue of $2.4 billion, plus or minus $150 million"     table "Revenue | $4.65 Billion | +/- | $300 Million"
  pm_pct    "Revenue is expected to be $43.0 billion, plus or minus 2%"
  point     "revenue of approximately $1.2 billion"
  seq_pct   "Revenue is expected to be up 6 to 8 percent from the fourth quarter" (base = reported quarter)
Gross margin: GAAP / non-GAAP / unspecified basis, e.g. "GAAP and non-GAAP gross margins are expected to be 70.6% and
71.0%, respectively, plus or minus 50 basis points", and LRCX/KLA-style tables with U.S. GAAP and Non-GAAP columns.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np

PARSER_VERSION = "2026-11-13.2"

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "twenty": 20,
             "twenty-five": 25, "thirty": 30, "forty": 40, "fifty": 50, "seventy-five": 75, "one hundred": 100}
UNIT = {"billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "mn": 1e6, "mm": 1e6, "m": 1e6, "mil": 1e6,
        "thousand": 1e3, "k": 1e3}

N = r"(\d+(?:\.\d+)?)"
UNITS = r"billion|million|thousand|bn|mn|mm|mil|b|m|k"
U = rf"({UNITS})?"
MONEY = rf"\$\s?{N}\s*{U}\b"
SEP = r"(?:to|and|through|-)"

REV_SUBJ = re.compile(r"\b(?:total |net |consolidated |company |gaap |overall )?(?:revenues?|net sales|sales)\b", re.I)
FWD_STRONG = re.compile(
    r"\b(?:expected|anticipated|forecasted|forecast|forecasting|projected|projecting|guided|guiding|expects?|"
    r"anticipates?|forecasts?|outlook|guidance|targets?|targeted|targeting|planning|plans)\b|"
    r"\b(?:to|will|should) be (?:in the range|between|approximately|about|roughly|around|up|down|flat|in line)|"
    r"\b(?:to|will) (?:increase|decrease|grow|decline|range)\b", re.I)
PAST = re.compile(r"\b(?:was|were|came in|reported|achieved|delivered|exceeded|increased|decreased|grew|declined|"
                  r"rose|fell|totaled|totalled|ended the|recorded|posted|met|missed|included|reached|surpassed)\b",
                  re.I)
# percentage-change guidance ("revenue to be up 6 to 8 percent from the fourth quarter") needs an explicit
# forward-looking verb in the same sentence; bullets and table notes ("Test revenue down 7% in Q122") do not count
STRICT_FWD = re.compile(r"\b(?:expect|anticipat|forecast|project|guid|outlook)\w*", re.I)
GM_SUBJ = re.compile(r"gross\s+(?:profit\s+)?margins?", re.I)
SEGMENT_HINT = re.compile(r"\b(?:segment|qct|qtl|data center|gaming|automotive|semiconductor systems|services|"
                          r"product revenue|license|licensing|royalt|incremental|excluding|bookings|backlog|"
                          r"shipments|orders|deferred|division|business unit|group)\b", re.I)
OUTLOOK_HDR = re.compile(r"(outlook|guidance|forecast|business update|expectations|quarter ending|"
                         r"q[1-4]\s*(?:fy|'|20)|(first|second|third|fourth) (?:fiscal )?quarter|"
                         r"(march|june|september|december) (?:\d{4} )?quarter)", re.I)
ANNUAL = re.compile(r"(fiscal|full|calendar)[- ]year|full-year|for the year|annual", re.I)
QTR = re.compile(r"\bquarter\b|\bq[1-4]\b", re.I)
BULLETS = "*·•▪■●◦‣⁃"
BULLET = re.compile(rf"^[\s\-–{re.escape(BULLETS)}]+(?=\w|\$)")
OUTLOOK_STRICT = re.compile(r"(outlook|guidance|forecast|expects the following|business update|"
                            r"expectations for|targets for)", re.I)


@dataclass
class Guide:
    metric: str
    basis: str
    kind: str
    low: float
    high: float
    mid: float
    snippet: str
    doc_type: str
    source: str
    pos: int = 0


def _norm(s: str) -> str:
    s = s.replace(" | ", " ").replace("|", " ")
    s = BULLET.sub("", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)                           # 1,365 -> 1365
    s = re.sub(r"\bbasis points?\b|\bbps\b|\bbp\b", "bps", s, flags=re.I)
    s = re.sub(r"\bpercentage points?\b", "pp", s, flags=re.I)
    s = re.sub(r"(\d)\s*points?\b", r"\1pp", s, flags=re.I)
    s = re.sub(r"\bper\s*cent\b|\bpercent\b|\bpct\b", "%", s, flags=re.I)
    for w, v in sorted(NUM_WORDS.items(), key=lambda kv: -len(kv[0])):
        s = re.sub(rf"\b{w}\b(?=\s*(?:%|bps|pp))", str(v), s, flags=re.I)
    s = re.sub(r"(\d)\s+(%|bps|pp)", r"\1\2", s)
    s = re.sub(r"\+\s*/\s*[-−]|plus or minus|plus/minus|\(\+/-\)", "±", s, flags=re.I)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _scale(u: str | None) -> float | None:
    return UNIT.get(u.lower()) if u else None


def _infer(values: list[float], unit: float | None, base: float | None, hint: float | None = None) -> float | None:
    """Unit multiplier: explicit, else a table header hint ("in millions"), else the single scale that puts the
    midpoint near the reported quarter's revenue. A bare number below 20 is never scaled up: it is a per-share
    figure ("$0.86 and $0.96"), not revenue."""
    if unit:
        return unit
    if hint:
        return hint
    if not base or max(values) < 20:
        return None
    mid = float(np.mean(values))
    fits = [s for s in (1e9, 1e6, 1e3, 1.0) if 0.4 <= mid * s / base <= 2.5]
    return fits[0] if len(fits) == 1 else None


def unit_hint(text: str) -> float | None:
    t = text.lower()
    if re.search(r"in billions|\$ ?billions|\(billions\)", t):
        return 1e9
    if re.search(r"in millions|\$ ?millions|\(millions\)|\$m\b", t):
        return 1e6
    if re.search(r"in thousands|\$ ?thousands|\(thousands\)", t):
        return 1e3
    return None


MAX_SUBJ_DIST = 300
OUTLOOK_TABLE = re.compile(r"(outlook|guidance|forecast|expects? the following|expected to be|are expected|"
                           r"is expected|anticipat|as follows|business update|targets? for)", re.I)
TNUM = re.compile(rf"\$?\s?(\d+(?:\.\d+)?)(?!\.?\d)\s*((?:{UNITS})(?![A-Za-z]))?(?!\s*%)", re.I)


def parse_table_revenue(s: str, base: float | None, hint: float | None, ctx: str = "") -> list[tuple]:
    """Outlook-table revenue rows whose cells carry no '$' or use low/mid/high columns.

      "Total net revenue | 7,100 | +/- | 400"           (AMAT, header "In millions")  -> pm_abs
      "Total Revenue | $3,025 | $3,100 | $3,175 | ..."  (NXP, Low | Mid | High)       -> range(low, high)
      "Revenue | 1,350 - 1,450"                                                        -> range
      "Revenue | 7,100"                                  (single cell)                  -> point
    `s` is the normalised row starting at the revenue label."""
    m0 = re.search(r"[\$\d]", s)
    if not m0:
        return []
    body = s[m0.start():]
    m = re.match(rf"\$?\s?{N}\s*((?:{UNITS})(?![A-Za-z]))?\s*,?\s*±\s*\$?\s?{N}\s*(%|(?:{UNITS})(?![A-Za-z]))?",
                 body, re.I)
    if m:
        c, cu, w, wu = float(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
        scale = _infer([c], _scale(cu), base, hint)
        if scale:
            if wu == "%":
                return [("pm_pct", c * scale * (1 - w / 100), c * scale * (1 + w / 100), m.span())]
            ws = _scale(wu) or scale
            return [("pm_abs", c * scale - w * ws, c * scale + w * ws, m.span())]
    m = re.match(rf"\$?\s?{N}\s*((?:{UNITS})(?![A-Za-z]))?\s*(?:to|-|and|through)\s*\$?\s?{N}\s*"
                 rf"((?:{UNITS})(?![A-Za-z]))?(?!\s*%)", body, re.I)
    if m:
        a, au, b, bu = float(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
        sb = _infer([a, b], _scale(bu), base, hint)
        sa = _scale(au) or sb
        if sa and sb and b * sb >= a * sa:
            return [("range", a * sa, b * sb, m.span())]
    nums = [(float(x), u) for x, u in TNUM.findall(body)]
    low_high_header = bool(re.search(r"\blow\b.*\bhigh\b", ctx, re.I | re.S))
    if len(nums) >= 3 and low_high_header:
        (a, ua), (b, _), (c, uc) = nums[:3]
        if a < b < c and abs(b - (a + c) / 2) <= 0.03 * b:
            sc = _infer([a, c], _scale(uc) or _scale(ua), base, hint)
            if sc:
                return [("range", a * sc, c * sc, (0, len(body)))]
    if len(nums) == 1:
        c, cu = nums[0]
        sc = _infer([c], _scale(cu), base, hint)
        if sc:
            return [("point", c * sc, c * sc, (0, len(body)))]
    return []


def parse_revenue(s: str, base: float | None, hint: float | None = None, allow_point: bool = True,
                  allow_pct: bool = True) -> list[tuple]:
    """The revenue amount that follows the subject: (kind, low, high, span), at most one candidate.

    `s` starts at the revenue subject; only the first amount expression is kept so a later per-share or segment
    figure in the same sentence can never be taken for revenue."""
    out = _revenue_candidates(s, base, hint, allow_point, allow_pct)
    out = [c for c in out if c[3][0] <= MAX_SUBJ_DIST]
    return sorted(out, key=lambda c: c[3][0])[:1]


def _revenue_candidates(s: str, base: float | None, hint: float | None, allow_point: bool = True,
                        allow_pct: bool = True) -> list[tuple]:
    out = []
    for m in re.finditer(rf"{MONEY}\s*,?\s*(?:at the mid-?point\s*,?\s*)?(?:,?\s*)±\s*(\$\s?)?{N}\s*"
                         rf"(%|(?:{UNITS})(?![A-Za-z]))?", s, re.I):
        c, cu, w, wu = float(m.group(1)), m.group(2), float(m.group(4)), m.group(5)
        scale = _infer([c], _scale(cu), base, hint)
        if not scale:
            continue
        if wu == "%":
            out.append(("pm_pct", c * scale * (1 - w / 100), c * scale * (1 + w / 100), m.span()))
        else:
            wscale = _scale(wu) or scale
            out.append(("pm_abs", c * scale - w * wscale, c * scale + w * wscale, m.span()))
    for m in re.finditer(rf"\$\s?{N}\s*{U}\s*{SEP}\s*\$?\s?{N}\s*{U}\b", s, re.I):
        if any(x[3][0] <= m.start() < x[3][1] for x in out):
            continue
        a, au, b, bu = float(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
        unit_b = _scale(bu)
        scale_b = _infer([a, b], unit_b, base, hint)
        scale_a = _scale(au) or scale_b
        if scale_a and scale_b and b * scale_b >= a * scale_a:
            out.append(("range", a * scale_a, b * scale_b, m.span()))
    if not out and allow_point:
        for m in re.finditer(rf"(?:approximately|about|roughly|around|of|be|~)\s*{MONEY}", s, re.I):
            c = float(m.group(1))
            scale = _infer([c], _scale(m.group(2)), base, hint)
            if scale:
                out.append(("point", c * scale, c * scale, m.span()))
                break
    if not out and base and allow_pct:
        sp = _parse_pct_change(s)
        if sp is not None:
            out.append(sp)
    return out


DIR = r"(up|down|increase|decrease|increasing|decreasing|grow|growth|decline|declining|higher|lower|flat)"
SIGN = {"up": 1, "increase": 1, "increasing": 1, "grow": 1, "growth": 1, "higher": 1,
        "down": -1, "decrease": -1, "decreasing": -1, "decline": -1, "declining": -1, "lower": -1}


def _parse_pct_change(s: str):
    sl = s.lower()
    if "$" in sl:
        return None
    basis = "yoy" if re.search(r"year[- ]over[- ]year|year[- ]ago|same quarter (?:of )?(?:last|the prior) year", sl) else "seq"
    m = re.search(rf"flat\s*(?:to|and|-)\s*{DIR}\s*(?:by\s*)?(?:approximately\s*|about\s*)?{N}%", sl)
    if m and m.group(1) in SIGN:
        v = SIGN[m.group(1)] * float(m.group(2))
        return (f"{basis}_pct", min(0.0, v), max(0.0, v), m.span())
    m = re.search(rf"{DIR}\s*(?:by\s*)?(?:approximately\s*|about\s*)?{N}%\s*(?:to|and|-)\s*{DIR}\s*(?:by\s*)?{N}%", sl)
    if m and m.group(1) in SIGN and m.group(3) in SIGN:
        a, b = SIGN[m.group(1)] * float(m.group(2)), SIGN[m.group(3)] * float(m.group(4))
        return (f"{basis}_pct", min(a, b), max(a, b), m.span())
    m = re.search(rf"{DIR}\s*(?:by\s*|of\s*|in the range of\s*|between\s*)?(?:approximately\s*|about\s*)?"
                  rf"{N}%?\s*(?:to|and|-)\s*{N}%", sl)
    if m and m.group(1) in SIGN:
        a, b = SIGN[m.group(1)] * float(m.group(2)), SIGN[m.group(1)] * float(m.group(3))
        return (f"{basis}_pct", min(a, b), max(a, b), m.span())
    m = re.search(rf"{DIR}\s*(?:by\s*|of\s*)?(?:approximately\s*|about\s*|roughly\s*)?{N}%", sl)
    if m and m.group(1) in SIGN:
        v = SIGN[m.group(1)] * float(m.group(2))
        return (f"{basis}_pct", v, v, m.span())
    if re.search(r"\b(?:approximately |roughly |about |relatively )?flat\b", sl) and \
            re.search(r"sequential|prior quarter|previous quarter|preceding quarter|from the (first|second|third|fourth) quarter", sl):
        m = re.search(r"flat", sl)
        return (f"{basis}_pct", 0.0, 0.0, m.span())
    return None


GM_P = r"(\d{1,2}(?:\.\d+)?)%"


def _gm_width(w, u) -> float:
    if w is None:
        return 0.0
    w = float(w)
    return w / 100.0 if (u or "").lower() == "bps" else w


def parse_gm(s: str) -> list[tuple]:
    """(basis, kind, low, high) in percent from one normalised segment."""
    sl = s.lower()
    out = []
    both = bool(re.search(r"gaap and non-gaap|gaap and adjusted", sl))
    nongaap = bool(re.search(r"non-gaap|non gaap|adjusted gross", sl))
    gaap = bool(re.search(r"(?<!non-)(?<!non )\bgaap\b", sl))
    m = re.search(rf"{GM_P}\s*and\s*{GM_P}\s*,?\s*respectively\s*,?\s*(?:±\s*{N}\s*(bps|%|pp))?", sl)
    if m and both:
        w = _gm_width(m.group(3), m.group(4))
        a, b = float(m.group(1)), float(m.group(2))
        kind = "pm" if w else "point"
        return [("gaap", kind, a - w, a + w), ("nongaap", kind, b - w, b + w)]
    if both:
        basis = "both"
    elif nongaap and gaap:
        basis = "mixed"
    elif nongaap:
        basis = "nongaap"
    elif gaap:
        basis = "gaap"
    else:
        basis = "unspecified"
    for m in re.finditer(rf"{GM_P}\s*,?\s*±\s*{N}\s*(bps|%|pp)", sl):
        c, w = float(m.group(1)), _gm_width(m.group(2), m.group(3))
        out.append((basis, "pm", c - w, c + w))
        break
    if not out:
        m = re.search(rf"(\d{{1,2}}(?:\.\d+)?)%?\s*(?:to|and|-)\s*{GM_P}", sl)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            if b >= a and b - a <= 15:
                out.append((basis, "range", a, b))
    if not out:
        m = re.search(rf"(?:approximately|about|roughly|around|of|be|~|at)\s*{GM_P}", sl)
        if m:
            c = float(m.group(1))
            out.append((basis, "point", c, c))
    if basis == "both":
        out = [(b, k, lo, hi) for (_, k, lo, hi) in out for b in ("gaap", "nongaap")]
    return out


COMPARISON = re.compile(r"\b(?:up|down|ppts?|pts?|change|vs\.?|versus|compared|increase|decrease|prior)\b", re.I)
PCT = r"(\d{1,2}(?:\.\d+)?)\s*%"


def parse_table_gm(seg_raw: str, ctx: str) -> list[tuple]:
    """Gross-margin row of an outlook table, parsed cell by cell -> [(basis, kind, low, high)].

    Cells: "45.2%" "+/-" "1%"  (centre, sign, width)     "41.0% ± 1.0%"   (pm in one cell)
           "36.0% to 38.0%" / "47% 51%" (range; a lost dash leaves two percentages in one cell)
           "22" "Million" / "0.2%"  (reconciliation columns: skipped)
    Bases: U.S. GAAP and Non-GAAP columns -> first estimate GAAP, last non-GAAP; Low | Mid | High columns -> triples.
    Rows comparing periods ("45.7% | 39.2% | up 6.5 ppts") or stating results ("was 21.6%") are rejected."""
    if PAST.search(seg_raw) or COMPARISON.search(seg_raw):
        return []
    cl = ctx.lower()
    has_gaap = "gaap" in cl.replace("non-gaap", "")
    has_ng = "non-gaap" in cl
    cells = [c.strip() for c in seg_raw.split("|")]
    label_idx = next((i for i, c in enumerate(cells) if GM_SUBJ.search(c)), 0)
    cells = [_norm(c) for c in cells[label_idx + 1:]]
    if any(re.fullmatch(r"[\d,.]+\s*bps", c.lower()) for c in cells):
        return []                                   # "1,410bps | 950bps": period-over-period change columns
    est = []
    i = 0
    while i < len(cells):
        c = cells[i].lower()
        m_rng = re.fullmatch(rf"{PCT}\s*(?:to|-|and)?\s*{PCT}", c)
        m_pm = re.fullmatch(rf"{PCT}\s*±\s*{N}\s*(bps|%|pp)?", c)
        m_pt = re.fullmatch(PCT, c)
        if m_rng:
            a, b = float(m_rng.group(1)), float(m_rng.group(2))
            if b >= a:
                est.append(("range", a, b))
        elif m_pm:
            ctr, w = float(m_pm.group(1)), _gm_width(m_pm.group(2), m_pm.group(3) or "%")
            est.append(("pm", ctr - w, ctr + w))
        elif m_pt and float(m_pt.group(1)) > 5:
            ctr = float(m_pt.group(1))
            if i + 2 < len(cells) and cells[i + 1].strip() == "±":
                mw = re.fullmatch(rf"{N}\s*(bps|%|pp)?", cells[i + 2].lower())
                if mw:
                    w = _gm_width(mw.group(1), mw.group(2) or "%")
                    est.append(("pm", ctr - w, ctr + w))
                    i += 3
                    continue
            est.append(("point", ctr, ctr))
        i += 1
    if not est:
        return []
    low_mid_high = bool(re.search(r"\blow\b.*\bhigh\b", ctx, re.I | re.S))
    if any(k == "point" for k, *_ in est) and not low_mid_high:
        # outlook tables carry at most a GAAP and a non-GAAP figure; more columns, or a dollar amount ahead of the
        # first percentage ("$689.3 | 48.1% | $933.7 | 61.7%"), mean a table of reported results
        first_pct = next(i for i, c in enumerate(cells) if re.search(PCT, c))
        if len(est) > 2 or any("$" in c for c in cells[:first_pct]):
            return []
    if low_mid_high and all(k == "point" for k, *_ in est):
        pts = [lo for _, lo, _ in est]
        if len(pts) == 6 and has_gaap and has_ng:
            return [("gaap", "range", pts[0], pts[2]), ("nongaap", "range", pts[3], pts[5])]
        if len(pts) == 3:
            b = "nongaap" if has_ng and not has_gaap else ("gaap" if has_gaap and not has_ng else "unspecified")
            return [(b, "range", pts[0], pts[2])]
        return []
    if has_gaap and has_ng:
        if len(est) >= 2:
            return [("gaap",) + est[0], ("nongaap",) + est[-1]]
        return [("unspecified",) + est[0]]
    basis = "nongaap" if has_ng else ("gaap" if has_gaap else "unspecified")
    return [(basis,) + est[0]]


def segments(text: str) -> list[tuple[str, str, str]]:
    """(kind, segment, context) units: prose sentences and table rows; context = preceding lines."""
    lines = text.split("\n")
    out: list[tuple[str, str, str]] = []
    para: list[str] = []
    recent: list[str] = []

    def flush_para():
        if not para:
            return
        p = " ".join(para)
        # bullets ("* Total revenues ... * GAAP gross margin ...") are separate statements even without periods
        for item in re.split(rf"\s+(?=[{re.escape(BULLETS)}]\s)", p):
            for sent in re.split(rf"(?<=[.;])\s+(?=[A-Z(\d{re.escape(BULLETS)}])", item):
                if len(sent) > 15:
                    out.append(("prose", sent, " // ".join(recent[-6:])))
        para.clear()

    for ln in lines:
        if " | " in ln:
            flush_para()
            out.append(("table", ln, " // ".join(recent[-10:])))
            recent.append(ln)
            continue
        if not ln.strip():
            flush_para()
            continue
        para.append(ln)
        recent.append(ln)
    flush_para()
    return out


def _sentence_like(ns: str) -> bool:
    return bool(FWD_STRONG.search(ns)) and not PAST.search(ns)


def extract(docs: list[dict], base_rev: float | None, prev_gm: float | None = None):
    """Revenue and gross-margin guidance candidates from one release's EX-99 documents."""
    revs: list[Guide] = []
    gms: list[Guide] = []
    for d in docs:
        typ = d.get("type", "")
        for pos, (kind, seg, ctx) in enumerate(segments(d.get("text", ""))):
            ns = _norm(seg)
            nl = ns.lower()
            prose_ok = _sentence_like(ns)
            in_outlook = bool(OUTLOOK_HDR.search(ctx)) or bool(OUTLOOK_HDR.search(seg))
            annual_only = bool(ANNUAL.search(nl)) and not QTR.search(nl)
            hint = unit_hint(seg) or (unit_hint(ctx) if kind == "table" else None)
            # ------------------------------------------------------------------ revenue
            subj = REV_SUBJ.search(ns)
            if subj and not GM_SUBJ.search(nl[: subj.start() + 1]) and not annual_only:
                row_ok = kind == "table" and bool(OUTLOOK_TABLE.search(ctx)) and not PAST.search(ns) and nl.startswith(
                    ("revenue", "net revenue", "total revenue", "revenues", "net sales", "sales", "total net revenue",
                     "net revenues", "total net sales", "total revenues", "quarterly revenue"))
                # a bullet or row inside an explicit outlook section that leads with revenue
                # ("Quarterly revenue in the range of $660 million to $680 million")
                ctx_ok = subj.start() <= 40 and not PAST.search(ns) and \
                    bool(OUTLOOK_STRICT.search(ctx) or OUTLOOK_STRICT.search(seg))
                if prose_ok or row_ok or ctx_ok:
                    # verb-less bullets/rows only count with an explicit range or +/- (never a point: highlight
                    # bullets such as "Revenue of $9,295 million for the fourth quarter" are last quarter's actuals)
                    found = parse_revenue(ns[subj.start():], base_rev, hint,
                                          allow_point=prose_ok or row_ok,
                                          allow_pct=prose_ok and bool(STRICT_FWD.search(ns)))
                    if not found and kind == "table" and row_ok:
                        found = parse_table_revenue(ns[subj.start():], base_rev, hint, ctx)
                    for k, lo, hi, _ in found:
                        if k in ("seq_pct", "yoy_pct"):
                            if base_rev is None or abs(lo) > 60 or abs(hi) > 60:
                                continue
                            g = Guide("revenue", k[:3], k, lo, hi, (lo + hi) / 2, seg[:500], typ, kind, pos)
                        else:
                            mid = (lo + hi) / 2
                            if base_rev and not (0.4 <= mid / base_rev <= 2.5):
                                continue
                            if hi < lo or (hi - lo) / max(mid, 1.0) > 0.5:
                                continue
                            g = Guide("revenue", "total", k, lo, hi, mid, seg[:500], typ, kind, pos)
                        pre = nl[max(0, subj.start() - 50): subj.end() + 3]
                        if SEGMENT_HINT.search(pre):
                            g.basis = "segment?" if g.basis == "total" else g.basis + "_segment?"
                        if re.search(r"discontinued|combined revenue|pro forma", nl):
                            g.basis = "incl_discontinued"      # not comparable with continuing-ops XBRL revenue
                        revs.append(g)
            # ------------------------------------------------------------------ gross margin
            gmm = GM_SUBJ.search(ns)
            if gmm and not (bool(ANNUAL.search(nl)) and not QTR.search(nl)) and \
                    not re.search(r"long[- ]term|target model|business model", nl):
                if prose_ok or (kind == "table" and bool(OUTLOOK_TABLE.search(ctx))):
                    if prose_ok:
                        cands = parse_gm(ns)
                    else:
                        cands = parse_table_gm(seg, ctx)
                    for basis, k, lo, hi in cands:
                        mid = (lo + hi) / 2
                        if not (5 <= mid <= 95) or hi - lo > 15:
                            continue
                        if prev_gm is not None and abs(mid - prev_gm) > 25:
                            continue
                        gms.append(Guide("gm", basis, k, lo, hi, mid, seg[:500], typ, kind, pos))
    return revs, gms


def choose_revenue(cands: list[Guide]) -> Guide | None:
    """Prefer total-company guidance from EX-99.1, explicit ranges over points, then the earliest statement in the
    document (the outlook paragraph or table comes before segment colour and reconciliations)."""
    if not cands:
        return None
    return max(cands, key=lambda g: (g.basis in ("total", "seq", "yoy"), g.doc_type == "EX-99.1",
                                     g.kind != "point", -g.pos))


def choose_gm(cands: list[Guide]) -> dict[str, Guide]:
    out = {}
    for basis in ("gaap", "nongaap", "unspecified", "mixed"):
        c = [g for g in cands if g.basis == basis]
        if c:
            out[basis] = sorted(c, key=lambda g: (g.doc_type != "EX-99.1", g.kind == "point"))[0]
    return out


def as_dict(g: Guide | None) -> dict:
    return asdict(g) if g else {}

