"""
문경 마성면 모곡리 소규모 유역 — 수리·수문 계산 엔진
  * 합리식(Rational) 첨두홍수량 : 기준값(검증용)
  * EPA SWMM 5.2 강우-유출 모의 : 주 계산엔진 (swmm-toolkit)

모든 기본값의 출처
  - 유역면적·토지이용·유로연장·표고 : 사용자 측량/설계 자료 (DXF, 엑셀)
  - 확률강우강도식 : WAMIS 강우강도식_IDF_General형 (gev, 지역 11, 문경 20051273)
  - Huff 3분위 : WAMIS 강우시간분포(Huff) — 지역 11, 초과확률 50%, 6차 다항식
  - 유출곡선지수(CN-II) : 홍수량 산정 표준지침(환경부, 2019) <표 3.1>
  - 수문학적 토양군 : WAMIS 유출곡선지수 자료(2000_낙동강_CN, 낙동강국가3) 중첩
"""
from __future__ import annotations

import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict

import numpy as np

# --------------------------------------------------------------------------
# 1. 기본 자료
# --------------------------------------------------------------------------
POINTS = ["D", "C", "B", "A"]          # 상류 → 하류 (유역이 D ⊂ C ⊂ B ⊂ A 로 포개짐)
LANDUSE = ["밭", "임야", "도로", "초지", "나지"]

# 지점별 누가 토지이용 면적 (㎡)
LANDUSE_CUM = {
    "D": [11211, 9841, 12862, 5038, 449],
    "C": [21707, 16401, 14315, 5122, 8081],
    "B": [36064, 21187, 14315, 5235, 7981],
    "A": [57322, 28273, 14315, 5350, 7881],
}
FLOW_LENGTH = {"D": 502.53, "C": 644.63, "B": 898.31, "A": 1213.37}   # 유로연장 (m)
Z_TOP = 350.0                                                          # 최원점 표고 (m)
Z_OUT = {"D": 247.0, "C": 222.5, "B": 183.0, "A": 149.0}              # 출구 표고 (m)
OVERLAND_LEN = 30.0      # 표면류 길이 (m)
OVERLAND_N = 0.6         # Kerby 지체계수
OVERLAND_SLOPE = 0.25    # 사면경사 (m/m)

# 합리식 유출계수
C_RATIONAL = {"밭": 0.6, "임야": 0.8, "도로": 0.95, "초지": 0.45, "나지": 0.5}

# WAMIS 확률강우강도식 General형 I = a / (t^n + b)   [I: mm/hr, t: 분]
IDF = {
    2: (758.72, 6.621833, 0.6765068),
    3: (866.4477, 6.75537, 0.6716904),
    5: (992.1321, 7.0839, 0.6682905),
    10: (1159.306, 7.688572, 0.6659747),
    20: (1330.695, 8.43107, 0.6651977),
    30: (1434.855, 8.923926, 0.6652471),
    50: (1571.504, 9.606532, 0.6657394),
    70: (1665.316, 10.09439, 0.666293),
    80: (1703.468, 10.29664, 0.666559),
    100: (1768.466, 10.64587, 0.6670597),
    200: (1981.237, 11.82467, 0.6690366),
    300: (2114.093, 12.58365, 0.6704661),
    500: (2291.36, 13.61857, 0.6725338),
}

# WAMIS Huff 3분위 (지역 11, 초과확률 50%) 6차 다항식 : x=시간(%), y=누가강우(%)
HUFF3 = [4.7181372548995003e-9, -1.3870757918544087e-6, 1.4697445324273725e-4,
         -6.8267058309280093e-3, 0.14603848724788765, -0.78672583298889809,
         0.12538050185014527]
HUFF3_TABLE = [0, 2, 6.5, 12.8, 19.8, 30.7, 50, 72.4, 88.2, 95.3, 100]

# 표준지침 <표 3.1> CN-II (토양군 A, B, C, D)
CN2_TABLE = {
    "밭": (63, 74, 82, 85),      # 밭(220)
    "임야": (55, 72, 82, 85),    # 산림(310~330)
    "초지": (49, 69, 79, 84),    # 기타초지(430)
    "나지": (77, 86, 91, 94),    # 기타나지(620)
    "도로": (83, 89, 92, 93),    # 교통지역(150) — SWMM에서는 불투수면으로 처리
}
# 지점별 누가 토양군 면적 (㎡) — WAMIS CN 자료 중첩 결과
HSG_CUM = {
    "D": {"A": 10602, "B": 0, "C": 28750, "D": 0},
    "C": {"A": 10607, "B": 0, "C": 54990, "D": 0},
    "B": {"A": 15799, "B": 0, "C": 68923, "D": 0},
    "A": {"A": 28468, "B": 0, "C": 83523, "D": 1092},
}
# SWMM 표면 조도계수 (SWMM 사용자 매뉴얼 표)
N_PERV = {"밭": 0.17, "임야": 0.40, "초지": 0.24, "나지": 0.05}


# --------------------------------------------------------------------------
# 2. 모의 조건
# --------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str = "기본"
    return_period: int = 30
    duration: int = 45                 # 강우 지속기간 (분)
    storm: str = "huff3"               # "huff3" | "block"(교호블록)
    amc: str = "III"                   # "II" | "III"
    landuse_cum: dict = field(default_factory=lambda: {k: list(v) for k, v in LANDUSE_CUM.items()})
    develop_pct: float = 0.0           # 밭·초지·나지 중 불투수(대지·도로)로 바뀌는 비율 (%)
    n_mult: float = 1.0                # 조도계수 배율 (사면·하도)
    ch_n: float = 0.040                # 하도 Manning n
    ch_width: float = 1.0              # 하도 바닥폭 (m)
    ch_depth: float = 2.0              # 하도 깊이 (m)
    ch_side: float = 1.0               # 하도 측면경사 (1:z)
    imp_n: float = 0.013
    ds_imp: float = 1.5                # 불투수 저류깊이 (mm)
    ds_perv: float = 5.0               # 투수 저류깊이 (mm)

    def to_dict(self):
        return asdict(self)


def effective_landuse(sc: Scenario) -> dict:
    """개발 시나리오를 적용한 누가 토지이용 (도로=불투수로 합산)."""
    f = sc.develop_pct / 100.0
    out = {}
    for k in POINTS:
        a = list(map(float, sc.landuse_cum[k]))
        moved = 0.0
        for i, nm in enumerate(LANDUSE):
            if nm in ("밭", "초지", "나지"):
                d = a[i] * f
                a[i] -= d
                moved += d
        a[LANDUSE.index("도로")] += moved
        out[k] = a
    return out


def incremental(cum: dict) -> dict:
    """누가 면적 → 지점 사이 소유역(증분) 면적."""
    inc, prev = {}, [0.0] * len(next(iter(cum.values())))
    for k in POINTS:
        inc[k] = [max(0.0, c - p) for c, p in zip(cum[k], prev)]
        prev = cum[k]
    return inc


def hsg_incremental() -> dict:
    inc, prev = {}, {g: 0 for g in "ABCD"}
    for k in POINTS:
        inc[k] = {g: max(0, HSG_CUM[k][g] - prev[g]) for g in "ABCD"}
        prev = HSG_CUM[k]
    return inc


def cn_amc(cn2: float, amc: str) -> float:
    if amc == "III":
        return 23 * cn2 / (10 + 0.13 * cn2)
    if amc == "I":
        return 4.2 * cn2 / (10 - 0.058 * cn2)
    return cn2


# --------------------------------------------------------------------------
# 3. 합리식 (검증 기준)
# --------------------------------------------------------------------------
def kerby_minutes(L=OVERLAND_LEN, N=OVERLAND_N, S=OVERLAND_SLOPE) -> float:
    return 1.44 * (L * N / math.sqrt(S)) ** 0.467


def kraven_velocity(S: float) -> float:
    if S > 3 / 400:
        return min(4.5, 4.592 - 0.01194 / S)
    return max(1.6, 35151.515 * S ** 2 - 79.393939 * S + 1.6181818)


def idf_intensity(T: int, t_min: float) -> float:
    a, b, n = IDF[T]
    return a / (t_min ** n + b)


def rational(sc: Scenario | None = None, T: int | None = None) -> dict:
    sc = sc or Scenario()
    T = T or sc.return_period
    lu = effective_landuse(sc)
    out = {}
    for k in POINTS:
        areas = lu[k]
        A = sum(areas)
        C = sum(C_RATIONAL[nm] * a for nm, a in zip(LANDUSE, areas)) / A
        S = (Z_TOP - Z_OUT[k]) / FLOW_LENGTH[k]
        t1 = kerby_minutes()
        V = kraven_velocity(S)
        t2 = FLOW_LENGTH[k] / (60 * V)
        tc = t1 + t2
        I = idf_intensity(T, tc)
        Q = 0.2778 * C * I * A / 1e6
        out[k] = dict(area_km2=A / 1e6, C=C, slope=S, t_inlet=t1, velocity=V, t_travel=t2,
                      tc=tc, I=I, Q=Q)
    return out


# --------------------------------------------------------------------------
# 4. 설계강우
# --------------------------------------------------------------------------
def design_depth(T: int, dur: int) -> float:
    return idf_intensity(T, dur) * dur / 60.0


def huff_hyetograph(T: int, dur: int, dt: int = 1) -> np.ndarray:
    P = design_depth(T, dur)
    x = np.linspace(0, 100, int(dur / dt) + 1)
    y = np.polyval(HUFF3, x)
    y = (y - y[0]) / (y[-1] - y[0])
    y = np.maximum.accumulate(np.clip(y, 0, 1))
    return np.diff(y) * P * 60.0 / dt           # mm/hr


def block_hyetograph(T: int, dur: int, dt: int = 1) -> np.ndarray:
    ts = np.arange(dt, dur + dt, dt)
    depth = np.array([idf_intensity(T, t) * t / 60.0 for t in ts])
    inc = np.diff(np.r_[0.0, depth])
    m = len(inc)
    c = (m - 1) // 2
    order = [c]
    for i in range(1, m):
        j = c + ((i + 1) // 2) * (1 if i % 2 else -1)
        if 0 <= j < m:
            order.append(j)
    order += [i for i in range(m) if i not in order]
    blocks = np.zeros(m)
    for d, pos in zip(inc, order):
        blocks[pos] = d
    return blocks * 60.0 / dt


def hyetograph(sc: Scenario) -> np.ndarray:
    if sc.storm == "block":
        return block_hyetograph(sc.return_period, sc.duration)
    return huff_hyetograph(sc.return_period, sc.duration)


# --------------------------------------------------------------------------
# 5. SWMM 입력파일
# --------------------------------------------------------------------------
def subcatchments(sc: Scenario) -> dict:
    lu_inc = incremental(effective_landuse(sc))
    hs = hsg_incremental()
    out = {}
    for k in POINTS:
        a = lu_inc[k]
        tot = sum(a)
        imp = a[LANDUSE.index("도로")]
        perv = {nm: v for nm, v in zip(LANDUSE, a) if nm != "도로"}
        pa = sum(perv.values()) or 1.0
        htot = sum(hs[k].values()) or 1.0
        fr = {g: v / htot for g, v in hs[k].items()}
        cn2 = sum(v * sum(fr[g] * CN2_TABLE[nm]["ABCD".index(g)] for g in "ABCD")
                  for nm, v in perv.items()) / pa
        npv = sum(N_PERV[nm] * v for nm, v in perv.items()) / pa
        out[k] = dict(area_ha=tot / 1e4, imp_pct=100 * imp / tot, width=tot / OVERLAND_LEN,
                      slope_pct=OVERLAND_SLOPE * 100, cn2=cn2, cn=cn_amc(cn2, sc.amc),
                      n_perv=npv * sc.n_mult, hsg=fr)
    return out


def build_inp(sc: Scenario, rain: np.ndarray | None = None) -> str:
    rain = hyetograph(sc) if rain is None else rain
    subs = subcatchments(sc)
    total_min = sc.duration + 180
    end_h, end_m = divmod(min(total_min, 24 * 60 - 1), 60)
    L = [
        "[TITLE]", f"Mungyeong Mogok-ri | {sc.name} | T={sc.return_period}yr D={sc.duration}min", "",
        "[OPTIONS]", "FLOW_UNITS CMS", "INFILTRATION CURVE_NUMBER", "FLOW_ROUTING KINWAVE",
        "START_DATE 01/01/2026", "START_TIME 00:00:00",
        "REPORT_START_DATE 01/01/2026", "REPORT_START_TIME 00:00:00",
        "END_DATE 01/01/2026", f"END_TIME {end_h:02d}:{end_m:02d}:00",
        "DRY_STEP 00:05:00", "WET_STEP 00:00:15", "ROUTING_STEP 0:00:05",
        "REPORT_STEP 00:01:00", "ALLOW_PONDING NO", "",
        "[RAINGAGES]", "RG1 INTENSITY 0:01 1.0 TIMESERIES DESIGN", "",
        "[JUNCTIONS]",
    ]
    zh = Z_TOP - (Z_TOP - Z_OUT["D"]) / FLOW_LENGTH["D"] * OVERLAND_LEN
    L.append(f"J_Dhead {zh:.2f} 3 0 0 0")
    for k in ("D", "C", "B"):
        L.append(f"J_{k} {Z_OUT[k]:.2f} 3 0 0 0")
    L += ["", "[OUTFALLS]", f"O_A {Z_OUT['A']:.2f} FREE NO", "", "[CONDUITS]"]
    links = [("L_D", "J_Dhead", "J_D", FLOW_LENGTH["D"] - OVERLAND_LEN),
             ("L_C", "J_D", "J_C", FLOW_LENGTH["C"] - FLOW_LENGTH["D"]),
             ("L_B", "J_C", "J_B", FLOW_LENGTH["B"] - FLOW_LENGTH["C"]),
             ("L_A", "J_B", "O_A", FLOW_LENGTH["A"] - FLOW_LENGTH["B"])]
    chn = sc.ch_n * sc.n_mult
    for nm, a, b, l in links:
        L.append(f"{nm} {a} {b} {l:.2f} {chn:.4f} 0 0 0 0")
    L += ["", "[XSECTIONS]"]
    for nm, *_ in links:
        L.append(f"{nm} TRAPEZOIDAL {sc.ch_depth} {sc.ch_width} {sc.ch_side} {sc.ch_side} 1")
    outlet = {"D": "J_Dhead", "C": "J_D", "B": "J_C", "A": "J_B"}
    L += ["", "[SUBCATCHMENTS]"]
    for k in POINTS:
        s = subs[k]
        L.append(f"S_{k} RG1 {outlet[k]} {s['area_ha']:.4f} {s['imp_pct']:.3f} "
                 f"{s['width']:.1f} {s['slope_pct']:.1f} 0")
    L += ["", "[SUBAREAS]"]
    for k in POINTS:
        L.append(f"S_{k} {sc.imp_n * sc.n_mult:.4f} {subs[k]['n_perv']:.4f} "
                 f"{sc.ds_imp} {sc.ds_perv} 0 OUTLET")
    L += ["", "[INFILTRATION]"]
    for k in POINTS:
        L.append(f"S_{k} {subs[k]['cn']:.3f} 0.5 7")
    L += ["", "[TIMESERIES]"]
    for i, v in enumerate(rain):
        L.append(f"DESIGN {i // 60:02d}:{i % 60:02d} {v:.4f}")
    L.append(f"DESIGN {len(rain) // 60:02d}:{len(rain) % 60:02d} 0")
    L += ["", "[REPORT]", "INPUT NO", "CONTROLS NO", "SUBCATCHMENTS ALL", "NODES ALL", "LINKS ALL", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------
# 6. SWMM 실행 및 결과 읽기
# --------------------------------------------------------------------------
def _run_engine(inp: str, rpt: str, out: str) -> str:
    """EPA SWMM 실행: swmm-toolkit(파이썬 바인딩) 우선, 없으면 로컬 runswmm."""
    try:
        from swmm.toolkit import solver  # type: ignore
        if hasattr(solver, "swmm_run"):
            solver.swmm_run(inp, rpt, out)
        else:  # 구버전 바인딩: 단계별 실행
            solver.swmm_open(inp, rpt, out)
            solver.swmm_start(1)
            while True:
                if solver.swmm_step() <= 0:
                    break
            solver.swmm_end()
            solver.swmm_report()
            solver.swmm_close()
        return "swmm-toolkit " + _toolkit_version()
    except ImportError:
        exe = os.environ.get("RUNSWMM") or shutil.which("runswmm")
        if not exe:
            raise RuntimeError("SWMM 엔진을 찾을 수 없습니다 (swmm-toolkit 설치 필요).")
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = os.path.dirname(exe) + ":" + env.get("LD_LIBRARY_PATH", "")
        subprocess.run([exe, inp, rpt, out], capture_output=True, env=env, check=False)
        return "runswmm (local)"


def _toolkit_version() -> str:
    try:
        from importlib.metadata import version
        return version("swmm-toolkit")
    except Exception:
        return ""


def read_out(path: str):
    """SWMM 바이너리 결과(.out) → 시간(분), 하도유량{링크:배열}, 강우강도."""
    b = open(path, "rb").read()
    idpos, inpos, outpos, nper, err, magic = struct.unpack("6i", b[-24:])
    _, ver, fu, nS, nN, nL, nP = struct.unpack("7i", b[:28])
    p = idpos
    names = []
    for _ in range(nS + nN + nL + nP):
        n = struct.unpack("i", b[p:p + 4])[0]
        names.append(b[p + 4:p + 4 + n].decode())
        p += 4 + n
    links = names[nS + nN:nS + nN + nL]
    p = inpos
    k = struct.unpack("i", b[p:p + 4])[0]; p += 4 + 4 * k + 4 * nS
    k = struct.unpack("i", b[p:p + 4])[0]; p += 4 + 4 * k + 4 * k * nN
    k = struct.unpack("i", b[p:p + 4])[0]; p += 4 + 4 * k + 4 * k * nL
    nv = []
    for _ in range(4):
        k = struct.unpack("i", b[p:p + 4])[0]; nv.append(k); p += 4 + 4 * k
    step = struct.unpack("i", b[p + 8:p + 12])[0]
    nsv, nnv, nlv, nsys = nv
    rec = 8 + 4 * (nS * nsv + nN * nnv + nL * nlv + nsys)
    t = np.arange(1, nper + 1) * step / 60.0
    flow = {l: np.empty(nper) for l in links}
    rain = np.empty(nper)
    for i in range(nper):
        q = outpos + i * rec + 8
        sub = np.frombuffer(b[q:q + 4 * nS * nsv], dtype="<f4").reshape(nS, nsv)
        q += 4 * nS * nsv + 4 * nN * nnv
        lk = np.frombuffer(b[q:q + 4 * nL * nlv], dtype="<f4").reshape(nL, nlv)
        for j, l in enumerate(links):
            flow[l][i] = lk[j, 0]
        rain[i] = sub[0, 0]
    return t, flow, rain


def _parse_report(txt: str) -> dict:
    out = {"errors": re.findall(r"ERROR \d+:.*", txt)}
    m = re.findall(r"Continuity Error \(%\) \.+\s+(-?[\d.]+)", txt)
    if m:
        out["runoff_continuity"] = float(m[0])
        out["routing_continuity"] = float(m[-1])
    m = re.search(r"Total Precipitation\s+\.+\s+[\d.]+\s+([\d.]+)", txt)
    out["precip_mm"] = float(m.group(1)) if m else None
    m = re.search(r"Surface Runoff\s+\.+\s+[\d.]+\s+([\d.]+)", txt)
    out["runoff_mm"] = float(m.group(1)) if m else None
    m = re.search(r"Infiltration Loss\s+\.+\s+[\d.]+\s+([\d.]+)", txt)
    out["infil_mm"] = float(m.group(1)) if m else None
    return out


@dataclass
class Result:
    scenario: dict
    t: np.ndarray
    flow: dict                # {"A": array, ...}  (㎥/s)
    rain: np.ndarray          # mm/hr
    peaks: dict               # {"A": {"Q":, "t":, "volume_m3":}}
    report: dict
    engine: str
    inp_text: str
    rpt_text: str
    subcatchments: dict


def run(sc: Scenario) -> Result:
    inp_text = build_inp(sc)
    with tempfile.TemporaryDirectory() as d:
        inp, rpt, out = (os.path.join(d, f"model.{e}") for e in ("inp", "rpt", "out"))
        with open(inp, "w") as f:
            f.write(inp_text)
        engine = _run_engine(inp, rpt, out)
        rpt_text = open(rpt).read() if os.path.exists(rpt) else ""
        rep = _parse_report(rpt_text)
        if rep["errors"] or not os.path.exists(out) or os.path.getsize(out) < 100:
            raise RuntimeError("SWMM 실행 오류: " + "; ".join(rep["errors"] or ["결과 파일 없음"]))
        t, flow_l, rain = read_out(out)
    flow = {k: flow_l["L_" + k] for k in POINTS}
    # 첨두유량·발생시각은 SWMM 보고서(Link Flow Summary, 계산시간간격 기준) 값 사용
    rpt_peaks = {}
    if "Link Flow Summary" in rpt_text:
        sec = rpt_text.split("Link Flow Summary")[1]
        for m in re.finditer(r"^\s+L_([DCBA])\s+CONDUIT\s+([\d.]+)\s+(\d+)\s+(\d\d):(\d\d)", sec, re.M):
            rpt_peaks[m.group(1)] = (float(m.group(2)), int(m.group(4)) * 60 + int(m.group(5)))
    peaks = {}
    for k in POINTS:
        q = flow[k]
        i = int(np.argmax(q))
        Q, tp = rpt_peaks.get(k, (float(q[i]), float(t[i])))
        peaks[k] = dict(Q=Q, t=float(tp),
                        volume_m3=float(np.sum((q[1:] + q[:-1]) / 2 * np.diff(t * 60))))
    return Result(sc.to_dict(), t, flow, rain, peaks, rep, engine, inp_text, rpt_text, subcatchments(sc))


def critical_duration(sc: Scenario, durations=None, point="A"):
    """지속기간을 바꿔가며 SWMM을 반복 실행해 첨두유량이 최대인 지속기간을 찾는다."""
    durations = durations or [20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 120, 180]
    rows = []
    for dur in durations:
        s = Scenario(**{**sc.to_dict(), "duration": dur})
        r = run(s)
        rows.append({"duration": dur, **{k: r.peaks[k]["Q"] for k in POINTS}})
    best = max(rows, key=lambda r: r[point])
    return rows, best["duration"]
