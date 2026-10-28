"""
소규모 유역 강우-유출 모의 웹앱 (EPA SWMM + 합리식 검증)
대상: 경북 문경시 마성면 모곡리 소규모 유역 4개 지점 (D ⊂ C ⊂ B ⊂ A)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from plotly.subplots import make_subplots

from core import hydro as H

st.set_page_config(page_title="소규모 유역 SWMM 모의", page_icon="🌧️", layout="wide")

COLORS = {"A": "#d62728", "B": "#ff7f0e", "C": "#2ca02c", "D": "#9467bd"}
RGB = {"A": [214, 39, 40], "B": [255, 127, 14], "C": [44, 160, 44], "D": [148, 103, 189]}
DATA = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# 캐시된 계산 함수 (같은 조건이면 다시 계산하지 않음)
# ---------------------------------------------------------------------------
def _sc_from_json(s: str) -> H.Scenario:
    return H.Scenario(**json.loads(s))


@st.cache_data(show_spinner=False)
def cached_run(sc_json: str) -> dict:
    r = H.run(_sc_from_json(sc_json))
    return dict(scenario=r.scenario, t=r.t, flow=r.flow, rain=r.rain, peaks=r.peaks,
                report=r.report, engine=r.engine, inp=r.inp_text, rpt=r.rpt_text,
                subs=r.subcatchments)


@st.cache_data(show_spinner=False)
def cached_critical(sc_json: str, point: str) -> tuple:
    rows, best = H.critical_duration(_sc_from_json(sc_json), point=point)
    return rows, best


@st.cache_data(show_spinner=False)
def cached_rational(sc_json: str, T: int) -> dict:
    return H.rational(_sc_from_json(sc_json), T=T)


@st.cache_data
def load_geo() -> dict:
    return json.loads((DATA / "watersheds.geojson").read_text(encoding="utf-8"))


def sc_json(sc: H.Scenario) -> str:
    return json.dumps(sc.to_dict(), ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------
# 사이드바 : 모의 조건 설정
# ---------------------------------------------------------------------------
st.sidebar.header("① 모의 조건 설정")
name = st.sidebar.text_input("시나리오 이름", value="기본(30년)")
T = st.sidebar.selectbox("재현기간 (년)", list(H.IDF.keys()), index=list(H.IDF.keys()).index(30))
storm_label = st.sidebar.radio("강우 시간분포", ["Huff 3분위 (표준지침)", "교호블록"], index=0)
storm = "huff3" if storm_label.startswith("Huff") else "block"
auto_dur = st.sidebar.checkbox("임계지속기간 자동 탐색 (지점 A 기준)", value=True)
dur = st.sidebar.slider("강우 지속기간 (분)", 10, 360, 45, 5, disabled=auto_dur)
amc_label = st.sidebar.radio("선행토양함수조건", ["AMC-III (표준지침 설계조건)", "AMC-II (보통)"], index=0)
amc = "III" if amc_label.startswith("AMC-III") else "II"
develop = st.sidebar.slider("개발 시나리오: 밭·초지·나지 → 불투수 전환 (%)", 0, 100, 0, 5)
n_mult = st.sidebar.slider("조도계수 배율 (사면·하도)", 0.5, 2.0, 1.0, 0.1)

with st.sidebar.expander("하도 단면 (사다리꼴)"):
    ch_n = st.number_input("하도 Manning n", 0.010, 0.150, 0.040, 0.005, format="%.3f")
    ch_w = st.number_input("바닥폭 (m)", 0.3, 10.0, 1.0, 0.1)
    ch_d = st.number_input("깊이 (m)", 0.3, 5.0, 2.0, 0.1)

with st.sidebar.expander("토지이용 면적 편집 (㎡, 지점별 누가)"):
    lu_df = pd.DataFrame(H.LANDUSE_CUM, index=H.LANDUSE).T.loc[H.POINTS]
    lu_edit = st.data_editor(lu_df, use_container_width=True, key="lu_editor")
    lu_cum = {k: [float(lu_edit.loc[k, c]) for c in H.LANDUSE] for k in H.POINTS}
    nest_bad = [f"{H.POINTS[i+1]}<{H.POINTS[i]} ({c})" for i in range(3) for j, c in enumerate(H.LANDUSE)
                if lu_cum[H.POINTS[i + 1]][j] < lu_cum[H.POINTS[i]][j] - 1e-6]
    if nest_bad:
        st.caption("※ 하류 지점 면적이 상류보다 작은 항목은 증분 면적 0으로 처리: " + ", ".join(nest_bad))

sc = H.Scenario(name=name, return_period=int(T), duration=int(dur), storm=storm, amc=amc,
                landuse_cum=lu_cum, develop_pct=float(develop), n_mult=float(n_mult),
                ch_n=float(ch_n), ch_width=float(ch_w), ch_depth=float(ch_d))

run_clicked = st.sidebar.button("▶ SWMM 실행", type="primary", use_container_width=True)
st.sidebar.caption("계산엔진: EPA SWMM 5.2 (swmm-toolkit). 실행할 때마다 입력파일(.inp)을 새로 만들어 모델을 구동합니다.")

if "scenarios" not in st.session_state:
    st.session_state.scenarios = {}

if run_clicked or "current" not in st.session_state:
    with st.spinner("SWMM 모델 실행 중..."):
        try:
            if auto_dur:
                rows, best = cached_critical(sc_json(sc), "A")
                sc.duration = int(best)
            res = cached_run(sc_json(sc))
            key = sc.name
            i = 2
            while key in st.session_state.scenarios and st.session_state.scenarios[key]["scenario"] != res["scenario"]:
                key = f"{sc.name} ({i})"
                i += 1
            st.session_state.scenarios[key] = res
            st.session_state.current = key
        except Exception as e:  # noqa: BLE001
            st.error(f"모델 실행 중 오류가 발생했습니다: {e}")
            st.stop()

cur_key = st.session_state.current
R = st.session_state.scenarios[cur_key]
S = R["scenario"]
rat = cached_rational(json.dumps(S, ensure_ascii=False, sort_keys=True), S["return_period"])

# ---------------------------------------------------------------------------
# 머리말
# ---------------------------------------------------------------------------
st.title("🌧️ 소규모 유역 강우-유출 모의 대시보드")
st.markdown(
    "경북 문경시 마성면 모곡리 소규모 유역 4개 지점(**D ⊂ C ⊂ B ⊂ A**)에 대해 "
    "**EPA SWMM**으로 강우-유출을 모의하고, 실무 기준인 **합리식** 결과와 비교합니다."
)
st.caption(
    f"현재 결과: **{cur_key}** · 재현기간 {S['return_period']}년 · 지속기간 {S['duration']}분 · "
    f"{'Huff 3분위' if S['storm']=='huff3' else '교호블록'} · AMC-{S['amc']} · 개발 {S['develop_pct']:.0f}% · "
    f"엔진 {R['engine']}"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 대시보드", "🔀 시나리오 비교", "⏱️ 임계지속기간", "✅ 검증 (합리식 vs SWMM)", "ℹ️ 모델 정보·입력파일"])

# ---------------------------------------------------------------------------
# 탭1 : 대시보드
# ---------------------------------------------------------------------------
with tab1:
    pt = st.radio("핵심지표 지점", ["A", "B", "C", "D"], horizontal=True, index=0, key="kpi_pt")
    pk = R["peaks"][pt]
    area_km2 = rat[pt]["area_km2"]
    rep = R["report"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("첨두유량 (SWMM)", f"{pk['Q']:.3f} ㎥/s",
              f"{100*(pk['Q']/rat[pt]['Q']-1):+.0f}% vs 합리식", delta_color="off")
    c2.metric("첨두발생시각", f"{pk['t']:.0f} 분")
    c3.metric("합리식 첨두유량", f"{rat[pt]['Q']:.3f} ㎥/s", f"tc {rat[pt]['tc']:.1f}분", delta_color="off")
    c4.metric("총유출량", f"{pk['volume_m3']:,.0f} ㎥")
    c5.metric("비유량", f"{pk['Q']/area_km2:.1f} ㎥/s/㎢")
    if rep.get("precip_mm"):
        st.caption(f"유역 전체: 총강우 {rep['precip_mm']:.1f} mm · 지표유출 {rep['runoff_mm']:.1f} mm · "
                   f"침투손실 {rep['infil_mm']:.1f} mm · 유출률 {rep['runoff_mm']/rep['precip_mm']:.2f} · "
                   f"연속성오차 {rep.get('runoff_continuity',0):.2f}% / {rep.get('routing_continuity',0):.2f}%")

    left, right = st.columns([1.6, 1])
    with left:
        sel = st.multiselect("수문곡선 표시 지점", ["A", "B", "C", "D"], default=["A", "B", "C", "D"])
        show_rat = st.checkbox("합리식 첨두유량 함께 표시", value=True)
        t = np.asarray(R["t"])
        tmax = st.slider("표시 시간 범위 (분)", 30, int(t.max()), min(int(t.max()), S["duration"] + 90), 10)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.28, 0.72], vertical_spacing=0.04)
        fig.add_trace(go.Bar(x=t, y=R["rain"], name="강우강도", marker_color="#4a7fc1"), row=1, col=1)
        for k in sel:
            fig.add_trace(go.Scatter(x=t, y=R["flow"][k], name=f"지점 {k}", line=dict(color=COLORS[k], width=2.5)),
                          row=2, col=1)
            if show_rat:
                fig.add_hline(y=rat[k]["Q"], line=dict(color=COLORS[k], dash="dash", width=1), row=2, col=1)
        fig.update_yaxes(title_text="mm/hr", autorange="reversed", row=1, col=1)
        fig.update_yaxes(title_text="유량 (㎥/s)", row=2, col=1)
        fig.update_xaxes(title_text="시간 (분)", range=[0, tmax], row=2, col=1)
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.08),
                          bargap=0)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("점선: 같은 재현기간의 합리식 첨두유량")

    with right:
        st.markdown("**유역 공간분포 — 비유량 (㎥/s/㎢)**")
        geo = load_geo()
        q_spec = {k: R["peaks"][k]["Q"] / rat[k]["area_km2"] for k in H.POINTS}
        qmin, qmax = min(q_spec.values()), max(q_spec.values())
        polys, paths, pts = [], [], []
        for f in geo["features"]:
            k = f["properties"]["id"]
            if f["properties"]["kind"] == "watershed":
                s = (q_spec[k] - qmin) / (qmax - qmin + 1e-9)
                polys.append(dict(id=k, polygon=f["geometry"]["coordinates"][0],
                                  color=[int(255 * s), int(80 + 60 * (1 - s)), int(255 * (1 - s)), 70],
                                  line=RGB[k], Q=round(R["peaks"][k]["Q"], 3), qs=round(q_spec[k], 1),
                                  area=round(rat[k]["area_km2"] * 100, 2)))
            elif f["properties"]["kind"] == "flowpath":
                paths.append(dict(id=k, path=f["geometry"]["coordinates"], color=RGB[k]))
            else:
                pts.append(dict(id=k, pos=f["geometry"]["coordinates"], color=RGB[k],
                                Q=round(R["peaks"][k]["Q"], 3), qs=round(q_spec[k], 1),
                                area=round(rat[k]["area_km2"] * 100, 2)))
        order = {"A": 0, "B": 1, "C": 2, "D": 3}
        polys.sort(key=lambda d: order[d["id"]])
        layers = [
            pdk.Layer("PolygonLayer", polys, get_polygon="polygon", get_fill_color="color",
                      get_line_color="line", line_width_min_pixels=2, pickable=True, stroked=True),
            pdk.Layer("PathLayer", paths, get_path="path", get_color="color", width_min_pixels=2),
            pdk.Layer("ScatterplotLayer", pts, get_position="pos", get_fill_color="color",
                      get_radius=12, radius_min_pixels=6, pickable=True),
            pdk.Layer("TextLayer", pts, get_position="pos", get_text="id", get_size=18,
                      get_color=[20, 20, 20], get_pixel_offset=[12, -12]),
        ]
        view = pdk.ViewState(latitude=36.6935, longitude=128.1060, zoom=15.2)
        deck = pdk.Deck(layers=layers, initial_view_state=view, map_provider="carto", map_style="light",
                        tooltip={"text": "지점 {id}\n면적 {area} ha\n첨두유량 {Q} ㎥/s\n비유량 {qs} ㎥/s/㎢"})
        st.pydeck_chart(deck, use_container_width=True)
        st.caption("유역경계·최장유로: 1:5,000 수치지형도 기반 사용자 구획 (D ⊂ C ⊂ B ⊂ A)")

    st.markdown("**지점별 결과**")
    tbl = pd.DataFrame({
        "유역면적(ha)": [rat[k]["area_km2"] * 100 for k in H.POINTS],
        "SWMM 첨두유량(㎥/s)": [R["peaks"][k]["Q"] for k in H.POINTS],
        "첨두시각(분)": [R["peaks"][k]["t"] for k in H.POINTS],
        "총유출량(㎥)": [R["peaks"][k]["volume_m3"] for k in H.POINTS],
        "합리식(㎥/s)": [rat[k]["Q"] for k in H.POINTS],
        "차이(%)": [100 * (R["peaks"][k]["Q"] / rat[k]["Q"] - 1) for k in H.POINTS],
    }, index=[f"지점 {k}" for k in H.POINTS]).iloc[::-1]
    st.dataframe(tbl.style.format({"유역면적(ha)": "{:.2f}", "SWMM 첨두유량(㎥/s)": "{:.3f}", "첨두시각(분)": "{:.0f}",
                                   "총유출량(㎥)": "{:,.0f}", "합리식(㎥/s)": "{:.3f}", "차이(%)": "{:+.1f}"}),
                 use_container_width=True)

# ---------------------------------------------------------------------------
# 탭2 : 시나리오 비교
# ---------------------------------------------------------------------------
with tab2:
    scen = st.session_state.scenarios
    st.markdown("사이드바에서 조건을 바꾸고 **▶ SWMM 실행**을 누를 때마다 시나리오가 여기에 쌓입니다. "
                "시나리오 이름을 바꿔가며 실행하세요.")
    names = list(scen.keys())
    chosen = st.multiselect("비교할 시나리오", names, default=names[-4:])
    cpt = st.radio("비교 지점", ["A", "B", "C", "D"], horizontal=True, key="cmp_pt")
    if chosen:
        fig2 = go.Figure()
        for nm in chosen:
            r = scen[nm]
            fig2.add_trace(go.Scatter(x=r["t"], y=r["flow"][cpt], name=nm, mode="lines"))
        fig2.update_layout(height=420, xaxis_title="시간 (분)", yaxis_title=f"지점 {cpt} 유량 (㎥/s)",
                           margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig2, use_container_width=True)

        rows = []
        for nm in chosen:
            r = scen[nm]
            s = r["scenario"]
            rows.append({"시나리오": nm, "재현기간": s["return_period"], "지속기간(분)": s["duration"],
                         "강우분포": "Huff3" if s["storm"] == "huff3" else "교호블록", "AMC": s["amc"],
                         "개발(%)": s["develop_pct"], "조도배율": s["n_mult"],
                         **{f"Q_{k}(㎥/s)": r["peaks"][k]["Q"] for k in ["A", "B", "C", "D"]},
                         f"첨두시각_{cpt}(분)": r["peaks"][cpt]["t"]})
        df = pd.DataFrame(rows).set_index("시나리오")
        st.dataframe(df, use_container_width=True)

        bar = go.Figure()
        for k in ["A", "B", "C", "D"]:
            bar.add_trace(go.Bar(name=f"지점 {k}", x=chosen, y=[scen[n]["peaks"][k]["Q"] for n in chosen],
                                 marker_color=COLORS[k]))
        bar.update_layout(barmode="group", height=360, yaxis_title="첨두유량 (㎥/s)",
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(bar, use_container_width=True)
    if st.button("시나리오 목록 비우기"):
        keep = st.session_state.current
        st.session_state.scenarios = {keep: st.session_state.scenarios[keep]}
        st.rerun()

# ---------------------------------------------------------------------------
# 탭3 : 임계지속기간
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("현재 조건(지속기간 제외)으로 지속기간을 바꿔가며 SWMM을 반복 실행해, 첨두유량이 가장 큰 "
                "**임계지속기간**을 찾습니다.")
    base = H.Scenario(**{**S})
    rows, best = cached_critical(sc_json(base), "A")
    dfc = pd.DataFrame(rows).set_index("duration")
    fig3 = go.Figure()
    for k in ["A", "B", "C", "D"]:
        fig3.add_trace(go.Scatter(x=dfc.index, y=dfc[k], name=f"지점 {k}", mode="lines+markers",
                                  line=dict(color=COLORS[k])))
    fig3.add_vline(x=best, line=dict(dash="dot", color="gray"),
                   annotation_text=f"임계지속기간 {best}분 (A)", annotation_position="top")
    fig3.update_layout(height=430, xaxis_title="강우 지속기간 (분)", yaxis_title="첨두유량 (㎥/s)",
                       margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(dfc.style.format("{:.3f}"), use_container_width=True)

# ---------------------------------------------------------------------------
# 탭4 : 검증
# ---------------------------------------------------------------------------
with tab4:
    st.markdown(
        "실무 기준값인 **합리식**(Kerby 유입시간 + 연속형 Kraven 유하시간, WAMIS 문경 확률강우강도식)과 "
        "**SWMM**(표준지침 CN·Huff 3분위, 임계지속기간) 첨두유량을 재현기간별로 비교합니다.")
    Ts = st.multiselect("재현기간", list(H.IDF.keys()), default=[30, 50, 80, 100])
    vpt = st.radio("지점", ["A", "B", "C", "D"], horizontal=True, key="ver_pt")
    if st.button("검증 계산 실행", type="primary"):
        st.session_state.ver_go = True
    if st.session_state.get("ver_go"):
        vrows = []
        prog = st.progress(0.0)
        for i, Tv in enumerate(Ts):
            ratv = H.rational(H.Scenario(**{**S, "return_period": Tv}), T=Tv)
            row = {"재현기간": Tv, "합리식": ratv[vpt]["Q"]}
            for amc_v in ("III", "II"):
                sv = H.Scenario(**{**S, "return_period": Tv, "amc": amc_v, "storm": "huff3"})
                rr, bd = cached_critical(sc_json(sv), vpt)
                qv = max(x[vpt] for x in rr)
                row[f"SWMM AMC-{amc_v}"] = qv
                row[f"임계지속(AMC-{amc_v})"] = bd
            vrows.append(row)
            prog.progress((i + 1) / max(1, len(Ts)))
        vdf = pd.DataFrame(vrows).set_index("재현기간")
        vdf["차이 III(%)"] = 100 * (vdf["SWMM AMC-III"] / vdf["합리식"] - 1)
        vdf["차이 II(%)"] = 100 * (vdf["SWMM AMC-II"] / vdf["합리식"] - 1)
        fig4 = go.Figure()
        for col, c in [("합리식", "#555"), ("SWMM AMC-III", "#d62728"), ("SWMM AMC-II", "#1f77b4")]:
            fig4.add_trace(go.Bar(name=col, x=[f"{t}년" for t in vdf.index], y=vdf[col], marker_color=c))
        fig4.update_layout(barmode="group", height=380, yaxis_title=f"지점 {vpt} 첨두유량 (㎥/s)",
                           margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig4, use_container_width=True)
        st.dataframe(vdf.style.format({"합리식": "{:.3f}", "SWMM AMC-III": "{:.3f}", "SWMM AMC-II": "{:.3f}",
                                       "차이 III(%)": "{:+.1f}", "차이 II(%)": "{:+.1f}"}),
                     use_container_width=True)
        st.info(
            "**해석** — 표준지침 설계조건(AMC-III)에서는 CN이 86~93으로 높아져 첨두 부근 유출률이 0.9 내외가 되므로 "
            "합리식(C=0.68~0.74)보다 첨두유량이 크게 산정됩니다. AMC-II 조건에서는 합리식과 근접하여, "
            "두 방법의 차이는 주로 **선행토양함수조건(유출률 가정)** 차이에서 비롯됨을 확인할 수 있습니다.")

# ---------------------------------------------------------------------------
# 탭5 : 모델 정보
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("SWMM 소유역 매개변수 (지점 사이 증분 유역)")
    subs = R["subs"]
    sdf = pd.DataFrame({
        "면적(ha)": [subs[k]["area_ha"] for k in H.POINTS],
        "불투수율(%)": [subs[k]["imp_pct"] for k in H.POINTS],
        "폭 W(m)": [subs[k]["width"] for k in H.POINTS],
        "경사(%)": [subs[k]["slope_pct"] for k in H.POINTS],
        "CN-II": [subs[k]["cn2"] for k in H.POINTS],
        f"CN 적용(AMC-{S['amc']})": [subs[k]["cn"] for k in H.POINTS],
        "투수 n": [subs[k]["n_perv"] for k in H.POINTS],
        "토양군 A/C/D 비율": [" / ".join(f"{subs[k]['hsg'][g]:.2f}" for g in "ACD") for k in H.POINTS],
    }, index=["S_D (D 상류)", "S_C (C−D)", "S_B (B−C)", "S_A (A−B)"])
    st.dataframe(sdf.style.format(precision=2), use_container_width=True)

    st.subheader("합리식 계산 내역")
    rdf = pd.DataFrame(rat).T.loc[H.POINTS]
    rdf.columns = ["면적(㎢)", "유출계수 C", "하도경사", "유입시간(분)", "유속(m/s)", "유하시간(분)",
                   "도달시간(분)", "강우강도(mm/hr)", "첨두유량(㎥/s)"]
    st.dataframe(rdf.style.format(precision=3), use_container_width=True)

    st.subheader("모델 구성")
    st.markdown(
        """
- **계산엔진**: EPA SWMM 5.2 — `swmm-toolkit`(EPA 공식 파이썬 바인딩)로 웹 서버에서 직접 실행
- **유역 구성**: 증분 소유역 4개(S_D, S_C, S_B, S_A) + 사다리꼴 하도 4구간(운동파 추적), D→C→B→A 연결
- **유출**: 도로=불투수면, 투수면은 NRCS 유출곡선지수(CN) 침투
- **사면흐름**: 표면류 길이 30 m(폭 W = 면적/30), 경사 25 %
- **설계강우**: WAMIS 문경(20051273) 확률강우강도식 General형 + Huff 3분위(지역 11) 또는 교호블록, 1분 간격
- **CN**: 홍수량 산정 표준지침 <표 3.1> + WAMIS 유출곡선지수 자료의 수문학적 토양군 중첩
- **검증 기준**: 합리식 (Kerby + 연속형 Kraven 도달시간)
        """)
    d1, d2 = st.columns(2)
    d1.download_button("SWMM 입력파일(.inp) 내려받기", R["inp"], file_name=f"{cur_key}.inp")
    d2.download_button("SWMM 보고서(.rpt) 내려받기", R["rpt"], file_name=f"{cur_key}.rpt")
    with st.expander("입력파일 미리보기"):
        st.code(R["inp"][:4000], language="ini")
