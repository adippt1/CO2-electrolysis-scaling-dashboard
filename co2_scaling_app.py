# CHEESE — CO₂ Handling & Electrolysis Efficiency Scaling Evaluator
# Tagline: Because scaling electrolysis shouldn’t be this gouda! 🧀
# Author: Aditya Prajapati + ChatGPT (GPT-5 Thinking)
# Copyright (c) 2025 Aditya Prajapati

from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# -------------------- Page & Sidebar --------------------
st.set_page_config(
    page_title="CHEESE — CO₂ Handling & Electrolysis Efficiency Scaling Evaluator",
    page_icon="🧀",
    layout="wide"
)

st.markdown("""
<style>
/* Tight, consistent inputs */
.block-container {max-width: 1250px;}
div[data-testid="stMetric"] {text-align:center;}
div[data-testid="stMetric"] > label {justify-content:center;}
/* Make number inputs more compact and consistent width */
section[data-testid="stSidebar"] .stNumberInput > div > div { width: 100%; }
.stNumberInput label p, .stSelectbox label p { font-weight: 600; }
.fe-grid .stNumberInput > div > div { min-width: 160px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        """
        ---
        **Created by**  
        **[Aditya Prajapati (Adi)](https://people.llnl.gov/prajapati3)**
        ---
        """,
        unsafe_allow_html=True
    )

# -------------------- Constants --------------------
EPS = 1e-12
F = 96485.33212  # C/mol e-
MV_OPTIONS = {
    "STP (0°C, 1 atm) — 22.414 L/mol": 22.414,
    "SATP (25°C, 1 atm) — 24.465 L/mol": 24.465,
}

# Product registry: electrons per mol, CO2 stoich per mol product, and E0 for EE display
DEFAULT_PRODUCTS = {
    "CO":      {"n_e": 2,  "co2_per_mol": 1.0, "E0": 1.33},
    "C2H4":    {"n_e": 12, "co2_per_mol": 2.0, "E0": 1.17},
    "CH3OH":   {"n_e": 6,  "co2_per_mol": 1.0, "E0": 0.02},
    "C2H5OH":  {"n_e": 12, "co2_per_mol": 2.0, "E0": 0.08},
    "MGO (Methylglyoxal)":     {"n_e": 12, "co2_per_mol": 3.0, "E0": 0.10},  # methylglyoxal (C3)
    "HCOO- (Formate)":    {"n_e": 2,  "co2_per_mol": 1.0, "E0": 0.20},  # formate
}
if "PRODUCTS" not in st.session_state:
    st.session_state.PRODUCTS = DEFAULT_PRODUCTS.copy()
PRODUCTS = st.session_state.PRODUCTS
PRODUCT_LIST: List[str] = list(PRODUCTS.keys())

# -------------------- Helpers --------------------
def to_m2(area_value: float, area_unit: str) -> float:
    return area_value * 1e-4 if area_unit == "cm²" else area_value

def to_A_per_m2(j_value: float, j_unit: str) -> float:
    if j_unit == "mA/cm²":
        return j_value * 10.0
    if j_unit == "A/cm²":
        return j_value * 1e4
    return j_value

def fe_to_frac(fe_pct: float) -> float:
    return max(0.0, min(1.0, (fe_pct or 0.0) / 100.0))

def amps(area_m2: float, j_A_m2: float) -> float:
    return area_m2 * j_A_m2

def prod_mol_s(I: float, fe_frac: float, ne_per_mol: int) -> float:
    return (I * fe_frac) / (max(ne_per_mol, EPS) * F)

def mol_s_to_slpm(n_dot: float, molar_volume_L: float) -> float:
    return n_dot * molar_volume_L * 60.0

def slpm_to_mol_s(flow_slpm: float, molar_volume_L: float) -> float:
    return flow_slpm / (molar_volume_L * 60.0 if molar_volume_L > 0 else np.inf)

def total_power_watts(I: float, V: float, n_units: int) -> float:
    return I * V * max(1, n_units)

# ---------- UI helpers for clean FE alignment ----------
def fe_grid_inputs(
    section_key: str,
    products: List[str],
    default_map: Optional[Dict[str, float]] = None,
    title: str = "Faradaic Efficiencies (%, sum ≤ 100)",
    per_row: int = 3
) -> Dict[str, float]:
    """Render a neatly aligned grid of FE (%) number_inputs, returning {product: value}."""
    st.markdown(f"#### {title}")
    fe_map: Dict[str, float] = {}
    # fallback defaults: CO=90, others 0
    defaults = {p: (90.0 if p == "CO" else 0.0) for p in products}
    if default_map:
        defaults.update({k: float(default_map.get(k, defaults[k])) for k in products})

    # rows of 'per_row' products
    for i in range(0, len(products), per_row):
        row = products[i:i+per_row]
        cols = st.columns(len(row), gap="small")
        with st.container():
            st.markdown('<div class="fe-grid">', unsafe_allow_html=True)
            for c, p in enumerate(row):
                with cols[c]:
                    fe_map[p] = st.number_input(
                        f"{p} FE (%)",
                        min_value=0.0, max_value=100.0,
                        value=defaults[p],
                        step=1.0,
                        key=f"{section_key}_fe_{p}"
                    )
            st.markdown("</div>", unsafe_allow_html=True)
    return fe_map

def fe_mean_stdev_grid(
    section_key: str,
    products: List[str],
    mean_defaults: Optional[Dict[str, float]] = None,
    stdev_defaults: Optional[Dict[str, float]] = None,
    title: str = "FE means & stdevs",
    per_row: int = 3
):
    """Aligned grid for Monte Carlo: per product column: mean (%) then stdev (%)."""
    st.subheader(title)
    mean_def = {p: (90.0 if p == "CO" else 0.0) for p in products}
    if mean_defaults:
        mean_def.update({k: float(mean_defaults.get(k, mean_def[k])) for k in products})
    sd_def = {p: (2.0 if p == "CO" else 1.0) for p in products}
    if stdev_defaults:
        sd_def.update({k: float(stdev_defaults.get(k, sd_def[k])) for k in products})

    fe_mean: Dict[str, float] = {}
    fe_sd: Dict[str, float] = {}

    for i in range(0, len(products), per_row):
        row = products[i:i+per_row]
        cols = st.columns(len(row), gap="small")
        for c, p in enumerate(row):
            with cols[c]:
                st.markdown(f"**{p}**")
                fe_mean[p] = st.number_input(
                    "mean (%)",
                    min_value=0.0, max_value=100.0,
                    value=mean_def[p],
                    step=1.0,
                    key=f"{section_key}_mean_{p}"
                )
                fe_sd[p] = st.number_input(
                    "stdev (%)",
                    min_value=0.0, max_value=100.0,
                    value=sd_def[p],
                    step=0.5,
                    key=f"{section_key}_sd_{p}"
                )
    return fe_mean, fe_sd

# -------------------- Data class --------------------
@dataclass
class ElectrolyzerInputs:
    area_value: float
    area_unit: str
    j_value: float
    j_unit: str
    V_cell: float
    fe_map_pct: Dict[str, float]  # FE% per product key
    n_units: int
    molar_vol_L: float
    mode: str  # "S" or "INLET"
    stoich_ratio: Optional[float] = None
    co2_in_slpm_input: Optional[float] = None

# -------------------- Core calculators (multi-product) --------------------
def compute_core_products(inp: ElectrolyzerInputs) -> Dict[str, float]:
    A_m2 = to_m2(inp.area_value, inp.area_unit)
    j_A_m2 = to_A_per_m2(inp.j_value, inp.j_unit)
    I_unit = amps(A_m2, j_A_m2)
    I_total = I_unit * max(1, inp.n_units)
    P_total_W = total_power_watts(I_unit, inp.V_cell, inp.n_units)

    co2_min_mol_s = 0.0
    EE_total = 0.0
    out = {}

    for p in PRODUCTS:
        fe_frac = fe_to_frac(inp.fe_map_pct.get(p, 0.0))
        n_e = PRODUCTS[p]["n_e"]
        co2_per = PRODUCTS[p]["co2_per_mol"]
        E0 = PRODUCTS[p]["E0"]

        n_p = prod_mol_s(I_total, fe_frac, n_e)
        out[f"{p}_mol_s"] = n_p
        out[f"{p}_slpm"] = mol_s_to_slpm(n_p, inp.molar_vol_L)
        co2_min_mol_s += n_p * co2_per
        EE_total += (E0 / max(inp.V_cell, EPS)) * fe_frac

    out["CO2_min_slpm"] = mol_s_to_slpm(co2_min_mol_s, inp.molar_vol_L)
    out["I_unit_A"] = I_unit
    out["I_total_A"] = I_total
    out["P_total_W"] = P_total_W
    out["EE_total"] = EE_total
    return out

def apply_feed_mode(core: Dict[str, float], inp: ElectrolyzerInputs) -> Dict[str, float]:
    co2_min_slpm = max(core["CO2_min_slpm"], 0.0)

    if co2_min_slpm <= EPS:
        if inp.mode == "S":
            S = max(1.0, float(inp.stoich_ratio or 1.0))
            co2_in_slpm = 0.0
            util = 1.0
            warn = "Total FE to carbon products is zero; CO₂ minimum is 0. Adjust FE split."
        else:
            co2_in_slpm = max(0.0, float(inp.co2_in_slpm_input or 0.0))
            S = np.inf if co2_in_slpm > 0 else 1.0
            util = 0.0 if co2_in_slpm > 0 else 1.0
            warn = "Total FE to carbon products is zero; CO₂ minimum is 0. Inlet has no effect."
        return {"CO2_in_slpm": co2_in_slpm, "Stoich_S": S, "CO2_utilization": util, "warning": warn}

    if inp.mode == "S":
        S = max(1.0, float(inp.stoich_ratio or 1.0))
        co2_in_slpm = S * co2_min_slpm
        util = 1.0 / S
        warn = None
    else:
        co2_in_slpm = max(0.0, float(inp.co2_in_slpm_input or 0.0))
        S = co2_in_slpm / max(co2_min_slpm, EPS)
        if co2_in_slpm < co2_min_slpm:
            util = min(1.0, 1.0 / max(S, EPS))
            warn = f"Provided CO₂ inlet ({co2_in_slpm:.3f} SLPM) is below the theoretical minimum ({co2_min_slpm:.3f} SLPM)."
        else:
            util = 1.0 / max(S, EPS)
            warn = None
    return {"CO2_in_slpm": co2_in_slpm, "Stoich_S": S, "CO2_utilization": min(1.0, util), "warning": warn}

def build_sensitivity_table_S(core: Dict[str, float], S_min: float, S_max: float, S_step: float) -> pd.DataFrame:
    co2_min_slpm = core["CO2_min_slpm"]
    prod_slpm_map = {p: core[f"{p}_slpm"] for p in PRODUCTS}
    total_prod_slpm = sum(prod_slpm_map.values())

    S_vals = np.arange(S_min, S_max + 1e-9, S_step)
    rows = []
    for S in S_vals:
        S = max(1.0, float(S))
        util = 1.0 / S
        co2_in = S * co2_min_slpm
        co2_out = co2_in - co2_min_slpm
        total_out = max(1e-12, co2_out + total_prod_slpm)

        row = {
            "Stoich S (inlet/min)": S,
            "CO2 Utilization (frac)": util,
            "CO2 Inlet (SLPM)": co2_in,
            "CO2 Outlet (SLPM)": co2_out,
            "Total Outlet (SLPM)": total_out,
            "CO2 vol%": 100 * co2_out / total_out,
        }
        for p in PRODUCTS:
            row[f"{p} (SLPM)"] = prod_slpm_map[p]
            row[f"{p} vol%"] = 100 * prod_slpm_map[p] / total_out
        rows.append(row)
    return pd.DataFrame(rows)

def build_sensitivity_table_U(core: Dict[str, float], Umin_pct: float, Umax_pct: float, Ustep_pct: float) -> pd.DataFrame:
    co2_min_slpm = core["CO2_min_slpm"]
    prod_slpm_map = {p: core[f"{p}_slpm"] for p in PRODUCTS}
    total_prod_slpm = sum(prod_slpm_map.values())

    U_vals_pct = np.arange(Umin_pct, Umax_pct + 1e-9, Ustep_pct)
    rows = []
    for U_pct in U_vals_pct:
        U = max(1e-6, min(1.0, U_pct / 100.0))
        S = 1.0 / U
        co2_in = S * co2_min_slpm
        co2_out = co2_in - co2_min_slpm
        total_out = max(1e-12, co2_out + total_prod_slpm)

        row = {
            "Utilization (%)": U_pct,
            "Stoich S (inlet/min)": S,
            "CO2 Inlet (SLPM)": co2_in,
            "CO2 Outlet (SLPM)": co2_out,
            "Total Outlet (SLPM)": total_out,
            "CO2 vol%": 100 * co2_out / total_out,
        }
        for p in PRODUCTS:
            row[f"{p} (SLPM)"] = prod_slpm_map[p]
            row[f"{p} vol%"] = 100 * prod_slpm_map[p] / total_out
        rows.append(row)
    return pd.DataFrame(rows)

# -------------------- Global Settings --------------------
st.sidebar.header("Global Settings")
mv_label_global = st.sidebar.selectbox("Gas molar volume basis", list(MV_OPTIONS.keys()), index=0, key="gs_mv")
molar_vol_global = MV_OPTIONS[mv_label_global]

use_stack_global = st.sidebar.checkbox("Use a stack (multiple identical units)?", value=True, key="gs_stack")
n_units_global = st.sidebar.number_input("Number of units in stack", min_value=1, value=10, step=1, key="gs_units")
st.sidebar.markdown("---")

# -------------------- UI --------------------
st.title("🧀 CHEESE: CO₂ Handling & Electrolysis Efficiency Scaling Evaluator")
st.caption("CO₂ → CO, C₂H₄, CH₃OH, C₂H₅OH, MGO, HCOO | Calculators + sensitivities + Monte Carlo")

main_tabs = st.tabs([
    "Calculator",
    "Calc: Size Active Area from CO₂ Inlet & Stoich",
    "Sensitivity: CO₂ Utilization",
    "Sensitivity: Area × Stack",
    "Sensitivity: CO₂ Supply Cap",
    "Monte Carlo",
    "Constants",
])

# -------------------- Calculator --------------------
with main_tabs[0]:
    st.header("Calculator")
    colA, colB, colC = st.columns(3)
    with colA:
        area_value = st.number_input("Active area per unit", min_value=0.0, value=100.0, step=1.0, key="calc_area")
        area_unit = st.selectbox("Area unit", ["cm²", "m²"], index=0, key="calc_area_unit")
    with colB:
        j_value = st.number_input("Current density", min_value=0.0, value=200.0, step=10.0, key="calc_j")
        j_unit = st.selectbox("j units", ["mA/cm²", "A/cm²", "A/m²"], index=0, key="calc_j_unit")
    with colC:
        V_cell = st.number_input("Cell voltage (V)", min_value=0.0, value=3.2, step=0.1, key="calc_V")

    fe_map_pct = fe_grid_inputs("calc", PRODUCT_LIST)

    st.divider()
    mode = st.radio("CO₂ feed input mode", ["Stoich (S)", "Inlet flow (SLPM)"], index=0, horizontal=True, key="calc_mode")
    if mode == "Stoich (S)":
        stoich_ratio = st.number_input("CO₂ Stoich S (inlet/min)", min_value=1.0, value=2.0, step=0.1, key="calc_S")
        co2_in_slpm_input = None
        mode_key = "S"
    else:
        co2_in_slpm_input = st.number_input("CO₂ Inlet flow (SLPM)", min_value=0.0, value=10.0, step=0.5, key="calc_inlet")
        stoich_ratio = None
        mode_key = "INLET"

    n_units_effective = n_units_global if use_stack_global else 1
    inp = ElectrolyzerInputs(
        area_value=area_value, area_unit=area_unit,
        j_value=j_value, j_unit=j_unit,
        V_cell=V_cell, fe_map_pct=fe_map_pct,
        n_units=n_units_effective, molar_vol_L=molar_vol_global,
        mode=mode_key, stoich_ratio=stoich_ratio,
        co2_in_slpm_input=co2_in_slpm_input,
    )
    core = compute_core_products(inp)
    feed = apply_feed_mode(core, inp)

    st.subheader("Results")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Per-Unit Current (A)", f"{core['I_unit_A']:.2f}")
    with m2: st.metric("Total Current (A)", f"{core['I_total_A']:.2f}")
    with m3: st.metric("Total Power (kW)", f"{core['P_total_W']/1000:.2f}")
    with m4: st.metric("Energy Efficiency (Σ E₀/E_cell·FE)", f"{core['EE_total']*100:.1f}%")

    m5, m6, m7, m8 = st.columns(4)
    with m5: st.metric("CO₂ Minimum (SLPM)", f"{core['CO2_min_slpm']:.3f}")
    with m6: st.metric("CO₂ Inlet (SLPM)", f"{feed['CO2_in_slpm']:.3f}")
    with m7:
        Sval = feed['Stoich_S']
        st.metric("Stoich S (inlet/min)", "∞" if not np.isfinite(Sval) else f"{Sval:.3f}")
    with m8: st.metric("Utilization (%)", f"{feed['CO2_utilization']*100:.1f}")
    if feed["warning"]:
        st.warning(feed["warning"])

    st.divider()
    st.markdown("#### Product flowrates (SLPM)")
    flow_cols = st.columns(len(PRODUCT_LIST))
    for i, p in enumerate(PRODUCT_LIST):
        with flow_cols[i]:
            st.metric(p, f"{core[f'{p}_slpm']:.3f}")

# -------------------- Calc: Size Active Area from CO₂ Inlet & Stoich --------------------
with main_tabs[1]:
    st.header("Calc: Size Active Area from CO₂ Inlet & Stoich")
    st.caption("Provide CO₂ inlet (SLPM), Stoich S, current density, and FE split. Computes required active area.")

    units_used = n_units_global if use_stack_global else 1
    col1, col2, col3 = st.columns(3)
    with col1:
        co2_in_slpm_sz = st.number_input("CO₂ Inlet (SLPM)", min_value=0.0, value=50.0, step=1.0, key="sz_inlet")
        S_sz = st.number_input("Stoich S (inlet/min)", min_value=1.0, value=2.0, step=0.1, key="sz_S")
    with col2:
        j_val_sz = st.number_input("Current density", min_value=0.0, value=200.0, step=10.0, key="sz_j")
        j_unit_sz = st.selectbox("j units", ["mA/cm²", "A/cm²", "A/m²"], index=0, key="sz_j_unit")
        V_cell_sz = st.number_input("Cell voltage (V)", min_value=0.0, value=3.2, step=0.1, key="sz_V")
    with col3:
        fe_map_sz = fe_grid_inputs("sz", PRODUCT_LIST, title="FE split (%)", per_row=3)

    j_A_m2_sz = to_A_per_m2(j_val_sz, j_unit_sz)
    co2_min_slpm_sz = co2_in_slpm_sz / max(S_sz, EPS)
    co2_min_mol_s_sz = slpm_to_mol_s(co2_min_slpm_sz, molar_vol_global)

    denom = 0.0
    for p in PRODUCT_LIST:
        fe_frac = fe_to_frac(fe_map_sz.get(p, 0.0))
        denom += fe_frac * PRODUCTS[p]["co2_per_mol"] / max(PRODUCTS[p]["n_e"], EPS)

    if denom <= EPS:
        st.error("FE split yields zero carbon products (Σ FE_i·CO₂_per_i/n_e_i = 0). Increase FEs.")
    else:
        I_total_sz = co2_min_mol_s_sz * F / denom
        A_total_m2 = I_total_sz / max(j_A_m2_sz, EPS)
        A_total_cm2 = A_total_m2 * 1e4
        A_per_unit_m2 = A_total_m2 / max(units_used, 1)
        A_per_unit_cm2 = A_per_unit_m2 * 1e4

        # Product rates for sized area
        prod_rows = []
        for p in PRODUCT_LIST:
            fe_frac = fe_to_frac(fe_map_sz.get(p, 0.0))
            n_e = PRODUCTS[p]["n_e"]
            n_p = prod_mol_s(I_total_sz, fe_frac, n_e)
            slpm_p = mol_s_to_slpm(n_p, molar_vol_global)
            prod_rows.append((p, slpm_p))

        P_total_kW_sz = (I_total_sz * V_cell_sz) / 1000.0
        util_sz = 1.0 / max(S_sz, EPS)

        show_cm2 = st.toggle("Display resultant area in cm²", value=False, key="sz_area_toggle")
        if show_cm2:
            area_unit_label = "cm²"
            total_area_display = A_total_cm2
            per_unit_area_display = A_per_unit_cm2
            fmt_total = "{:,.0f}"
            fmt_per_unit = "{:,.0f}"
        else:
            area_unit_label = "m²"
            total_area_display = A_total_m2
            per_unit_area_display = A_per_unit_m2
            fmt_total = "{:.3f}"
            fmt_per_unit = "{:.4f}"

        st.subheader("Sizing Results")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(f"Total active area ({area_unit_label})", fmt_total.format(total_area_display))
            st.metric("Total current (A)", f"{I_total_sz:.1f}")
        with c2:
            st.metric(f"Per-unit area ({area_unit_label})", fmt_per_unit.format(per_unit_area_display))
            st.metric("Power (kW)", f"{P_total_kW_sz:.2f}")
        with c3:
            st.metric("CO₂ Minimum (SLPM)", f"{co2_min_slpm_sz:.3f}")
            st.metric("Utilization (%)", f"{util_sz*100:.1f}")

        st.caption("Per-unit metrics assume equal area per unit. Uncheck 'Use a stack' in Global Settings for single-unit sizing.")

        st.subheader("Resulting Product Rates (SLPM) for sized area")
        colsR = st.columns(len(PRODUCT_LIST))
        for i, (p, slpm_p) in enumerate(prod_rows):
            with colsR[i]:
                st.metric(p, f"{slpm_p:.3f}")

# -------------------- Sensitivity: CO₂ Utilization --------------------
with main_tabs[2]:
    st.header("Sensitivity: CO₂ Utilization")
    st.caption("Sweep Utilization (%) or view composition vs S. FE is held constant.")

    col1, col2, col3 = st.columns(3)
    with col1:
        area_u = st.number_input("Area per unit (cm²)", min_value=0.0, value=100.0, step=5.0, key="u_area")
        j_u = st.number_input("Current density (mA/cm²)", min_value=0.0, value=200.0, step=10.0, key="u_j")
        V_u = st.number_input("Cell voltage (V)", min_value=0.0, value=3.2, step=0.1, key="u_V")
    with col2:
        fe_map_u = fe_grid_inputs("u", PRODUCT_LIST, title="FE split (%)", per_row=3)
        units_u = n_units_global if use_stack_global else 1
        st.info(f"Using global stack setting: {units_u} unit(s).")
    with col3:
        molar_vol_u = molar_vol_global
        st.write(f"**Gas basis:** {molar_vol_u:.3f} L/mol")

    Umin = st.number_input("Utilization min (%)", min_value=1.0, value=20.0, step=1.0, key="u_min")
    Umax = st.number_input("Utilization max (%)", min_value=1.0, value=100.0, step=1.0, key="u_max")
    Ustep = st.number_input("Utilization step (%)", min_value=1.0, value=5.0, step=1.0, key="u_step")

    inp_u = ElectrolyzerInputs(
        area_value=area_u, area_unit="cm²", j_value=j_u, j_unit="mA/cm²",
        V_cell=V_u, fe_map_pct=fe_map_u,
        n_units=units_u, molar_vol_L=molar_vol_u,
        mode="S", stoich_ratio=1.0
    )
    core_u = compute_core_products(inp_u)

    df_util = build_sensitivity_table_U(core_u, Umin, Umax, Ustep)
    value_vars_flows = [f"{p} (SLPM)" for p in PRODUCT_LIST] + ["CO2 Outlet (SLPM)"]
    df_flows = df_util.melt(
        id_vars=["Utilization (%)"],
        value_vars=value_vars_flows,
        var_name="Stream", value_name="SLPM"
    )
    chart_flows = alt.Chart(df_flows).mark_line(point=True).encode(
        x=alt.X("Utilization (%):Q"),
        y=alt.Y("SLPM:Q"),
        color="Stream:N",
        tooltip=["Utilization (%)", "Stream", "SLPM"]
    ).properties(title="Outlet flows vs Utilization (%)", height=320)

    df_sens_S = build_sensitivity_table_S(core_u, S_min=1.0, S_max=max(1.0, 1.0/(Umin/100.0)), S_step=0.5)
    value_vars_comp = ["CO2 vol%"] + [f"{p} vol%" for p in PRODUCT_LIST]
    df_comp = df_sens_S.melt(
        id_vars=["Stoich S (inlet/min)"],
        value_vars=value_vars_comp,
        var_name="Species", value_name="vol%"
    )
    chart_comp = alt.Chart(df_comp).mark_line(point=True).encode(
        x=alt.X("Stoich S (inlet/min):Q"),
        y=alt.Y("vol%:Q"),
        color="Species:N",
        tooltip=["Stoich S (inlet/min)", "Species", "vol%"]
    ).properties(title="Outlet composition vs Stoich S", height=320)

    left, right = st.columns(2)
    with left:
        st.altair_chart(chart_flows, use_container_width=True)
    with right:
        st.altair_chart(chart_comp, use_container_width=True)

    st.download_button(
        "Download utilization sweep (CSV)",
        data=df_util.to_csv(index=False).encode("utf-8"),
        file_name="utilization_sweep.csv",
        mime="text/csv",
        key="u_dl"
    )

# -------------------- Sensitivity: Area × Stack --------------------
with main_tabs[3]:
    st.header("Sensitivity: Area × Stack")
    st.caption("Sweep active area per unit and # of units. Visualize production, power, and CO₂ needs.")

    col1, col2, col3 = st.columns(3)
    with col1:
        j_value1 = st.number_input("Current density", min_value=0.0, value=200.0, step=10.0, key="axs_j")
        j_unit1 = st.selectbox("j units", ["mA/cm²", "A/cm²", "A/m²"], index=0, key="axs_j_unit")
        V_cell1 = st.number_input("Cell voltage (V)", min_value=0.0, value=3.2, step=0.1, key="axs_V")
    with col2:
        fe_map1 = fe_grid_inputs("axs", PRODUCT_LIST, title="FE split (%)", per_row=3)
        S1 = st.number_input("Stoich S for sweep", min_value=1.0, value=2.0, step=0.1, key="axs_S")
    with col3:
        area_min = st.number_input("Area per unit - min (cm²)", min_value=0.0, value=25.0, step=5.0, key="axs_area_min")
        area_max = st.number_input("Area per unit - max (cm²)", min_value=0.0, value=400.0, step=10.0, key="axs_area_max")
        area_step = st.number_input("Area step (cm²)", min_value=1.0, value=25.0, step=1.0, key="axs_area_step")
        n_min = st.number_input("# Units - min", min_value=1, value=1, step=1, key="axs_n_min")
        n_max = st.number_input("# Units - max", min_value=1, value=50, step=1, key="axs_n_max")
        n_step = st.number_input("# Units step", min_value=1, value=5, step=1, key="axs_n_step")

    area_vals_cm2 = np.arange(area_min, area_max + 1e-9, area_step)
    n_vals = np.arange(n_min, n_max + 1, n_step)

    rows = []
    for area_cm2 in area_vals_cm2:
        area_m2 = area_cm2 * 1e-4
        j_A_m2 = to_A_per_m2(j_value1, j_unit1)
        I_unit = amps(area_m2, j_A_m2)
        for n_units1 in n_vals:
            I_total = I_unit * n_units1

            prod_slpm = {}
            co2_min_mol_s = 0.0
            for p in PRODUCT_LIST:
                fe_frac = fe_to_frac(fe_map1.get(p, 0.0))
                n_e = PRODUCTS[p]["n_e"]
                n_p = prod_mol_s(I_total, fe_frac, n_e)
                prod_slpm[p] = mol_s_to_slpm(n_p, molar_vol_global)
                co2_min_mol_s += n_p * PRODUCTS[p]["co2_per_mol"]

            co2_min_slpm = mol_s_to_slpm(co2_min_mol_s, molar_vol_global)
            co2_in_slpm = S1 * co2_min_slpm
            P_total_kW = (I_total * V_cell1) / 1000.0

            row = {
                "Area_cm2": area_cm2,
                "Units": int(n_units1),
                "CO2_min_SLPM": co2_min_slpm,
                "CO2_in_SLPM": co2_in_slpm,
                "Power_kW": P_total_kW
            }
            for p in PRODUCT_LIST:
                row[f"{p}_SLPM"] = prod_slpm[p]
            rows.append(row)

    df_grid = pd.DataFrame(rows)

    metric_choices = [f"{p}_SLPM" for p in PRODUCT_LIST] + ["CO2_in_SLPM", "CO2_min_SLPM", "Power_kW"]
    metric = st.selectbox("Heatmap metric", metric_choices, index=0, key="axs_metric")
    heat = alt.Chart(df_grid).mark_rect().encode(
        x=alt.X("Area_cm2:O", title="Area per unit (cm²)"),
        y=alt.Y("Units:O", title="# of Units"),
        color=alt.Color(f"{metric}:Q", title=metric),
        tooltip=["Area_cm2", "Units"] + metric_choices
    ).properties(height=420)
    st.altair_chart(heat, use_container_width=True)

    st.download_button(
        "Download Area×Stack grid (CSV)",
        data=df_grid.to_csv(index=False).encode("utf-8"),
        file_name="area_stack_grid.csv",
        mime="text/csv",
        key="axs_dl"
    )

# -------------------- Sensitivity: CO₂ Supply Cap --------------------
with main_tabs[4]:
    st.header("Sensitivity: CO₂ Supply Cap")
    st.caption("Impose a maximum CO₂ inlet and evaluate feasibility, utilization, and recommendations.")

    col1, col2, col3 = st.columns(3)
    with col1:
        area_value2 = st.number_input("Area per unit (cm²)", min_value=0.0, value=100.0, step=5.0, key="cap_area")
        j_value2 = st.number_input("Current density", min_value=0.0, value=200.0, step=10.0, key="cap_j")
        j_unit2 = st.selectbox("j units", ["mA/cm²", "A/cm²", "A/m²"], index=0, key="cap_j_unit")
    with col2:
        V_cell2 = st.number_input("Cell voltage (V)", min_value=0.0, value=3.2, step=0.1, key="cap_V")
        fe_map2 = fe_grid_inputs("cap", PRODUCT_LIST, title="FE split (%)", per_row=3)
    with col3:
        n_units2 = n_units_global if use_stack_global else 1
        st.info(f"Using global stack setting: {n_units2} unit(s).")
        co2_cap = st.number_input("CO₂ supply cap (SLPM)", min_value=0.0, value=50.0, step=1.0, key="cap_cap")

    A_m2 = area_value2 * 1e-4
    j_A_m2 = to_A_per_m2(j_value2, j_unit2)
    I_unit = amps(A_m2, j_A_m2)
    I_total = I_unit * n_units2

    co2_min_mol_s = 0.0
    for p in PRODUCT_LIST:
        fe_frac = fe_to_frac(fe_map2.get(p, 0.0))
        n_e = PRODUCTS[p]["n_e"]
        n_p = prod_mol_s(I_total, fe_frac, n_e)
        co2_min_mol_s += n_p * PRODUCTS[p]["co2_per_mol"]
    co2_min_slpm2 = mol_s_to_slpm(co2_min_mol_s, molar_vol_global)

    P_total_kW2 = (I_total * V_cell2) / 1000.0
    S_min_cap = (co2_cap / max(co2_min_slpm2, EPS)) if co2_min_slpm2 > 0 else np.inf
    util_max_cap = min(1.0, 1.0 / max(S_min_cap, EPS))

    st.subheader("Results under cap")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("CO₂ Minimum (SLPM)", f"{co2_min_slpm2:.3f}")
    with c2: st.metric("CO₂ Cap (SLPM)", f"{co2_cap:.3f}")
    with c3: st.metric("Max Utilization allowed", f"{100*util_max_cap:.1f}%")
    with c4: st.metric("Total Power (kW)", f"{P_total_kW2:.2f}")

    if np.isinf(S_min_cap) or co2_cap < co2_min_slpm2:
        st.warning("Cap is below the theoretical minimum CO₂ required at these operating conditions. Reduce current (j), area, units, or adjust FE split.")
    else:
        st.success("Feasible. You may increase utilization up to the shown maximum by reducing S accordingly.")

# -------------------- Monte Carlo --------------------
with main_tabs[5]:
    st.header("Monte Carlo")
    st.caption("Sample uncertainties in FE, j, V, area, and S to see distributions of outputs.")

    left, right = st.columns([1,1])
    with left:
        N = st.number_input("Samples (N)", min_value=100, value=5000, step=100, key="mc_N")
        seed = st.number_input("Random seed", min_value=0, value=42, step=1, key="mc_seed")
        np.random.seed(seed)

        area_mc = st.number_input("Area per unit (cm²) — mean", min_value=0.0, value=100.0, step=1.0, key="mc_area_mean")
        area_cv = st.number_input("Area coefficient of variation (%)", min_value=0.0, value=5.0, step=0.5, key="mc_area_cv")

        j_mc = st.number_input("j (mA/cm²) — mean", min_value=0.0, value=200.0, step=10.0, key="mc_j_mean")
        j_cv = st.number_input("j coefficient of variation (%)", min_value=0.0, value=5.0, step=0.5, key="mc_j_cv")

        V_mc = st.number_input("Cell voltage (V) — mean", min_value=0.0, value=3.2, step=0.1, key="mc_V_mean")
        V_cv = st.number_input("Voltage coefficient of variation (%)", min_value=0.0, value=2.0, step=0.5, key="mc_V_cv")

        units_mc = n_units_global if use_stack_global else 1
        st.info(f"Using global stack setting: {units_mc} unit(s).")

        S_mean = st.number_input("S — mean", min_value=1.0, value=2.0, step=0.1, key="mc_S_mean")
        S_cv = st.number_input("S coefficient of variation (%)", min_value=0.0, value=5.0, step=0.5, key="mc_S_cv")

    with right:
        fe_mean, fe_sd = fe_mean_stdev_grid("mc", PRODUCT_LIST)

    def lognormal_samples(mean, cv_pct, size):
        if mean <= 0:
            return np.zeros(size)
        cv = cv_pct / 100.0
        if cv <= 0:
            return np.full(size, mean)
        sigma2 = np.log(1 + cv**2)
        mu = np.log(max(mean, EPS)) - 0.5 * sigma2
        sigma = np.sqrt(sigma2)
        return np.random.lognormal(mean=mu, sigma=sigma, size=size)

    area_s = lognormal_samples(area_mc, area_cv, N)
    j_s = lognormal_samples(j_mc, j_cv, N)
    V_s = lognormal_samples(V_mc, V_cv, N)
    S_s = np.maximum(1.0, lognormal_samples(S_mean, S_cv, N))

    def truncnorm(mean, sd, size):
        x = np.random.normal(mean, sd, size)
        return np.clip(np.nan_to_num(x, nan=0.0), 0.0, 100.0)

    fe_samples = {p: truncnorm(fe_mean[p], fe_sd[p], N) for p in PRODUCT_LIST}
    fe_stack = np.vstack([fe_samples[p] for p in PRODUCT_LIST]).T
    fe_sum = fe_stack.sum(axis=1)
    scale = np.where(fe_sum > 100.0, 100.0 / np.maximum(fe_sum, EPS), 1.0)
    fe_stack = (fe_stack.T * scale).T
    for i, p in enumerate(PRODUCT_LIST):
        fe_samples[p] = fe_stack[:, i]

    results = {"CO2_min_SLPM": [], "CO2_in_SLPM": [], "Utilization_frac": [], "Power_kW": []}
    for p in PRODUCT_LIST:
        results[f"{p}_SLPM"] = []

    for i in range(N):
        A_m2_i = area_s[i] * 1e-4
        j_A_m2_i = j_s[i] * 10.0
        I_unit_i = amps(A_m2_i, j_A_m2_i)
        I_total_i = I_unit_i * max(1, units_mc)

        co2_min_mol_s_i = 0.0
        for p in PRODUCT_LIST:
            fe_frac_i = fe_samples[p][i] / 100.0
            n_e = PRODUCTS[p]["n_e"]
            n_p_i = prod_mol_s(I_total_i, fe_frac_i, n_e)
            slpm_p_i = mol_s_to_slpm(n_p_i, molar_vol_global)
            results[f"{p}_SLPM"].append(slpm_p_i)
            co2_min_mol_s_i += n_p_i * PRODUCTS[p]["co2_per_mol"]

        co2_min_slpm_i = mol_s_to_slpm(co2_min_mol_s_i, molar_vol_global)
        results["CO2_min_SLPM"].append(co2_min_slpm_i)

        S_i = max(1.0, S_s[i])
        co2_in_slpm_i = S_i * co2_min_slpm_i
        util_i = 1.0 / S_i
        results["CO2_in_SLPM"].append(co2_in_slpm_i)
        results["Utilization_frac"].append(util_i)

        Vnow = max(V_s[i], EPS)
        results["Power_kW"].append((I_total_i * Vnow) / 1000.0)

    df_mc = pd.DataFrame(results)

    st.subheader("Distributions")
    plot_cols = st.columns(2)
    with plot_cols[0]:
        chart_util = alt.Chart(df_mc).mark_bar().encode(
            x=alt.X("Utilization_frac:Q", bin=alt.Bin(maxbins=40), title="Utilization (fraction)"),
            y=alt.Y("count()", title="Count")
        ).properties(title="Utilization distribution", height=300)
        st.altair_chart(chart_util, use_container_width=True)

        chart_pwr = alt.Chart(df_mc).mark_bar().encode(
            x=alt.X("Power_kW:Q", bin=alt.Bin(maxbins=40), title="Power (kW)"),
            y=alt.Y("count()", title="Count")
        ).properties(title="Power distribution", height=300)
        st.altair_chart(chart_pwr, use_container_width=True)

    with plot_cols[1]:
        chart_cmin = alt.Chart(df_mc).mark_bar().encode(
            x=alt.X("CO2_min_SLPM:Q", bin=alt.Bin(maxbins=40), title="CO₂ minimum (SLPM)"),
            y=alt.Y("count()", title="Count")
        ).properties(title="CO₂ minimum distribution", height=300)
        st.altair_chart(chart_cmin, use_container_width=True)

        chart_cin = alt.Chart(df_mc).mark_bar().encode(
            x=alt.X("CO2_in_SLPM:Q", bin=alt.Bin(maxbins=40), title="CO₂ inlet (SLPM)"),
            y=alt.Y("count()", title="Count")
        ).properties(title="CO₂ inlet distribution", height=300)
        st.altair_chart(chart_cin, use_container_width=True)

    st.subheader("Product distributions")
    prod_to_plot = st.multiselect("Select products to plot", PRODUCT_LIST, default=[PRODUCT_LIST[0]], key="mc_plot_sel")
    if prod_to_plot:
        charts = []
        for p in prod_to_plot:
            c = alt.Chart(df_mc).mark_bar().encode(
                x=alt.X(f"{p}_SLPM:Q", bin=alt.Bin(maxbins=40), title=f"{p} (SLPM)"),
                y=alt.Y("count()", title="Count")
            ).properties(title=f"{p}", height=220)
            charts.append(c)
        st.altair_chart(alt.vconcat(*charts), use_container_width=True)

    st.download_button(
        "Download Monte Carlo results (CSV)",
        data=df_mc.to_csv(index=False).encode("utf-8"),
        file_name="cheese_monte_carlo.csv",
        mime="text/csv",
        key="mc_dl"
    )

# -------------------- Constants --------------------
with main_tabs[6]:
    st.header("Constants & Reference Reactions")
    st.markdown(f"""
- **Faraday constant (F):** {F:.5f} C·mol⁻¹ e⁻  
- **Molar volume bases:** STP = {MV_OPTIONS['STP (0°C, 1 atm) — 22.414 L/mol']:.3f} L·mol⁻¹; SATP = {MV_OPTIONS['SATP (25°C, 1 atm) — 24.465 L/mol']:.3f} L·mol⁻¹  
- **Products currently enabled:** {", ".join(PRODUCT_LIST)}
""")

    st.subheader("nₑ and CO₂ stoichiometry (fixed)")
    df_props = pd.DataFrame({
        "Product": PRODUCT_LIST,
        "nₑ (e⁻/mol product)": [PRODUCTS[p]["n_e"] for p in PRODUCT_LIST],
        "CO₂ per mol product":  [PRODUCTS[p]["co2_per_mol"] for p in PRODUCT_LIST],
        "E⁰ (V) [EE display only]":  [PRODUCTS[p]["E0"] for p in PRODUCT_LIST],
    })
    st.dataframe(df_props, hide_index=True, width='stretch')
