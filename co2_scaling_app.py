
# CHEESE — CO₂ Handling & Electrolysis Efficiency Scaling Evaluator
# Tagline: Because scaling electrolysis shouldn’t be this gouda! 🧀
# Author: Aditya Prajapati +ChatGPT (GPT-5 Thinking)
# Copyright (c) 2025 Aditya Prajapati


from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(
    page_title="CHEESE — CO₂ Handling & Electrolysis Efficiency Scaling Evaluator",
    page_icon="🧀",
    layout="wide"
)
# --- Author credit in sidebar ---
with st.sidebar:
    st.markdown(
        """
        ---
        **Created by [Aditya Prajapati](https://people.llnl.gov/prajapati3)**  
        | LLNL
        """,
        unsafe_allow_html=True
    )
# -------------------- Constants --------------------
F = 96485.33212  # C/mol e-
# Molar volume options (L/mol)
MV_OPTIONS = {
    "STP (0°C, 1 atm) — 22.414 L/mol": 22.414,
    "SATP (25°C, 1 atm) — 24.465 L/mol": 24.465,
}
# Electrons per mole of product
NE = {"CO": 2, "C2H4": 12}
# Stoichiometric CO2 per mole product (carbon balance)
STOICH_CO2_PER_PRODUCT = {"CO": 1.0, "C2H4": 2.0}
# Standard potentials used for energy efficiency metric (fixed)
E0_CO = 1.33   # V
E0_C2H4 = 1.17 # V

# -------------------- Helpers --------------------
def to_m2(area_value: float, area_unit: str) -> float:
    return area_value * 1e-4 if area_unit == "cm²" else area_value

def to_A_per_m2(j_value: float, j_unit: str) -> float:
    if j_unit == "mA/cm²":
        return j_value * 10.0            # 1 mA/cm² = 10 A/m²
    if j_unit == "A/cm²":
        return j_value * 1e4             # 1 A/cm² = 1e4 A/m²
    return j_value                        # A/m²

def fe_to_frac(fe_pct: float) -> float:
    return max(0.0, min(1.0, fe_pct / 100.0))

def amps(area_m2: float, j_A_m2: float) -> float:
    return area_m2 * j_A_m2

def prod_mol_s(I: float, fe_frac: float, ne_per_mol: int) -> float:
    """Moles of product per second from Faraday's law."""
    return (I * fe_frac) / (ne_per_mol * F)

def mol_s_to_slpm(n_dot: float, molar_volume_L: float) -> float:
    """Convert mol/s to standard liters per minute (SLPM)."""
    return n_dot * molar_volume_L * 60.0

def slpm_to_mol_s(flow_slpm: float, molar_volume_L: float) -> float:
    """Convert SLPM to mol/s."""
    return flow_slpm / (molar_volume_L * 60.0)

def total_power_watts(I: float, V: float, n_units: int) -> float:
    return I * V * n_units

@dataclass
class ElectrolyzerInputs:
    area_value: float
    area_unit: str
    j_value: float
    j_unit: str
    V_cell: float
    fe_co_pct: float
    fe_c2h4_pct: float
    n_units: int
    molar_vol_L: float
    # Feed specification (choose one via mode)
    mode: str  # "S" or "INLET"
    stoich_ratio: Optional[float] = None
    co2_in_slpm_input: Optional[float] = None

def compute_core_products(inp: ElectrolyzerInputs) -> Dict[str, float]:
    """Compute current, power, and product formation independent of CO2 feed mode."""
    A_m2 = to_m2(inp.area_value, inp.area_unit)
    j_A_m2 = to_A_per_m2(inp.j_value, inp.j_unit)
    I_unit = amps(A_m2, j_A_m2)  # per-unit current
    I_total = I_unit * inp.n_units

    fe_co, fe_c2h4 = fe_to_frac(inp.fe_co_pct), fe_to_frac(inp.fe_c2h4_pct)

    # Production rates (mol/s)
    n_CO = prod_mol_s(I_total, fe_co, NE["CO"])
    n_C2H4 = prod_mol_s(I_total, fe_c2h4, NE["C2H4"])

    # Convert to SLPM
    slpm_CO = mol_s_to_slpm(n_CO, inp.molar_vol_L)
    slpm_C2H4 = mol_s_to_slpm(n_C2H4, inp.molar_vol_L)

    # Theoretical minimum CO2 required by carbon balance
    co2_min_mol_s = n_CO * STOICH_CO2_PER_PRODUCT["CO"] + n_C2H4 * STOICH_CO2_PER_PRODUCT["C2H4"]
    co2_min_slpm = mol_s_to_slpm(co2_min_mol_s, inp.molar_vol_L)

    P_total_W = total_power_watts(I_unit, inp.V_cell, inp.n_units)

    # Energy efficiency (overall, weighted by FE contributions), fixed E0's
    EE_total = (E0_CO/inp.V_cell)*fe_co + (E0_C2H4/inp.V_cell)*fe_c2h4

    return {
        "A_m2": A_m2,
        "j_A_m2": j_A_m2,
        "I_unit_A": I_unit,
        "I_total_A": I_total,
        "P_total_W": P_total_W,
        "CO_mol_s": n_CO,
        "C2H4_mol_s": n_C2H4,
        "CO_slpm": slpm_CO,
        "C2H4_slpm": slpm_C2H4,
        "CO2_min_slpm": co2_min_slpm,
        "EE_total": EE_total,
    }

def apply_feed_mode(core: Dict[str, float], inp: ElectrolyzerInputs) -> Dict[str, float]:
    """Given core production and a feed mode, compute inlet, S, and utilization."""
    co2_min_slpm = core["CO2_min_slpm"]

    if inp.mode == "S":
        S = max(1.0, float(inp.stoich_ratio or 1.0))
        co2_in_slpm = S * co2_min_slpm
        util = 1.0 / S
        warn = None
    else:  # "INLET"
        co2_in_slpm_raw = max(0.0, float(inp.co2_in_slpm_input or 0.0))
        if co2_in_slpm_raw < co2_min_slpm:
            S = max(1e-12, co2_in_slpm_raw / co2_min_slpm)
            util = min(1.0, 1.0 / S)  # will be >1; cap at 1
            co2_in_slpm = co2_in_slpm_raw
            warn = (
                f"Provided CO₂ inlet ({co2_in_slpm_raw:.3f} SLPM) is below the theoretical minimum "
                f"({co2_min_slpm:.3f} SLPM). Product rates assume Faradaic production; physically you'd be CO₂-limited."
            )
        else:
            S = co2_in_slpm_raw / co2_min_slpm
            util = 1.0 / S
            co2_in_slpm = co2_in_slpm_raw
            warn = None

    return {
        "CO2_in_slpm": co2_in_slpm,
        "Stoich_S": S,
        "CO2_utilization": min(1.0, util),
        "warning": warn,
    }

def build_sensitivity_table_S(core: Dict[str, float], S_min: float, S_max: float, S_step: float) -> pd.DataFrame:
    """Sweep S and compute inlet flow and outlet composition; FE is held constant (from core)."""
    CO_slpm = core["CO_slpm"]
    C2H4_slpm = core["C2H4_slpm"]
    co2_min_slpm = core["CO2_min_slpm"]

    S_vals = np.arange(S_min, S_max + 1e-9, S_step)
    rows = []
    for S in S_vals:
        S = max(1.0, float(S))
        util = 1.0 / S
        co2_in_slpm = S * co2_min_slpm
        co2_out_slpm = co2_in_slpm - co2_min_slpm
        total_out_slpm = max(1e-12, co2_out_slpm + CO_slpm + C2H4_slpm)

        rows.append({
            "Stoich S (inlet/min)": S,
            "CO2 Utilization (frac)": util,
            "CO2 Inlet (SLPM)": co2_in_slpm,
            "CO2 Outlet (SLPM)": co2_out_slpm,
            "CO (SLPM)": CO_slpm,
            "C2H4 (SLPM)": C2H4_slpm,
            "Total Outlet (SLPM)": total_out_slpm,
            "CO vol%": 100 * CO_slpm / total_out_slpm,
            "C2H4 vol%": 100 * C2H4_slpm / total_out_slpm,
            "CO2 vol%": 100 * co2_out_slpm / total_out_slpm,
        })
    return pd.DataFrame(rows)

def build_sensitivity_table_U(core: Dict[str, float], Umin_pct: float, Umax_pct: float, Ustep_pct: float) -> pd.DataFrame:
    """Sweep Utilization (%) as the independent variable and compute outlet flows and composition."""
    CO_slpm = core["CO_slpm"]
    C2H4_slpm = core["C2H4_slpm"]
    co2_min_slpm = core["CO2_min_slpm"]

    U_vals_pct = np.arange(Umin_pct, Umax_pct + 1e-9, Ustep_pct)
    rows = []
    for U_pct in U_vals_pct:
        U = max(1e-6, min(1.0, U_pct / 100.0))
        S = 1.0 / U
        co2_in_slpm = S * co2_min_slpm
        co2_out_slpm = co2_in_slpm - co2_min_slpm
        total_out_slpm = max(1e-12, co2_out_slpm + CO_slpm + C2H4_slpm)
        rows.append({
            "Utilization (%)": U_pct,
            "Stoich S (inlet/min)": S,
            "CO2 Inlet (SLPM)": co2_in_slpm,
            "CO2 Outlet (SLPM)": co2_out_slpm,
            "CO (SLPM)": CO_slpm,
            "C2H4 (SLPM)": C2H4_slpm,
            "Total Outlet (SLPM)": total_out_slpm,
            "CO vol%": 100 * CO_slpm / total_out_slpm,
            "C2H4 vol%": 100 * C2H4_slpm / total_out_slpm,
            "CO2 vol%": 100 * co2_out_slpm / total_out_slpm,
        })
    return pd.DataFrame(rows)

# -------------------- UI --------------------
st.title("🧀 CHEESE: CO₂ Handling & Electrolysis Efficiency Scaling Evaluator")
st.caption("Because scaling electrolysis shouldn’t be this gouda! 🧀")
st.caption("CO₂ → CO and CO₂ → C₂H₄ | Two calculators + utilization sensitivity + area×stack + supply cap + constants")

main_tabs = st.tabs([
    "Calculator",
    "Calc: Size Active Area from CO₂ Inlet & Stoich",
    "Sensitivity: CO₂ Utilization",
    "Sensitivity: Area × Stack",
    "Sensitivity: CO₂ Supply Cap",
    "Constants"
])

# -------------------- Calculator --------------------
with main_tabs[0]:
    st.header("Calculator")
    with st.sidebar:
        st.header("Global Settings")
        mv_label = st.selectbox("Gas molar volume basis", list(MV_OPTIONS.keys()), index=0, key="mv0")
        molar_vol = MV_OPTIONS[mv_label]

        st.write("---")
        st.subheader("Stack Configuration")
        use_stack = st.checkbox("Use a stack (multiple identical units)?", value=True, key="stack0")
        n_units = st.number_input("Number of units in stack", min_value=1, value=10, step=1, key="units0")

    # Core inputs
    colA, colB, colC = st.columns(3)
    with colA:
        area_value = st.number_input("Active area per unit", min_value=0.0, value=100.0, step=1.0, help="Geometric active area per unit", key="area0")
        area_unit = st.selectbox("Area unit", ["cm²", "m²"], index=0, key="areau0")
    with colB:
        j_value = st.number_input("Current density", min_value=0.0, value=200.0, step=10.0, key="j0")
        j_unit = st.selectbox("j units", ["mA/cm²", "A/cm²", "A/m²"], index=0, key="ju0")
    with colC:
        V_cell = st.number_input("Cell voltage (V)", min_value=0.0, value=3.2, step=0.1, key="v0")

    colD, colE, colF = st.columns(3)
    with colD:
        fe_co_pct = st.number_input("FE to CO (%)", min_value=0.0, max_value=100.0, value=90.0, step=1.0, key="feco0")
    with colE:
        fe_c2h4_pct = st.number_input("FE to C₂H₄ (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="fec2h40")
    with colF:
        mode = st.radio("CO₂ feed input mode", ["Stoich (S)", "Inlet flow (SLPM)"], index=0, horizontal=True, key="mode0")

    # Feed inputs depending on mode
    mode_key = "S"
    stoich_ratio = None
    co2_in_slpm_input = None
    if mode == "Stoich (S)":
        stoich_ratio = st.number_input(
            "CO₂ Stoichiometric Ratio S (inlet/min)", min_value=1.0, value=2.0, step=0.1,
            help="S = CO₂_in / CO₂_min. S=1 → 100% utilization. Higher S → more excess CO₂ and lower utilization.", key="s0"
        )
        mode_key = "S"
    else:
        co2_in_slpm_input = st.number_input("CO₂ Inlet flow (SLPM)", min_value=0.0, value=10.0, step=0.5, key="inlet0")
        mode_key = "INLET"

    if not use_stack:
        n_units = 1

    inp = ElectrolyzerInputs(
        area_value=area_value,
        area_unit=area_unit,
        j_value=j_value,
        j_unit=j_unit,
        V_cell=V_cell,
        fe_co_pct=fe_co_pct,
        fe_c2h4_pct=fe_c2h4_pct,
        n_units=n_units,
        molar_vol_L=molar_vol,
        mode=mode_key,
        stoich_ratio=stoich_ratio,
        co2_in_slpm_input=co2_in_slpm_input,
    )

    # Calculations
    core = compute_core_products(inp)
    feed = apply_feed_mode(core, inp)

    # Results
    st.subheader("Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Per-Unit Current (A)", f"{core['I_unit_A']:.2f}")
    with c2:
        st.metric("Total Current (A)", f"{core['I_total_A']:.2f}")
    with c3:
        st.metric("Total Power (kW)", f"{core['P_total_W']/1000:.2f}")
    with c4:
        st.metric("Energy Efficiency (Σ E₀/E_cell·FE)", f"{core['EE_total']*100:.1f}%")

    c5, c6, c7 = st.columns(3)
    with c5:
        st.metric("CO Production (SLPM)", f"{core['CO_slpm']:.3f}")
    with c6:
        st.metric("C₂H₄ Production (SLPM)", f"{core['C2H4_slpm']:.3f}")
    with c7:
        st.metric("CO₂ Minimum (SLPM)", f"{core['CO2_min_slpm']:.3f}")

    c8, c9 = st.columns(2)
    with c8:
        st.metric("CO₂ Inlet (SLPM)", f"{feed['CO2_in_slpm']:.3f}")
    with c9:
        st.metric("Stoich S (inlet/min)", f"{feed['Stoich_S']:.3f}")

    if feed["warning"]:
        st.warning(feed["warning"])

# -------------------- Calc: Size Active Area from CO₂ Inlet & Stoich --------------------
with main_tabs[1]:
    st.header("Calc: Size Active Area from CO₂ Inlet & Stoich")
    st.caption("Give **CO₂ inlet (SLPM)**, **Stoich S**, **current density**, and FE. The tool computes the required **active area** to process that feed.")

    with st.sidebar:
        st.subheader("Sizing Settings")
        mv_label_s = st.selectbox("Gas molar volume basis (sizing)", list(MV_OPTIONS.keys()), index=0, key="mv_sizing")
        molar_vol_s = MV_OPTIONS[mv_label_s]
        use_stack_s = st.checkbox("Use a stack?", value=True, key="stack_sizing")
        n_units_s = st.number_input("# Units (for per-unit area)", min_value=1, value=10, step=1, key="units_sizing")

    col1, col2, col3 = st.columns(3)
    with col1:
        co2_in_slpm_sz = st.number_input("CO₂ Inlet (SLPM)", min_value=0.0, value=50.0, step=1.0, key="sz_inlet")
        S_sz = st.number_input("Stoich S (inlet/min)", min_value=1.0, value=2.0, step=0.1, key="sz_S")
    with col2:
        j_val_sz = st.number_input("Current density", min_value=0.0, value=200.0, step=10.0, key="sz_j")
        j_unit_sz = st.selectbox("j units", ["mA/cm²", "A/cm²", "A/m²"], index=0, key="sz_ju")
        V_cell_sz = st.number_input("Cell voltage (V)", min_value=0.0, value=3.2, step=0.1, key="sz_V")
    with col3:
        feco_sz = st.number_input("FE to CO (%)", min_value=0.0, max_value=100.0, value=90.0, step=1.0, key="sz_feco")
        fec2h4_sz = st.number_input("FE to C₂H₄ (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="sz_fec2h4")

    # Convert and compute
    j_A_m2_sz = to_A_per_m2(j_val_sz, j_unit_sz)
    co2_min_slpm_sz = co2_in_slpm_sz / max(1e-12, S_sz)
    co2_min_mol_s_sz = slpm_to_mol_s(co2_min_slpm_sz, molar_vol_s)

    # Determine total current required from carbon balance + FE split:
    fe_co_frac = fe_to_frac(feco_sz)
    fe_c2h4_frac = fe_to_frac(fec2h4_sz)
    denom = (fe_co_frac/2.0) + (fe_c2h4_frac/6.0)  # [FE_CO/2 + FE_C2H4/6]

    if denom <= 1e-12:
        st.error("FE split yields zero carbon products (FE_CO/2 + FE_C2H4/6 = 0). Increase FE to CO and/or C₂H₄.")
    else:
        I_total_sz = co2_min_mol_s_sz * F / denom  # A
        A_total_m2 = I_total_sz / max(1e-12, j_A_m2_sz)  # m²
        A_total_cm2 = A_total_m2 * 1e4
        units_used = n_units_s if use_stack_s else 1
        A_per_unit_m2 = A_total_m2 / units_used
        A_per_unit_cm2 = A_per_unit_m2 * 1e4

        # Product rates at this size (consistency check)
        n_CO_sz = prod_mol_s(I_total_sz, fe_co_frac, NE["CO"])
        n_C2H4_sz = prod_mol_s(I_total_sz, fe_c2h4_frac, NE["C2H4"])
        CO_slpm_sz = mol_s_to_slpm(n_CO_sz, molar_vol_s)
        C2H4_slpm_sz = mol_s_to_slpm(n_C2H4_sz, molar_vol_s)

        P_total_kW_sz = (I_total_sz * V_cell_sz) / 1000.0
        util_sz = 1.0 / S_sz

        # ---- Toggle to show resultant area in cm² instead of m² ----
        show_cm2 = st.toggle("Display resultant area in cm²", value=False, key="sz_toggle")
        if show_cm2:
            area_unit_label = "cm²"
            total_area_display = A_total_cm2
            per_unit_area_display = A_per_unit_cm2
            fmt_total = "{:,.0f}"        # typically large numbers in cm²
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

        st.caption("Per-unit metrics assume equal area per unit. If you uncheck 'Use a stack', results are for a single unit.")

        st.subheader("Resulting Product Rates (for sized area)")
        st.write(f"- CO: **{CO_slpm_sz:.3f} SLPM**")
        st.write(f"- C₂H₄: **{C2H4_slpm_sz:.3f} SLPM**")

# -------------------- Sensitivity: CO2 Utilization --------------------
with main_tabs[2]:
    st.header("Sensitivity: CO₂ Utilization")
    st.caption("Sweep **Utilization (%)** as the independent variable for outlet flows (left). The composition vs **Stoich S** plot (right) is also shown. FE is held constant.")

    # Baseline operating inputs for the sweep
    col1, col2, col3 = st.columns(3)
    with col1:
        area_u = st.number_input("Area per unit (cm²)", min_value=0.0, value=100.0, step=5.0, key="u_area")
        j_u = st.number_input("Current density (mA/cm²)", min_value=0.0, value=200.0, step=10.0, key="u_j")
        V_u = st.number_input("Cell voltage (V)", min_value=0.0, value=3.2, step=0.1, key="u_V")
    with col2:
        fe_co_u = st.number_input("FE to CO (%)", min_value=0.0, max_value=100.0, value=90.0, step=1.0, key="u_feco")
        fe_c2h4_u = st.number_input("FE to C₂H₄ (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="u_fec2h4")
        units_u = st.number_input("# Units", min_value=1, value=10, step=1, key="u_units")
    with col3:
        mv_labelu = st.selectbox("Gas molar volume basis", list(MV_OPTIONS.keys()), index=0, key="u_mv")
        molar_vol_u = MV_OPTIONS[mv_labelu]

    # Utilization sweep
    Umin = st.number_input("Utilization min (%)", min_value=1.0, value=20.0, step=1.0, key="u_min")
    Umax = st.number_input("Utilization max (%)", min_value=1.0, value=100.0, step=1.0, key="u_max")
    Ustep = st.number_input("Utilization step (%)", min_value=1.0, value=5.0, step=1.0, key="u_step")

    # Compute baseline core
    inp_u = ElectrolyzerInputs(
        area_value=area_u, area_unit="cm²", j_value=j_u, j_unit="mA/cm²",
        V_cell=V_u, fe_co_pct=fe_co_u, fe_c2h4_pct=fe_c2h4_u, n_units=units_u, molar_vol_L=molar_vol_u,
        mode="S", stoich_ratio=1.0
    )
    core_u = compute_core_products(inp_u)

    # Build utilization sweep for outlet flows
    df_util = build_sensitivity_table_U(core_u, Umin, Umax, Ustep)

    # Left chart: Outlet flows vs Utilization (%)
    df_flows = df_util.melt(
        id_vars=["Utilization (%)"],
        value_vars=["CO2 Outlet (SLPM)", "CO (SLPM)", "C2H4 (SLPM)"],
        var_name="Stream", value_name="SLPM"
    )
    chart_flows = alt.Chart(df_flows).mark_line(point=True).encode(
        x=alt.X("Utilization (%):Q"),
        y=alt.Y("SLPM:Q"),
        color="Stream:N",
        tooltip=["Utilization (%)", "Stream", "SLPM"]
    ).properties(title="Outlet flows vs Utilization (%)", height=300)

    # Right chart: Composition vs S (derived from Utilization)
    df_sens_S = build_sensitivity_table_S(core_u, S_min=1.0, S_max=max(1.0, 1.0/(Umin/100.0)), S_step=0.5)
    df_comp = df_sens_S.melt(
        id_vars=["Stoich S (inlet/min)"],
        value_vars=["CO2 vol%", "CO vol%", "C2H4 vol%"],
        var_name="Species", value_name="vol%"
    )
    chart_comp = alt.Chart(df_comp).mark_line(point=True).encode(
        x=alt.X("Stoich S (inlet/min):Q"),
        y=alt.Y("vol%:Q"),
        color="Species:N",
        tooltip=["Stoich S (inlet/min)", "Species", "vol%"]
    ).properties(title="Outlet composition vs Stoich S", height=300)

    left, right = st.columns(2)
    with left:
        st.altair_chart(chart_flows, use_container_width=True)
    with right:
        st.altair_chart(chart_comp, use_container_width=True)

    st.download_button(
        "Download utilization sweep (CSV)",
        data=df_util.to_csv(index=False).encode("utf-8"),
        file_name="utilization_sweep.csv",
        mime="text/csv"
    )

# -------------------- Sensitivity: Area × Stack --------------------
with main_tabs[3]:
    st.header("Sensitivity: Area × Stack")
    st.caption("Sweep active area per unit and # of units. Visualize production, power, and CO₂ needs.")
    mv_label1 = st.selectbox("Gas molar volume basis", list(MV_OPTIONS.keys()), index=0, key="mv1")
    molar_vol1 = MV_OPTIONS[mv_label1]

    # Baseline operating inputs (hold these fixed during the sweep)
    col1, col2, col3 = st.columns(3)
    with col1:
        j_value1 = st.number_input("Current density", min_value=0.0, value=200.0, step=10.0, key="j1")
        j_unit1 = st.selectbox("j units", ["mA/cm²", "A/cm²", "A/m²"], index=0, key="ju1")
        V_cell1 = st.number_input("Cell voltage (V)", min_value=0.0, value=3.2, step=0.1, key="v1")
    with col2:
        fe_co_pct1 = st.number_input("FE to CO (%)", min_value=0.0, max_value=100.0, value=90.0, step=1.0, key="feco1")
        fe_c2h4_pct1 = st.number_input("FE to C₂H₄ (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="fec2h41")
        S1 = st.number_input("Stoich S for sweep", min_value=1.0, value=2.0, step=0.1, key="s1")
    with col3:
        area_min = st.number_input("Area per unit - min (cm²)", min_value=0.0, value=25.0, step=5.0, key="amin")
        area_max = st.number_input("Area per unit - max (cm²)", min_value=0.0, value=400.0, step=10.0, key="amax")
        area_step = st.number_input("Area step (cm²)", min_value=1.0, value=25.0, step=1.0, key="astep")
        n_min = st.number_input("# Units - min", min_value=1, value=1, step=1, key="nmin")
        n_max = st.number_input("# Units - max", min_value=1, value=50, step=1, key="nmax")
        n_step = st.number_input("# Units step", min_value=1, value=5, step=1, key="nstep")

    area_vals_cm2 = np.arange(area_min, area_max + 1e-9, area_step)
    n_vals = np.arange(n_min, n_max + 1, n_step)

    rows = []
    for area_cm2 in area_vals_cm2:
        area_m2 = area_cm2 * 1e-4
        j_A_m2 = to_A_per_m2(j_value1, j_unit1)
        I_unit = amps(area_m2, j_A_m2)
        for n_units1 in n_vals:
            I_total = I_unit * n_units1
            fe_co1, fe_c2h41 = fe_to_frac(fe_co_pct1), fe_to_frac(fe_c2h4_pct1)
            n_CO = prod_mol_s(I_total, fe_co1, NE["CO"])
            n_C2H4 = prod_mol_s(I_total, fe_c2h41, NE["C2H4"])
            CO_slpm = mol_s_to_slpm(n_CO, molar_vol1)
            C2H4_slpm = mol_s_to_slpm(n_C2H4, molar_vol1)
            co2_min_slpm = mol_s_to_slpm(
                n_CO * STOICH_CO2_PER_PRODUCT["CO"] + n_C2H4 * STOICH_CO2_PER_PRODUCT["C2H4"],
                molar_vol1
            )
            co2_in_slpm = S1 * co2_min_slpm
            P_total_kW = (I_total * V_cell1) / 1000.0
            rows.append({
                "Area_cm2": area_cm2,
                "Units": int(n_units1),
                "CO_SLPM": CO_slpm,
                "C2H4_SLPM": C2H4_slpm,
                "CO2_min_SLPM": co2_min_slpm,
                "CO2_in_SLPM": co2_in_slpm,
                "Power_kW": P_total_kW
            })

    df_grid = pd.DataFrame(rows)

    metric = st.selectbox("Heatmap metric", ["CO_SLPM", "CO2_in_SLPM", "Power_kW"], index=0, key="hmmetric")
    heat = alt.Chart(df_grid).mark_rect().encode(
        x=alt.X("Area_cm2:O", title="Area per unit (cm²)"),
        y=alt.Y("Units:O", title="# of Units"),
        color=alt.Color(f"{metric}:Q", title=metric),
        tooltip=["Area_cm2", "Units", "CO_SLPM", "C2H4_SLPM", "CO2_in_SLPM", "Power_kW"]
    ).properties(height=420)
    st.altair_chart(heat, use_container_width=True)

    st.download_button(
        "Download Area×Stack grid (CSV)",
        data=df_grid.to_csv(index=False).encode("utf-8"),
        file_name="area_stack_grid.csv",
        mime="text/csv"
    )

# -------------------- Sensitivity: CO2 Supply Cap --------------------
with main_tabs[4]:
    st.header("Sensitivity: CO₂ Supply Cap")
    st.caption("Impose a maximum CO₂ inlet and evaluate feasibility, utilization, and recommendations.")
    mv_label2 = st.selectbox("Gas molar volume basis", list(MV_OPTIONS.keys()), index=0, key="mv2")
    molar_vol2 = MV_OPTIONS[mv_label2]

    col1, col2, col3 = st.columns(3)
    with col1:
        area_value2 = st.number_input("Area per unit (cm²)", min_value=0.0, value=100.0, step=5.0, key="area2")
        area_unit2 = "cm²"
        j_value2 = st.number_input("Current density", min_value=0.0, value=200.0, step=10.0, key="j2")
        j_unit2 = st.selectbox("j units", ["mA/cm²", "A/cm²", "A/m²"], index=0, key="ju2")
    with col2:
        V_cell2 = st.number_input("Cell voltage (V)", min_value=0.0, value=3.2, step=0.1, key="v2")
        fe_co_pct2 = st.number_input("FE to CO (%)", min_value=0.0, max_value=100.0, value=90.0, step=1.0, key="feco2")
        fe_c2h4_pct2 = st.number_input("FE to C₂H₄ (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="fec2h42")
    with col3:
        n_units2 = st.number_input("# Units", min_value=1, value=10, step=1, key="units2")
        co2_cap = st.number_input("CO₂ supply cap (SLPM)", min_value=0.0, value=50.0, step=1.0, key="cap2")

    # Compute baseline needs
    A_m2 = to_m2(area_value2, area_unit2)
    j_A_m2 = to_A_per_m2(j_value2, j_unit2)
    I_unit = amps(A_m2, j_A_m2)
    I_total = I_unit * n_units2
    fe_co2, fe_c2h42 = fe_to_frac(fe_co_pct2), fe_to_frac(fe_c2h4_pct2)
    n_CO = prod_mol_s(I_total, fe_co2, NE["CO"])
    n_C2H4 = prod_mol_s(I_total, fe_c2h42, NE["C2H4"])
    co2_min_slpm2 = mol_s_to_slpm(
        n_CO * STOICH_CO2_PER_PRODUCT["CO"] + n_C2H4 * STOICH_CO2_PER_PRODUCT["C2H4"],
        molar_vol2
    )
    P_total_kW2 = (I_total * V_cell2) / 1000.0
    S_min_cap = (co2_cap / co2_min_slpm2) if co2_min_slpm2 > 0 else np.inf
    util_max_cap = min(1.0, 1.0 / max(1e-12, S_min_cap))

    st.subheader("Results under cap")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("CO₂ Minimum (SLPM)", f"{co2_min_slpm2:.3f}")
    with c2:
        st.metric("CO₂ Cap (SLPM)", f"{co2_cap:.3f}")
    with c3:
        st.metric("Max Utilization allowed", f"{100*util_max_cap:.1f}%")
    with c4:
        st.metric("Total Power (kW)", f"{P_total_kW2:.2f}")

    if co2_cap < co2_min_slpm2:
        st.warning("Cap is below the theoretical minimum CO₂ required at these operating conditions. Reduce current (j), area, units, or increase FE to CO/C₂H₄ to be feasible.")
    else:
        st.success("Feasible. You may increase utilization up to the shown maximum by reducing S accordingly.")

# -------------------- Constants --------------------
with main_tabs[5]:
    st.header("Constants & Reference Reactions")
    st.markdown(f"""
- **Faraday constant (F):** {F:.5f} C·mol⁻¹ e⁻  
- **Molar volume bases:** STP = {MV_OPTIONS['STP (0°C, 1 atm) — 22.414 L/mol']:.3f} L·mol⁻¹; SATP = {MV_OPTIONS['SATP (25°C, 1 atm) — 24.465 L/mol']:.3f} L·mol⁻¹  
- **Electrons per product (nₑ):** CO = {NE['CO']} e⁻·mol⁻¹; C₂H₄ = {NE['C2H4']} e⁻·mol⁻¹  
- **Stoichiometric CO₂ per product:** 1 CO₂ → 1 CO; 2 CO₂ → 1 C₂H₄  
- **Energy efficiency definition:** EE = Σ (E₀ / E_cell × FE). Using fixed E₀ values: E₀(CO) = {E0_CO:.2f} V, E₀(C₂H₄) = {E0_C2H4:.2f} V  
- **Relationships:** Stoich **S = CO₂_in / CO₂_min**; Utilization **U = 1/S**
""")

    st.subheader("Reference overall reactions (showing e⁻ counts)")
    st.markdown(r"""
- **CO₂ → CO (2 e⁻ per CO):**  
  (alkaline): **CO₂ + H₂O + 2 e⁻ → CO + 2 OH⁻**
- **CO₂ → C₂H₄ (12 e⁻ per C₂H₄):**  
  (alkaline): **2 CO₂ + 8H₂O + 12 e⁻  → C₂H₄ + 12OH⁻**
""")
