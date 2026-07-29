# CHEESE — CO₂ Handling & Electrolyzer Efficiency Scaling Evaluator
# Tagline: Because scaling electrolysis shouldn’t be this gouda! 🧀
# Author: Aditya Prajapati 
# Copyright (c) 2025 Aditya Prajapati
# Dependencies: streamlit, numpy, pandas, altair, plotly

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import streamlit as st

# -------------------- Page setup --------------------
st.set_page_config(
    page_title="CHEESE — CO₂ Handling for scaling",
    page_icon="🧀",
    layout="wide",
)

st.title("🧀 CHEESE — CO₂ Handling & Electrolyzer Efficiency Scaling Evaluator")
st.caption("Because scaling electrolysis shouldn’t be this gouda!")

st.markdown(
    """
<style>
.block-container {max-width: 1250px;}
div[data-testid="stMetric"] {text-align:center;}
div[data-testid="stMetric"] > label {justify-content:center;}
.fe-grid .stNumberInput > div > div { min-width: 160px; }
</style>
""",
    unsafe_allow_html=True,
)
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
F = 96485.33212  # C/mol e-
R = 8.314462618  # J/(mol·K)
MW_CO2_G_MOL = 44.0095
SECONDS_PER_MIN = 60.0
EPS = 1e-12

# Molar volume options (L/mol)
MV_OPTIONS = {
    "STP (0°C, 1 atm) — 22.414 L/mol": 22.414,
    "SATP (25°C, 1 atm) — 24.465 L/mol": 24.465,
}

# -------------------- Sidebar controls (global) --------------------
st.sidebar.header("Global Settings")

basis_label = st.sidebar.selectbox(
    "Gas molar volume basis",
    options=list(MV_OPTIONS.keys()),
    index=0,
    key="global_basis",
    help="Used to convert between gas molar flow and volumetric flow, and to calculate gas densities from MW.",
)

mv_L_per_mol = MV_OPTIONS[basis_label]  # L/mol
mv_m3_per_mol = mv_L_per_mol / 1000.0  # m³/mol

# Global display units. Calculations remain internally in SLPM for gas and
# kg/h + L/h for condensed products.
if "_previous_gas_flow_unit" not in st.session_state:
    legacy_unit = "SCCM" if st.session_state.get("gs_use_sccm", False) else "SLPM"
    st.session_state["_previous_gas_flow_unit"] = legacy_unit
if "gs_gas_flow_unit" not in st.session_state:
    st.session_state["gs_gas_flow_unit"] = st.session_state["_previous_gas_flow_unit"]

GAS_FLOW_UNIT = st.sidebar.selectbox(
    "Gas-flow display unit",
    options=["SCCM", "SLPM"],
    key="gs_gas_flow_unit",
    help="All gas inputs, outputs, plots, and CSV files use this display unit. Internal calculations remain on a standard dry-flow basis.",
)
GAS_FLOW_SCALE = 1000.0 if GAS_FLOW_UNIT == "SCCM" else 1.0

LIQUID_FLOW_UNIT = st.sidebar.selectbox(
    "Liquid-product display unit",
    options=["mg/h", "g/h", "kg/h", "µL/min", "mL/h", "L/h"],
    index=0,
    key="gs_liquid_flow_unit",
    help="Choose a mass-rate or volume-rate unit. Volume rates use the liquid densities listed in Constants & Properties.",
)

# Convert existing user-entered gas-flow values when the display unit changes.
_previous_unit = st.session_state["_previous_gas_flow_unit"]
if GAS_FLOW_UNIT != _previous_unit:
    _unit_conversion = 1000.0 if GAS_FLOW_UNIT == "SCCM" else 1.0 / 1000.0
    for _flow_widget_key in (
        "calc_inlet", "cb_inlet", "sz_inlet", "cap_cap",
        "exp_v1", "exp_v2", "exp_liquid_carbon", "exp_salt_carbon",
        "exp_dissolved_carbon", "exp_product_crossover", "exp_other_loss",
        "exp_anode_co2",
    ):
        if _flow_widget_key in st.session_state:
            st.session_state[_flow_widget_key] = float(st.session_state[_flow_widget_key]) * _unit_conversion

    if "axs_metric" in st.session_state:
        _old_metric = str(st.session_state["axs_metric"])
        _old_suffix = f"_{_previous_unit}"
        if _old_metric.endswith(_old_suffix):
            st.session_state["axs_metric"] = _old_metric[:-len(_old_suffix)] + f"_{GAS_FLOW_UNIT}"

    st.session_state["_previous_gas_flow_unit"] = GAS_FLOW_UNIT

use_stack_global = st.sidebar.checkbox("Use a stack (multiple identical units)?", value=True, key="gs_stack")
n_units_global = st.sidebar.number_input("Number of units in stack", min_value=1, value=10, step=1, key="gs_units")

# Real gas settings are global because the same test conditions usually apply
# across sizing, carbon-balance, and sensitivity calculations. Standard-flow
# calculations remain on the selected STP/SATP basis.
with st.sidebar.expander("Real gas conditions (optional)", expanded=False):
    gas_temperature_C = st.number_input(
        "Gas temperature (°C)",
        min_value=-20.0, max_value=150.0, value=25.0, step=1.0,
        key="gs_gas_temperature_C",
        help="Used only to translate standard dry flow into actual wet volumetric flow.",
    )
    gas_relative_humidity_pct = st.number_input(
        "Relative humidity (%)",
        min_value=0.0, max_value=100.0, value=0.0, step=5.0,
        key="gs_gas_rh_pct",
    )
    gas_outlet_pressure_bar_abs = st.number_input(
        "Gas outlet pressure (bar absolute)",
        min_value=0.05, value=1.01325, step=0.05,
        key="gs_gas_outlet_p_bar_abs",
        help="Use absolute pressure, not gauge pressure.",
    )
    gas_pressure_drop_bar = st.number_input(
        "Measured gas ΔP: inlet − outlet (bar)",
        min_value=0.0, value=0.0, step=0.01,
        key="gs_gas_dp_bar",
        help="This is the pressure drop measured across the gas flow path. Inlet absolute pressure = outlet absolute pressure + ΔP.",
    )
    liquid_pressure_drop_bar = st.number_input(
        "Liquid-side ΔP (bar, optional reference)",
        min_value=0.0, value=0.0, step=0.01,
        key="gs_liquid_dp_bar",
        help="Recorded as a hydraulic reference. It is not treated as membrane differential pressure or pumping power without absolute pressures and liquid flowrate.",
    )
    st.caption("SLPM/SCCM remain standard dry-flow units. These settings add an actual wet-flow interpretation without changing the electrochemical material balance.")

gas_inlet_pressure_bar_abs = gas_outlet_pressure_bar_abs + gas_pressure_drop_bar

# -------------------- Helper: numeric sanitizer  --------------------
NUMERIC_COLS = {
    "MW (g/mol)",
    "nₑ⁻ to product",
    "LHV (MJ/kg)",
    "HHV (MJ/kg)",
    "ρ_liq (kg/L)",
    "E0 (V) [display]",
}

CLEAN_NULLS = {
    r"^\s*$": np.nan,
    "—": np.nan,
    "–": np.nan,
    "NA": np.nan,
    "N/A": np.nan,
    "n/a": np.nan,
}

def sanitize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in df.columns:
        if c in NUMERIC_COLS:
            series = out[c]
            if series.dtype == "O":
                series = (
                    series.replace(CLEAN_NULLS, regex=True)
                    .astype(str)
                    .str.replace(r"[^\d\.\-eE+]", "", regex=True)
                )
            out[c] = pd.to_numeric(series, errors="coerce").astype(float)
    return out

# -------------------- Product properties  --------------------

# MGO is the only E0 I calculated digging data through the internet. 
#Rest of the E0 are from this excellent review article: https://pubs.acs.org/doi/full/10.1021/acs.chemrev.8b00705
PRODUCTS: List[Dict] = [
    # Gases
    {"Product": "CO",         "Phase": "gas",    "MW (g/mol)": 28.010, "nₑ⁻ to product": 2,  "co2_per_mol": 1.0, "LHV (MJ/kg)": 10.1,  "HHV (MJ/kg)": 12.6, "ρ_liq (kg/L)": np.nan, "E0 (V) [display]": 1.33},
    {"Product": "H₂",         "Phase": "gas",    "MW (g/mol)": 2.016,  "nₑ⁻ to product": 2,  "co2_per_mol": 0.0, "LHV (MJ/kg)": 120.0, "HHV (MJ/kg)": 141.9,"ρ_liq (kg/L)": np.nan, "E0 (V) [display]": 1.23},
    {"Product": "CH₄",        "Phase": "gas",    "MW (g/mol)": 16.043, "nₑ⁻ to product": 8,  "co2_per_mol": 1.0, "LHV (MJ/kg)": 50.0,  "HHV (MJ/kg)": 55.5, "ρ_liq (kg/L)": np.nan, "E0 (V) [display]": 1.06},
    {"Product": "C₂H₄",       "Phase": "gas",    "MW (g/mol)": 28.054, "nₑ⁻ to product": 12, "co2_per_mol": 2.0, "LHV (MJ/kg)": 47.2,  "HHV (MJ/kg)": 51.9, "ρ_liq (kg/L)": np.nan, "E0 (V) [display]": 1.15},
    # Liquids (at ~25 °C)
    {"Product": "Methanol",   "Phase": "liquid", "MW (g/mol)": 32.042, "nₑ⁻ to product": 6,  "co2_per_mol": 1.0, "LHV (MJ/kg)": 19.9,  "HHV (MJ/kg)": 22.7, "ρ_liq (kg/L)": 0.791, "E0 (V) [display]": 1.20},
    {"Product": "Ethanol",    "Phase": "liquid", "MW (g/mol)": 46.069, "nₑ⁻ to product": 12, "co2_per_mol": 2.0, "LHV (MJ/kg)": 26.8,  "HHV (MJ/kg)": 29.7, "ρ_liq (kg/L)": 0.789, "E0 (V) [display]": 1.14},
    {"Product": "Formate",    "Phase": "liquid", "MW (g/mol)": 46.026, "nₑ⁻ to product": 2,  "co2_per_mol": 1.0, "LHV (MJ/kg)": 5.9,   "HHV (MJ/kg)": 6.3,  "ρ_liq (kg/L)": 1.220, "E0 (V) [display]": 1.35},
    {"Product": "MGO",        "Phase": "liquid", "MW (g/mol)": 72.060, "nₑ⁻ to product": 12, "co2_per_mol": 3.0, "LHV (MJ/kg)": np.nan,"HHV (MJ/kg)": np.nan,"ρ_liq (kg/L)": 1.050, "E0 (V) [display]": 1.25},
]

PRODUCT_LIST = [p["Product"] for p in PRODUCTS]
GASES = [p["Product"] for p in PRODUCTS if p["Phase"].lower() == "gas"]
LIQUIDS = [p["Product"] for p in PRODUCTS if p["Phase"].lower() == "liquid"]
PRODUCT_MAP = {p["Product"]: p for p in PRODUCTS}

# -------------------- Utility helpers --------------------
def to_m2(area_value: float, area_unit: str) -> float:
    return area_value * 1e-4 if area_unit == "cm²" else area_value

def to_A_per_m2(j_value: float, j_unit: str) -> float:
    if j_unit == "mA/cm²": return j_value * 10.0
    if j_unit == "A/cm²":  return j_value * 1e4
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

def slpm_to_display(flow_slpm: float) -> float:
    """Convert an internal SLPM value to the selected display unit."""
    return flow_slpm * GAS_FLOW_SCALE

def display_to_slpm(flow_display: float) -> float:
    """Convert a user-facing SLPM/SCCM value back to internal SLPM."""
    return flow_display / GAS_FLOW_SCALE

def format_gas_flow(flow_slpm: float) -> str:
    """Format an internal SLPM value in the selected display unit."""
    return f"{slpm_to_display(flow_slpm):,.3f}"


def liquid_rate_to_display(kg_h: float, L_h: Optional[float], unit: Optional[str] = None) -> float:
    """Convert internal condensed-product rates to the selected display unit."""
    chosen = unit or LIQUID_FLOW_UNIT
    kg_h = float(kg_h or 0.0)
    L_h_value = float(L_h or 0.0)
    if chosen == "kg/h":
        return kg_h
    if chosen == "g/h":
        return kg_h * 1e3
    if chosen == "mg/h":
        return kg_h * 1e6
    if chosen == "L/h":
        return L_h_value
    if chosen == "mL/h":
        return L_h_value * 1e3
    if chosen == "µL/min":
        return L_h_value * 1e6 / 60.0
    raise ValueError(f"Unsupported liquid rate unit: {chosen}")

def format_liquid_rate(kg_h: float, L_h: Optional[float]) -> str:
    value = liquid_rate_to_display(kg_h, L_h)
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    if abs(value) >= 10:
        return f"{value:,.2f}"
    return f"{value:,.4f}"

def make_sankey_figure(
    labels: List[str],
    source: List[int],
    target: List[int],
    values_slpm: List[float],
    node_colors: List[str],
    link_colors: List[str],
    title: str,
    height: int = 540,
) -> go.Figure:
    """Create a high-contrast carbon Sankey with readable text."""
    values_display = [slpm_to_display(max(0.0, float(v))) for v in values_slpm]
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                valueformat=".3f",
                valuesuffix=f" {GAS_FLOW_UNIT} CO₂-eq",
                textfont=dict(
                    family="Arial Black, Arial, Helvetica, sans-serif",
                    size=15,
                    color="black",
                ),
                node=dict(
                    label=labels,
                    color=node_colors,
                    pad=22,
                    thickness=22,
                    line=dict(color="#263238", width=0.8),
                    hovertemplate="%{label}<br>%{value:.3f} " + GAS_FLOW_UNIT + " CO₂-eq<extra></extra>",
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=values_display,
                    color=link_colors,
                    hovertemplate="%{source.label} → %{target.label}<br>%{value:.3f} " + GAS_FLOW_UNIT + " CO₂-eq<extra></extra>",
                ),
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(family="Arial Black, Arial, Helvetica, sans-serif", size=18, color="black")),
        font=dict(family="Arial Black, Arial, Helvetica, sans-serif", size=15, color="black"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="black", font=dict(family="Arial Black, Arial, Helvetica, sans-serif", size=13, color="black")),
        height=height,
        margin=dict(l=18, r=18, t=70, b=18),
    )
    fig.update_traces(
        textfont=dict(
            family="Arial Black, Arial, Helvetica, sans-serif",
            size=15,
            color="black",
        ),
        selector=dict(type="sankey"),
    )
    return fig

def gas_flow_number_input(
    base_label: str,
    default_slpm: float,
    step_slpm: float,
    key: str,
) -> float:
    """Create a gas-flow input that follows the global gas-unit dropdown."""
    if key not in st.session_state:
        st.session_state[key] = slpm_to_display(default_slpm)
    return st.number_input(
        f"{base_label} ({GAS_FLOW_UNIT})",
        min_value=0.0,
        step=slpm_to_display(step_slpm),
        key=key,
    )

def water_vapor_pressure_kpa(temperature_C: float) -> float:
    """Approximate saturation vapor pressure of water from −20 to 150 °C."""
    T = min(150.0, max(-20.0, float(temperature_C)))
    if T < 1.0:
        # Buck-type expression, suitable for sub-ambient temperatures.
        return 0.61115 * np.exp((23.036 - T / 333.7) * T / (279.82 + T))
    if T <= 99.0:
        A, B, C = 8.07131, 1730.63, 233.426
    else:
        A, B, C = 8.14019, 1810.94, 244.485
    p_mmHg = 10 ** (A - B / (C + T))
    return p_mmHg * 0.133322368

def standard_dry_to_actual_wet_lpm(
    flow_slpm: float,
    temperature_C: float,
    pressure_bar_abs: float,
    relative_humidity_pct: float,
) -> Tuple[float, float, Optional[str]]:
    """Convert standard dry-gas flow to actual wet L/min at T, P, and RH.

    Returns (actual wet L/min, water vapor vol%, optional warning).
    """
    if flow_slpm <= EPS:
        return 0.0, 0.0, None

    T_K = float(temperature_C) + 273.15
    P_total_Pa = max(float(pressure_bar_abs), EPS) * 1e5
    p_sat_kPa = water_vapor_pressure_kpa(temperature_C)
    p_h2o_Pa = max(0.0, min(1.0, relative_humidity_pct / 100.0)) * p_sat_kPa * 1000.0

    warning = None
    if p_h2o_Pa >= 0.95 * P_total_Pa:
        p_h2o_Pa = 0.95 * P_total_Pa
        warning = "Water-vapor pressure approached total pressure; RH contribution was capped for numerical stability."

    P_dry_Pa = max(P_total_Pa - p_h2o_Pa, EPS)
    n_dry_mol_s = slpm_to_mol_s(flow_slpm, mv_L_per_mol)
    actual_m3_s = n_dry_mol_s * R * T_K / P_dry_Pa
    actual_L_min = actual_m3_s * 1000.0 * 60.0
    water_vol_pct = 100.0 * p_h2o_Pa / P_total_Pa
    return actual_L_min, water_vol_pct, warning

def product_specific_energy_rows(
    core: Dict[str, float],
    fe_map_pct: Dict[str, float],
    V_cell: float,
) -> pd.DataFrame:
    """Build product-level production and electrical-energy metrics."""
    I_total = core.get("I_total_A", 0.0)
    power_W = I_total * V_cell
    rows = []

    for p in PRODUCT_LIST:
        fe_frac = fe_to_frac(fe_map_pct.get(p, 0.0))
        if fe_frac <= EPS:
            continue

        props = PRODUCT_MAP[p]
        n_e = props["nₑ⁻ to product"]
        MW_kg_mol = props["MW (g/mol)"] / 1000.0
        n_p = core.get(f"{p}_mol_s", 0.0)
        mass_kg_h = n_p * MW_kg_mol * 3600.0

        sec_kWh_kg = (
            n_e * F * V_cell / (fe_frac * MW_kg_mol * 3.6e6)
            if MW_kg_mol > EPS else np.nan
        )
        thermo_eff_pct = (
            100.0 * fe_frac * props["E0 (V) [display]"] / V_cell
            if V_cell > EPS and pd.notna(props["E0 (V) [display]"]) else np.nan
        )

        lhv = props["LHV (MJ/kg)"]
        hhv = props["HHV (MJ/kg)"]
        mass_kg_s = mass_kg_h / 3600.0
        lhv_contribution = (100.0 * mass_kg_s * lhv * 1e6 / power_W) if power_W > EPS and pd.notna(lhv) else np.nan
        hhv_contribution = (100.0 * mass_kg_s * hhv * 1e6 / power_W) if power_W > EPS and pd.notna(hhv) else np.nan

        if props["Phase"].lower() == "gas":
            display_rate = f"{format_gas_flow(core.get(f'{p}_slpm', 0.0))} {GAS_FLOW_UNIT}"
        else:
            rho = props["ρ_liq (kg/L)"]
            liquid_L_h = mass_kg_h / rho if pd.notna(rho) and rho > EPS else 0.0
            display_rate = f"{format_liquid_rate(mass_kg_h, liquid_L_h)} {LIQUID_FLOW_UNIT}"

        rows.append({
            "Product": p,
            "FE (%)": 100.0 * fe_frac,
            "Displayed production rate": display_rate,
            "Production (kg/h)": mass_kg_h,
            "Specific electricity (kWh/kg)": sec_kWh_kg,
            "Thermodynamic efficiency contribution (%)": thermo_eff_pct,
            "LHV efficiency contribution (%)": lhv_contribution,
            "HHV efficiency contribution (%)": hhv_contribution,
        })

    return pd.DataFrame(rows)

def first_positive_minimum(candidates: Dict[str, float]) -> Tuple[str, float]:
    valid = {k: v for k, v in candidates.items() if np.isfinite(v) and v > EPS}
    if not valid:
        return "No active replacement trigger", np.inf
    trigger = min(valid, key=valid.get)
    return trigger, valid[trigger]

def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """NumPy-version-compatible trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))

def mflow_to_mass_and_vol(n_mol_s: float, MW_g_mol: float, rho_liq_kg_L: Optional[float]) -> Tuple[float, Optional[float]]:
    kg_h = n_mol_s * MW_g_mol * 3600.0 / 1000.0
    if rho_liq_kg_L is None or rho_liq_kg_L <= 0:
        return kg_h, None
    L_h = kg_h / rho_liq_kg_L
    return kg_h, L_h

def total_power_watts(I: float, V: float, n_units: int) -> float:
    return I * V * max(1, n_units)

# ---------- UI helpers for synchronized FE inputs ----------
FE_SYNC_SECTIONS = ["calc", "cb", "sz", "u", "axs"]

def _global_fe_key(product: str) -> str:
    return f"shared_fe_{product}"

def _section_fe_key(section_key: str, product: str) -> str:
    return f"{section_key}_fe_{product}"

def initialize_shared_fe_state() -> None:
    """Initialize one system FE state and mirror it into each tab's widgets."""
    for product in PRODUCT_LIST:
        global_key = _global_fe_key(product)
        if global_key not in st.session_state:
            recovered_value = None
            for section in FE_SYNC_SECTIONS:
                candidate_key = _section_fe_key(section, product)
                if candidate_key in st.session_state:
                    recovered_value = float(st.session_state[candidate_key])
                    break
            st.session_state[global_key] = (90.0 if product == "CO" else 0.0) if recovered_value is None else recovered_value
        for section in FE_SYNC_SECTIONS:
            widget_key = _section_fe_key(section, product)
            if widget_key not in st.session_state:
                st.session_state[widget_key] = float(st.session_state[global_key])

def sync_fe_from_tab(section_key: str, product: str) -> None:
    """Propagate an FE edit from any tab to every other FE grid."""
    source_key = _section_fe_key(section_key, product)
    value = float(st.session_state[source_key])
    st.session_state[_global_fe_key(product)] = value
    for section in FE_SYNC_SECTIONS:
        st.session_state[_section_fe_key(section, product)] = value
    if st.session_state.get("dur_product") == product:
        st.session_state["dur_FE0"] = value

def sync_durability_fe_to_system() -> None:
    product = st.session_state.get("dur_product", PRODUCT_LIST[0])
    value = float(st.session_state.get("dur_FE0", st.session_state[_global_fe_key(product)]))
    st.session_state[_global_fe_key(product)] = value
    for section in FE_SYNC_SECTIONS:
        st.session_state[_section_fe_key(section, product)] = value

def load_durability_fe_for_product() -> None:
    product = st.session_state.get("dur_product", PRODUCT_LIST[0])
    st.session_state["dur_FE0"] = float(st.session_state[_global_fe_key(product)])

initialize_shared_fe_state()

def fe_grid_inputs(
    section_key: str,
    products: List[str],
    default_map: Optional[Dict[str, float]] = None,
    title: str = "Faradaic Efficiencies (%, sum ≤ 100)",
    per_row: int = 3
) -> Dict[str, float]:
    st.markdown(f"#### {title}")
    st.caption("Shared system FE: changing a value here updates the same product FE in every analysis tab.")
    fe_map: Dict[str, float] = {}
    for i in range(0, len(products), per_row):
        row = products[i:i+per_row]
        cols = st.columns(len(row), gap="small")
        st.markdown('<div class="fe-grid">', unsafe_allow_html=True)
        for c, product in enumerate(row):
            with cols[c]:
                widget_key = _section_fe_key(section_key, product)
                fe_map[product] = st.number_input(
                    f"{product} FE (%)",
                    min_value=0.0, max_value=100.0,
                    step=1.0,
                    key=widget_key,
                    on_change=sync_fe_from_tab,
                    args=(section_key, product),
                )
        st.markdown("</div>", unsafe_allow_html=True)
    return fe_map

# -------------------- Core calculators (multi-product, gas vs liquid) --------------------
@dataclass
class ElectrolyzerInputs:
    area_value: float
    area_unit: str
    j_value: float
    j_unit: str
    V_cell: float
    fe_map_pct: Dict[str, float]
    n_units: int
    molar_vol_L: float

def compute_core_products(inp: ElectrolyzerInputs) -> Dict[str, float]:
    A_m2 = to_m2(inp.area_value, inp.area_unit)
    j_A_m2 = to_A_per_m2(inp.j_value, inp.j_unit)
    I_unit = amps(A_m2, j_A_m2)
    I_total = I_unit * max(1, inp.n_units)

    co2_min_mol_s = 0.0
    out: Dict[str, float] = {}
    gas_total_slpm = 0.0

    for p in PRODUCT_LIST:
        fe_frac = fe_to_frac(inp.fe_map_pct.get(p, 0.0))
        n_e = PRODUCT_MAP[p]["nₑ⁻ to product"]
        co2_per = PRODUCT_MAP[p]["co2_per_mol"]
        phase = PRODUCT_MAP[p]["Phase"].lower()

        n_p = prod_mol_s(I_total, fe_frac, n_e)  # mol/s
        out[f"{p}_mol_s"] = n_p

        if phase == "gas":
            slpm = mol_s_to_slpm(n_p, inp.molar_vol_L)
            out[f"{p}_slpm"] = slpm
            gas_total_slpm += slpm
        else:
            out[f"{p}_slpm"] = 0.0

        co2_min_mol_s += n_p * co2_per

    out["Gas_products_total_SLPM"] = gas_total_slpm
    out["CO2_min_slpm"] = mol_s_to_slpm(co2_min_mol_s, inp.molar_vol_L)
    out["I_unit_A"] = I_unit
    out["I_total_A"] = I_total
    return out

def build_sensitivity_table_S(core: Dict[str, float], S_min: float, S_max: float, S_step: float) -> pd.DataFrame:
    co2_min_slpm = core["CO2_min_slpm"]
    gas_prod_slpm = {p: core.get(f"{p}_slpm", 0.0) for p in GASES}
    gas_prod_total_slpm = sum(gas_prod_slpm.values())

    S_vals = np.arange(S_min, S_max + 1e-9, S_step)
    rows = []
    for S in S_vals:
        S = max(1.0, float(S))
        util = 1.0 / S
        co2_in_slpm = S * co2_min_slpm
        co2_out_slpm = co2_in_slpm - co2_min_slpm
        gas_total_out_slpm = max(1e-9, co2_out_slpm + gas_prod_total_slpm)

        row = {
            "Stoich S (inlet/min)": S,
            "CO2 Utilization (frac)": util,
            f"CO2 Inlet ({GAS_FLOW_UNIT})": slpm_to_display(co2_in_slpm),
            f"CO2 Outlet ({GAS_FLOW_UNIT})": slpm_to_display(co2_out_slpm),
            f"Gas Total Outlet ({GAS_FLOW_UNIT})": slpm_to_display(gas_total_out_slpm),
            "CO2 vol%": 100 * co2_out_slpm / gas_total_out_slpm,
        }
        for p in GASES:
            row[f"{p} ({GAS_FLOW_UNIT})"] = slpm_to_display(gas_prod_slpm[p])
            row[f"{p} vol%"] = 100 * gas_prod_slpm[p] / gas_total_out_slpm
        rows.append(row)
    return pd.DataFrame(rows)

def build_sensitivity_table_U(core: Dict[str, float], Umin_pct: float, Umax_pct: float, Ustep_pct: float) -> pd.DataFrame:
    co2_min_slpm = core["CO2_min_slpm"]
    gas_prod_slpm = {p: core.get(f"{p}_slpm", 0.0) for p in GASES}
    gas_prod_total_slpm = sum(gas_prod_slpm.values())

    U_vals_pct = np.arange(Umin_pct, Umax_pct + 1e-9, Ustep_pct)
    rows = []
    for U_pct in U_vals_pct:
        U = max(1e-6, min(1.0, U_pct / 100.0))
        S = 1.0 / U
        co2_in_slpm = S * co2_min_slpm
        co2_out_slpm = co2_in_slpm - co2_min_slpm
        gas_total_out_slpm = max(1e-9, co2_out_slpm + gas_prod_total_slpm)

        row = {
            "Utilization (%)": U_pct,
            "Stoich S (inlet/min)": S,
            f"CO2 Inlet ({GAS_FLOW_UNIT})": slpm_to_display(co2_in_slpm),
            f"CO2 Outlet ({GAS_FLOW_UNIT})": slpm_to_display(co2_out_slpm),
            f"Gas Total Outlet ({GAS_FLOW_UNIT})": slpm_to_display(gas_total_out_slpm),
            "CO2 vol%": 100 * co2_out_slpm / gas_total_out_slpm,
        }
        for p in GASES:
            row[f"{p} ({GAS_FLOW_UNIT})"] = slpm_to_display(gas_prod_slpm[p])
            row[f"{p} vol%"] = 100 * gas_prod_slpm[p] / gas_total_out_slpm
        rows.append(row)
    return pd.DataFrame(rows)

# -------------------- Tabs --------------------
tab_instructions, tab_calc, tab_carbon, tab_size, tab_s2, tab_s3, tab_durability = st.tabs([
    "Instructions",
    "Calculator",
    "Carbon & Energy",
    "Area Sizing",
    "CO₂ Utilization",
    "Area × Stack",
    "Durability",
])

# -------------------- Tab: Instructions --------------------
with tab_instructions:
    with st.expander("How to Use the CHEESEboard", expanded=False):
        st.markdown("""
        ###  Quick Guide
    
        **Purpose:**  
        This dashboard helps estimate CO₂ electrolyzer scaling parameters, product outputs, and sensitivities.
        - Gas products: H₂, CO, CH₄, C₂H₄
        - Liquid products, Methanol, Ethanol, Formate, Methylglyoxal (MGO)
    
        **Tabs Overview:**
        - **Calculator:**  
          Input area, current density, cell voltage, and Faradaic efficiencies (FEs).  
          Choose between `Stoich (S)` or `Inlet Flow` modes to compute:
            - Gas and liquid product rates  
            - CO₂ utilization (%)  
            - Power and total current
            
        Stoich is the "Stoichiometry". It is the ratio of actual CO₂ fed to the theoretical minimum CO₂ required to produce the observed products.
               
                • S = 1 means 100% CO₂ utilization (no excess feed).
               
                • S > 1 means excess CO₂ feed and lower utilization (e.g., S = 2 → 50% utilization).
         
        
        - **Carbon & Energy:**  
          Choose **Plan from performance assumptions** for deployment scenarios or **Decode an experiment** to infer CO₂ loss/crossover from measured inlet flow, outlet flow, and GC composition. Both workflows provide carbon metrics and improved Sankey diagrams. The same tab reports product-specific energy efficiency and specific electricity consumption.

        - **Area Sizing:**  
          Provides the **required electrode area** per unit and total area for a given CO₂ inlet and stoichiometric ratio (S).  
          Includes per-product outputs in the selected gas and liquid display units.
    
        - **Sensitivity: CO₂ Utilization:**  
          Sweeps utilization (%) to show gas outlet composition and flowrate trends.
    
        - **Sensitivity: Area × Stack:**  
          Visualizes scaling trade-offs between cell area and number of units in the stack using a heatmap.
    
        - **Sensitivity: CO₂ Supply Cap:**  
          Determines the maximum achievable utilization given a CO₂ feed limitation.

        - **Durability:**  
          Converts voltage rise, FE loss, and carbon-efficiency loss into stack life, replacement frequency, lifetime production, and lifetime-average energy demand.
    
        - **Constants & Reference (this tab):**  
          Lists all physical constants, product properties, and data sources.
    
        💡**Tips:**  
        - You can download any result table via the “Download CSV” buttons.  
        - Hover over plots for tooltips showing precise data points.  
        - Use the sidebar dropdowns to choose gas-flow and liquid-product display units.
        - Adjust **molar volume basis (STP/SATP)** in the sidebar to update standard gas volumetric conversions.
        - Open **Real gas conditions** only when you want to translate standard dry flow into actual wet flow at your measured temperature, humidity, outlet pressure, and gas ΔP.
        - If you find any mistakes please feel free to [reach out](https://people.llnl.gov/prajapati3)!
    
        ---
        """)

    st.subheader("Constants & Properties")
    st.markdown(f"""
- **Faraday constant (F):** `{F:.5f}` C·mol⁻¹ e⁻  
- **Molar volume bases:**  
  • STP = `{MV_OPTIONS['STP (0°C, 1 atm) — 22.414 L/mol']:.3f}` L·mol⁻¹  
  • SATP = `{MV_OPTIONS['SATP (25°C, 1 atm) — 24.465 L/mol']:.3f}` L·mol⁻¹  
- **Current basis:** `{basis_label}`  
- **Gas flow display unit:** `{GAS_FLOW_UNIT}`  
- **Liquid-product display unit:** `{LIQUID_FLOW_UNIT}`  
- **Stacking:** `{'ON' if use_stack_global else 'OFF'}` — Units: `{n_units_global}`  
- **Real gas interpretation:** `{gas_temperature_C:.1f} °C`, `{gas_relative_humidity_pct:.1f}% RH`, outlet `{gas_outlet_pressure_bar_abs:.4f} bar(a)`, gas ΔP `{gas_pressure_drop_bar:.4f} bar`  
- **Liquid-side ΔP reference:** `{liquid_pressure_drop_bar:.4f} bar`  
- **Gas products:** {", ".join(GASES) if GASES else "None"}  
- **Liquid products (treated as condensed):** {", ".join(LIQUIDS) if LIQUIDS else "None"}  
""")

    # Display name overrides for constants view
    def display_name(prod_key: str) -> str:
        if prod_key == "MGO":
            return "Methylglyoxal (MGO)"
        return prod_key

    c1, c2 = st.columns(2)
    with c1:
        gas_density_unit = st.selectbox(
            "Gas density display unit",
            options=["kg/m³", "g/L"],
            index=0,
            key="gas_density_unit",
        )
    with c2:
        liq_density_unit = st.selectbox(
            "Liquid density display unit",
            options=["kg/L", "g/mL"],
            index=0,
            key="liq_density_unit",
        )

    raw_df = pd.DataFrame.from_records(PRODUCTS)
    products_df = sanitize_numeric_columns(raw_df)

    mw_kg_per_mol = products_df["MW (g/mol)"] / 1000.0
    rho_gas_si = (mw_kg_per_mol / mv_m3_per_mol).where(products_df["Phase"].str.lower().eq("gas"), np.nan)

    # Build display dataframe with name override and units
    display_df = products_df[[
        "Product","Phase","MW (g/mol)","nₑ⁻ to product","co2_per_mol","LHV (MJ/kg)","HHV (MJ/kg)","E0 (V) [display]"
    ]].copy()
    display_df.insert(0, "Name", display_df["Product"].apply(display_name))
    display_df.drop(columns=["Product"], inplace=True)

    # Add densities with chosen units (1 kg/m³ = 1 g/L; 1 kg/L = 1 g/mL)
    display_df[f"ρ (gas @ {basis_label.split('—')[0].strip()}) [{gas_density_unit}]"] = rho_gas_si
    display_df[f"ρ (liquid) [{liq_density_unit}]"] = products_df["ρ_liq (kg/L)"]

    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "MW (g/mol)": st.column_config.NumberColumn("MW (g/mol)", format="%.3f"),
            "nₑ⁻ to product": st.column_config.NumberColumn("nₑ⁻ to product", format="%d"),
            "co2_per_mol": st.column_config.NumberColumn("CO₂ per mol product", format="%.2f"),
            "LHV (MJ/kg)": st.column_config.NumberColumn("LHV (MJ/kg)", format="%.2f"),
            "HHV (MJ/kg)": st.column_config.NumberColumn("HHV (MJ/kg)", format="%.2f"),
            "E0 (V) [display]": st.column_config.NumberColumn("E⁰ (V) [display only]", format="%.2f"),
        },
    )

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download constants table (CSV)",
        data=csv,
        file_name="cheese_constants.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_constants_csv",
    )
    st.markdown("---")
    st.subheader("References")
    st.markdown("""
    1. [Nitopi, Stephanie, et al. "Progress and perspectives of electrochemical CO2 reduction on copper in aqueous electrolyte." 
    Chemical reviews 119.12 (2019): 7610-7672.](https://pubs.acs.org/doi/full/10.1021/acs.chemrev.8b00705)
    
    2. [Perry, John H. "Chemical engineers' handbook." (1950): 533.](https://pubs.acs.org/doi/pdf/10.1021/ed027p533.1): 
    Link is just an exerpt but a good starting point for one to go out in the wild to find this book.

    3. [Data, C. P. T. NIST Chemistry WebBook, NIST Standard Reference Database Number 69, 2005.](https://webbook.nist.gov/chemistry/)
    """)
    

# -------------------- Tab: Calculator (Area/j with S or Inlet) --------------------
with tab_calc:
    st.subheader("Calculator: Provide Area, j, FE; choose Stoich S or Inlet")
    st.caption("Multi-product, true gas vs liquid handling. Shows per-product outputs.")

    colA, colB, colC = st.columns(3)
    with colA:
        area_value = st.number_input("Active area per unit", min_value=0.0, value=100.0, step=1.0, key="calc_area")
        area_unit  = st.selectbox("Area unit", ["cm²", "m²"], index=0, key="calc_area_unit")
    with colB:
        j_value = st.number_input("Current density", min_value=0.0, value=200.0, step=10.0, key="calc_j")
        j_unit  = st.selectbox("j units", ["mA/cm²", "A/cm²", "A/m²"], index=0, key="calc_j_unit")
    with colC:
        V_cell  = st.number_input("Cell voltage (V)", min_value=0.0, value=3.2, step=0.1, key="calc_V")

    fe_map_pct: Dict[str, float] = fe_grid_inputs("calc", PRODUCT_LIST, title="FE split (%)")

    st.divider()

    # Migrate an older saved radio value if this app is hot-reloaded in an existing session.
    if st.session_state.get("calc_mode") in {"Stoich (S)", "Inlet flow (SLPM)", "Inlet flow (SCCM)"}:
        st.session_state["calc_mode"] = (
            "stoich" if st.session_state["calc_mode"] == "Stoich (S)" else "inlet"
        )

    mode = st.radio(
        "CO₂ feed input mode",
        options=["stoich", "inlet"],
        index=0,
        horizontal=True,
        key="calc_mode",
        format_func=lambda choice: "Stoich (S)" if choice == "stoich" else f"Inlet flow ({GAS_FLOW_UNIT})",
    )
    if mode == "stoich":
        S = st.number_input("CO₂ Stoich S (inlet/min)", min_value=1.0, value=2.0, step=0.1, key="calc_S")
        co2_in_slpm_input = None
    else:
        co2_in_display_input = gas_flow_number_input(
            "CO₂ Inlet flow",
            default_slpm=10.0,
            step_slpm=0.5,
            key="calc_inlet",
        )
        co2_in_slpm_input = display_to_slpm(co2_in_display_input)
        S = None

    n_units_effective = n_units_global if use_stack_global else 1
    inp = ElectrolyzerInputs(
        area_value=area_value, area_unit=area_unit,
        j_value=j_value, j_unit=j_unit,
        V_cell=V_cell, fe_map_pct=fe_map_pct,
        n_units=n_units_effective, molar_vol_L=mv_L_per_mol,
    )
    core = compute_core_products(inp)

    # Determine inlet, S, utilization
    if core["CO2_min_slpm"] <= EPS:
        co2_in_slpm = 0.0 if (mode == "stoich") else float(co2_in_slpm_input or 0.0)
        Stoich_S = (np.inf if co2_in_slpm > 0 else 1.0) if mode != "stoich" else float(S or 1.0)
        util = 0.0 if co2_in_slpm > 0 else 1.0
        warn = "Total FE to carbon products is zero; CO₂ minimum is 0."
    else:
        if mode == "stoich":
            Stoich_S = max(1.0, float(S or 1.0))
            co2_in_slpm = Stoich_S * core["CO2_min_slpm"]
            util = 1.0 / Stoich_S
            warn = None
        else:
            co2_in_slpm = max(0.0, float(co2_in_slpm_input or 0.0))
            Stoich_S = co2_in_slpm / max(core["CO2_min_slpm"], EPS)
            util = min(1.0, 1.0 / max(Stoich_S, EPS))
            warn = None if co2_in_slpm >= core["CO2_min_slpm"] else (
                f"Inlet ({format_gas_flow(co2_in_slpm)} {GAS_FLOW_UNIT}) < "
                f"minimum ({format_gas_flow(core['CO2_min_slpm'])} {GAS_FLOW_UNIT})."
            )

    # GAS metrics
    st.subheader("Gas-side Results (true outlet)")
    g1, g2, g3, g4 = st.columns(4)
    with g1: st.metric(f"CO₂ Minimum ({GAS_FLOW_UNIT})", format_gas_flow(core["CO2_min_slpm"]))
    with g2: st.metric(f"CO₂ Inlet ({GAS_FLOW_UNIT})", format_gas_flow(co2_in_slpm))
    with g3: st.metric("Stoich S (inlet/min)", "∞" if not np.isfinite(Stoich_S) else f"{Stoich_S:.3f}")
    with g4: st.metric("Utilization (%)", f"{util*100:.1f}")

    g5, g6, g7 = st.columns(3)
    with g5: st.metric("Per-Unit Current (A)", f"{core['I_unit_A']:.2f}")
    with g6: st.metric("Total Current (A)", f"{core['I_total_A']:.2f}")
    with g7: st.metric("Power (kW)", f"{(core['I_total_A']*V_cell)/1000.0:.2f}")

    st.markdown(f"#### Gas product flowrates ({GAS_FLOW_UNIT})")
    gas_cols = st.columns(max(1, len(GASES)))
    for i, p in enumerate(GASES):
        with gas_cols[i]:
            st.metric(p, format_gas_flow(core.get(f"{p}_slpm", 0.0)))
    gas_total_out_slpm = sum(core.get(f"{p}_slpm", 0.0) for p in GASES) + max(co2_in_slpm - core["CO2_min_slpm"], 0.0)
    st.metric(f"Gas Total Outlet ({GAS_FLOW_UNIT})", format_gas_flow(gas_total_out_slpm))

    with st.expander("Actual wet flow at measured gas conditions", expanded=False):
        actual_in_L_min, water_in_pct, actual_warn_in = standard_dry_to_actual_wet_lpm(
            co2_in_slpm, gas_temperature_C, gas_inlet_pressure_bar_abs, gas_relative_humidity_pct
        )
        actual_out_L_min, water_out_pct, actual_warn_out = standard_dry_to_actual_wet_lpm(
            gas_total_out_slpm, gas_temperature_C, gas_outlet_pressure_bar_abs, gas_relative_humidity_pct
        )
        rg1, rg2, rg3, rg4 = st.columns(4)
        with rg1: st.metric("Inlet pressure", f"{gas_inlet_pressure_bar_abs:.3f} bar(a)")
        with rg2: st.metric("Gas ΔP", f"{gas_pressure_drop_bar:.3f} bar")
        with rg3: st.metric("Actual wet inlet", f"{actual_in_L_min:,.3f} L/min")
        with rg4: st.metric("Actual wet outlet", f"{actual_out_L_min:,.3f} L/min")
        st.caption(
            f"At {gas_temperature_C:.1f} °C and {gas_relative_humidity_pct:.1f}% RH, water vapor is approximately "
            f"{water_in_pct:.2f} vol% at the inlet and {water_out_pct:.2f} vol% at the outlet. "
            f"The electrochemical balance still uses dry standard flow in {GAS_FLOW_UNIT}."
        )
        if liquid_pressure_drop_bar > 0:
            st.info(
                f"Liquid-side ΔP is recorded as {liquid_pressure_drop_bar:.3f} bar. It is not interpreted as transmembrane pressure because absolute liquid and gas pressures are not both specified."
            )
        if actual_warn_in or actual_warn_out:
            st.warning(actual_warn_in or actual_warn_out)

    # LIQUID metrics
    st.subheader(f"Liquid production ({LIQUID_FLOW_UNIT})")
    liq_rows = []
    for p in LIQUIDS:
        n_p = core.get(f"{p}_mol_s", 0.0)
        MW = PRODUCT_MAP[p]["MW (g/mol)"]
        rho = PRODUCT_MAP[p]["ρ_liq (kg/L)"]
        kg_h, L_h = mflow_to_mass_and_vol(n_p, MW, rho)
        L_h_value = L_h if L_h is not None else 0.0
        liq_rows.append({
            "Product": p,
            f"Production rate ({LIQUID_FLOW_UNIT})": liquid_rate_to_display(kg_h, L_h_value),
            "mol/s": n_p,
            "ρ (kg/L)": (rho if rho else 0.0),
            "MW (g/mol)": MW,
        })
    if liq_rows:
        df_liq = pd.DataFrame(liq_rows)
        st.dataframe(
            df_liq, hide_index=True, use_container_width=True,
            column_config={f"Production rate ({LIQUID_FLOW_UNIT})": st.column_config.NumberColumn(format="%.5g")},
        )

    if warn:
        st.warning(warn)

# -------------------- Tab: Carbon balance and product-specific energy --------------------
with tab_carbon:
    st.subheader("Carbon Balance & Product-Specific Energy")
    st.caption("Use the planning model to explore a future system, or decode crossover directly from measured inlet/outlet data.")

    carbon_workflow = st.radio(
        "Choose a carbon-balance workflow",
        options=["planning", "experiment"],
        horizontal=True,
        key="cb_workflow",
        format_func=lambda x: "Plan from performance assumptions" if x == "planning" else "Decode an experiment",
    )

    with st.expander("How the two workflows differ", expanded=True):
        st.markdown(
            r"""
**Planning mode** starts from the applied current and the shared
Faradaic-efficiency distribution. The molar production rate of product
\(i\) is:
            """
        )

        st.latex(
            r"""
            \dot{n}_i
            =
            \frac{I\,FE_i}{n_{e,i}F}
            """
        )

        st.markdown(
            r"""
The total rate of CO₂ incorporated into all carbon-containing products is:
            """
        )

        st.latex(
            r"""
            \dot{n}_{\mathrm{CO_2,product}}
            =
            \sum_i
            \nu_{\mathrm{CO_2},i}\,
            \dot{n}_i
            """
        )

        st.markdown(
            r"""
where:

- \(I\) is the total current.
- \(FE_i\) is the Faradaic efficiency of product \(i\), expressed as a fraction.
- \(n_{e,i}\) is the number of electrons required per mole of product \(i\).
- \(F\) is Faraday's constant.
- \(\nu_{\mathrm{CO_2},i}\) is the number of CO₂ molecules incorporated per molecule of product \(i\).

You then specify carbonate/bicarbonate formation, product recovery,
anode-side carbon recovery, and CO₂ recycle. CHEESE uses these assumptions
to estimate carbon efficiency, fresh-feed demand, and the distribution of
carbon among the different pathways.

---

**Experimental mode** starts from the measured CO₂ inlet flow
\(V_1\), measured total cathode outlet flow \(V_2\), and the dry-gas
composition measured by GC.

For an outlet containing only CO₂, CO, and H₂, the leftover CO₂ flow is:
            """
        )

        st.latex(
            r"""
            V_{\mathrm{CO_2,leftover}}
            =
            V_2
            \left(
            1-c_{\mathrm{CO}}-c_{\mathrm{H_2}}
            \right)
            """
        )

        st.markdown(
            r"""
The CO₂-equivalent carbon converted to CO is:
            """
        )

        st.latex(
            r"""
            V_{\mathrm{CO_2,converted}}
            =
            V_2c_{\mathrm{CO}}
            """
        )

        st.markdown(
            r"""
The residual CO₂ loss obtained from the carbon balance is:
            """
        )

        st.latex(
            r"""
            V_{\mathrm{CO_2,loss}}
            =
            V_1
            -
            V_{\mathrm{CO_2,leftover}}
            -
            V_{\mathrm{CO_2,converted}}
            """
        )

        st.markdown(
            r"""
Substituting the expressions for leftover CO₂ and converted CO₂ gives:
            """
        )

        st.latex(
            r"""
            V_{\mathrm{CO_2,loss}}
            =
            V_1-V_2+V_2c_{\mathrm{H_2}}
            """
        )

        st.markdown(
            r"""
CHEESE also supports CH₄, C₂H₄, optional liquid-product carbon, and
corrections for salt precipitation, dissolved-carbon accumulation,
product crossover, directly measured anode CO₂, and other identified
carbon-loss pathways.

The experimentally calculated residual should initially be interpreted as
**unaccounted CO₂-equivalent carbon loss**. It represents inorganic-carbon
crossover only after other possible carbon-loss pathways have been ruled
out or explicitly corrected.
            """
        )

    st.markdown("### 1. Electrochemical operating point")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cb_area = st.number_input("Active area per unit", min_value=0.0, value=100.0, step=5.0, key="cb_area")
        cb_area_unit = st.selectbox("Area unit", ["cm²", "m²"], index=0, key="cb_area_unit")
    with c2:
        cb_j = st.number_input("Current density", min_value=0.0, value=200.0, step=10.0, key="cb_j")
        cb_j_unit = st.selectbox("j units", ["mA/cm²", "A/cm²", "A/m²"], index=0, key="cb_j_unit")
    with c3:
        cb_V = st.number_input("Cell voltage (V)", min_value=0.0, value=3.2, step=0.1, key="cb_V")
    with c4:
        cb_units = n_units_global if use_stack_global else 1
        st.info(f"Using global stack setting: **{cb_units} unit(s)**")

    cb_fe_map = fe_grid_inputs("cb", PRODUCT_LIST, title="Shared Faradaic-efficiency split (%)", per_row=4)
    cb_fe_sum = sum(cb_fe_map.values())
    if cb_fe_sum > 100.0 + 1e-9:
        st.warning(f"The shared FE sum is {cb_fe_sum:.1f}%. Values above 100% are not physically closed.")

    cb_inp = ElectrolyzerInputs(
        area_value=cb_area,
        area_unit=cb_area_unit,
        j_value=cb_j,
        j_unit=cb_j_unit,
        V_cell=cb_V,
        fe_map_pct=cb_fe_map,
        n_units=cb_units,
        molar_vol_L=mv_L_per_mol,
    )
    cb_core = compute_core_products(cb_inp)
    product_carbon_slpm = cb_core["CO2_min_slpm"]

    if carbon_workflow == "planning":
        st.markdown("### 2. Feed and assumed carbon pathways")
        feed_left, feed_right = st.columns([1, 2])
        with feed_left:
            if st.session_state.get("cb_feed_mode") in {"Stoich (S)", "Inlet flow (SLPM)", "Inlet flow (SCCM)"}:
                st.session_state["cb_feed_mode"] = "stoich" if st.session_state["cb_feed_mode"] == "Stoich (S)" else "inlet"
            cb_feed_mode = st.radio(
                "CO₂ feed input mode",
                options=["stoich", "inlet"],
                horizontal=True,
                key="cb_feed_mode",
                format_func=lambda x: "Stoich (S)" if x == "stoich" else f"Inlet flow ({GAS_FLOW_UNIT})",
            )
            if cb_feed_mode == "stoich":
                cb_S = st.number_input("Stoich S based on product carbon", min_value=1.0, value=2.0, step=0.1, key="cb_S")
                cb_feed_slpm = cb_S * product_carbon_slpm
            else:
                cb_feed_display = gas_flow_number_input("Gross CO₂ inlet", default_slpm=10.0, step_slpm=0.5, key="cb_inlet")
                cb_feed_slpm = display_to_slpm(cb_feed_display)
                cb_S = cb_feed_slpm / max(product_carbon_slpm, EPS) if product_carbon_slpm > EPS else np.inf

        with feed_right:
            p1, p2, p3 = st.columns(3)
            with p1:
                carbonate_ratio_pct = st.number_input(
                    "Carbonate/bicarbonate loss (% of product-bound carbon)",
                    min_value=0.0, value=50.0, step=5.0, key="cb_carbonate_ratio",
                    help="100% means one additional mole of CO₂ is diverted to carbonate/bicarbonate per mole of CO₂ incorporated into products. This is numerically equivalent to R₁ = 1 in the CO-only experimental model.",
                )
                dissolved_feed_pct = st.number_input(
                    "Dissolved/unaccounted carbon (% of gross feed)",
                    min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="cb_dissolved_pct",
                )
            with p2:
                product_recovery_pct = st.number_input(
                    "Carbonaceous-product recovery (%)",
                    min_value=0.0, max_value=100.0, value=100.0, step=1.0, key="cb_product_recovery",
                    help="Accounts for product crossover, incomplete capture, or downstream product loss.",
                )
                anode_recovery_pct = st.number_input(
                    "Anode CO₂ recovery from carbonate (%)",
                    min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="cb_anode_recovery",
                )
            with p3:
                unreacted_recycle_pct = st.number_input(
                    "Unreacted CO₂ recovered for recycle (%)",
                    min_value=0.0, max_value=100.0, value=0.0, step=5.0, key="cb_recycle_recovery",
                )
                st.caption("Recycle and anode recovery reduce fresh-feed demand; they do not change the gross single-pass balance.")

        carbonate_slpm = product_carbon_slpm * carbonate_ratio_pct / 100.0
        dissolved_slpm = cb_feed_slpm * dissolved_feed_pct / 100.0
        required_before_unreacted_slpm = product_carbon_slpm + carbonate_slpm + dissolved_slpm
        feasible_carbon_balance = cb_feed_slpm + EPS >= required_before_unreacted_slpm

        if product_carbon_slpm <= EPS:
            st.error("No carbon-containing products are being formed. Increase FE for at least one carbon product.")
        elif not feasible_carbon_balance:
            minimum_S_with_losses = required_before_unreacted_slpm / max(product_carbon_slpm, EPS)
            st.error(
                f"The gross feed is too small for the selected product and loss pathways. "
                f"Required minimum = {format_gas_flow(required_before_unreacted_slpm)} {GAS_FLOW_UNIT} "
                f"(effective S = {minimum_S_with_losses:.2f})."
            )
        else:
            unreacted_slpm = max(0.0, cb_feed_slpm - required_before_unreacted_slpm)
            recovered_product_slpm = product_carbon_slpm * product_recovery_pct / 100.0
            product_loss_slpm = product_carbon_slpm - recovered_product_slpm
            anode_recovered_slpm = carbonate_slpm * anode_recovery_pct / 100.0
            net_carbonate_loss_slpm = carbonate_slpm - anode_recovered_slpm
            recycle_slpm = unreacted_slpm * unreacted_recycle_pct / 100.0
            purge_slpm = unreacted_slpm - recycle_slpm
            net_fresh_feed_slpm = max(cb_feed_slpm - anode_recovered_slpm - recycle_slpm, EPS)

            single_pass_recovered_ce = recovered_product_slpm / max(cb_feed_slpm, EPS)
            cathodic_carbon_selectivity = product_carbon_slpm / max(product_carbon_slpm + carbonate_slpm, EPS)
            overall_fresh_ce = recovered_product_slpm / max(net_fresh_feed_slpm, EPS)
            gross_co2_consumption = (product_carbon_slpm + carbonate_slpm + dissolved_slpm) / max(cb_feed_slpm, EPS)

            carbon_product_mass_kg_h = 0.0
            for p in PRODUCT_LIST:
                if PRODUCT_MAP[p]["co2_per_mol"] > 0:
                    n_p = cb_core.get(f"{p}_mol_s", 0.0)
                    carbon_product_mass_kg_h += n_p * PRODUCT_MAP[p]["MW (g/mol)"] * 3600.0 / 1000.0
            recovered_product_mass_kg_h = carbon_product_mass_kg_h * product_recovery_pct / 100.0
            fresh_co2_mol_s = slpm_to_mol_s(net_fresh_feed_slpm, mv_L_per_mol)
            fresh_co2_kg_h = fresh_co2_mol_s * MW_CO2_G_MOL * 3600.0 / 1000.0
            kg_co2_per_kg_product = fresh_co2_kg_h / recovered_product_mass_kg_h if recovered_product_mass_kg_h > EPS else np.nan

            st.markdown("### 3. Carbon-efficiency results")
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("Recovered single-pass CE", f"{100*single_pass_recovered_ce:.1f}%")
            with m2: st.metric("Cathodic carbon selectivity", f"{100*cathodic_carbon_selectivity:.1f}%")
            with m3: st.metric("Overall fresh-feed CE", f"{100*overall_fresh_ce:.1f}%")
            with m4: st.metric("Gross CO₂ consumption", f"{100*gross_co2_consumption:.1f}%")

            m5, m6, m7, m8 = st.columns(4)
            with m5: st.metric(f"Gross CO₂ feed ({GAS_FLOW_UNIT})", format_gas_flow(cb_feed_slpm))
            with m6: st.metric(f"Net fresh CO₂ ({GAS_FLOW_UNIT})", format_gas_flow(net_fresh_feed_slpm))
            with m7: st.metric("Recovered carbon products", f"{recovered_product_mass_kg_h:,.4f} kg/h")
            with m8: st.metric("Fresh CO₂ / recovered product", "—" if np.isnan(kg_co2_per_kg_product) else f"{kg_co2_per_kg_product:,.3f} kg/kg")

            sankey_labels = [
                "Gross CO₂ feed", "Product-bound carbon", "Carbonate / bicarbonate",
                "Dissolved / unaccounted", "Unreacted CO₂", "Recovered product",
                "Product crossover / loss", "Recovered anode CO₂", "Net carbonate loss",
                "Recycle CO₂", "Purge CO₂",
            ]
            sankey_source = [0, 1, 1, 0, 2, 2, 0, 0, 4, 4]
            sankey_target = [1, 5, 6, 2, 7, 8, 3, 4, 9, 10]
            sankey_values_slpm = [
                product_carbon_slpm, recovered_product_slpm, product_loss_slpm,
                carbonate_slpm, anode_recovered_slpm, net_carbonate_loss_slpm,
                dissolved_slpm, unreacted_slpm, recycle_slpm, purge_slpm,
            ]
            sankey_fig = make_sankey_figure(
                labels=sankey_labels,
                source=sankey_source,
                target=sankey_target,
                values_slpm=sankey_values_slpm,
                node_colors=[
                    "#176B68", "#4C78A8", "#E3A33B", "#8D6E63", "#90A4AE",
                    "#2E8B57", "#D95F59", "#59A14F", "#C17C00", "#3B8EA5", "#7F8C8D",
                ],
                link_colors=[
                    "rgba(76,120,168,0.48)", "rgba(46,139,87,0.52)", "rgba(217,95,89,0.48)",
                    "rgba(227,163,59,0.52)", "rgba(89,161,79,0.50)", "rgba(193,124,0,0.48)",
                    "rgba(141,110,99,0.45)", "rgba(144,164,174,0.45)", "rgba(59,142,165,0.50)",
                    "rgba(127,140,141,0.45)",
                ],
                title="Planned carbon pathways on a CO₂-equivalent molar-flow basis",
            )
            st.plotly_chart(sankey_fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

            carbon_rows = [
                ("Gross CO₂ feed", cb_feed_slpm),
                ("Product-bound carbon", product_carbon_slpm),
                ("Recovered product carbon", recovered_product_slpm),
                ("Product crossover / recovery loss", product_loss_slpm),
                ("Carbonate / bicarbonate carbon", carbonate_slpm),
                ("Recovered anode CO₂", anode_recovered_slpm),
                ("Net carbonate loss", net_carbonate_loss_slpm),
                ("Dissolved / unaccounted carbon", dissolved_slpm),
                ("Unreacted CO₂", unreacted_slpm),
                ("Recycle CO₂", recycle_slpm),
                ("Purge CO₂", purge_slpm),
                ("Net fresh CO₂ requirement", net_fresh_feed_slpm),
            ]
            carbon_df = pd.DataFrame({
                "Carbon pathway": [r[0] for r in carbon_rows],
                f"CO₂-equivalent flow ({GAS_FLOW_UNIT})": [slpm_to_display(r[1]) for r in carbon_rows],
                "mol CO₂-eq/s": [slpm_to_mol_s(r[1], mv_L_per_mol) for r in carbon_rows],
            })
            with st.expander("View and download carbon-flow table", expanded=False):
                st.dataframe(carbon_df, hide_index=True, use_container_width=True)
                st.download_button(
                    "Download carbon balance (CSV)",
                    data=carbon_df.to_csv(index=False).encode("utf-8"),
                    file_name="cheese_carbon_balance_planning.csv",
                    mime="text/csv",
                    key="cb_download",
                )

            with st.expander("Actual wet-flow interpretation for this carbon case", expanded=False):
                actual_cb_in, cb_water_pct, cb_actual_warn = standard_dry_to_actual_wet_lpm(
                    cb_feed_slpm, gas_temperature_C, gas_inlet_pressure_bar_abs, gas_relative_humidity_pct
                )
                ac1, ac2, ac3 = st.columns(3)
                with ac1: st.metric("Actual wet gross inlet", f"{actual_cb_in:,.3f} L/min")
                with ac2: st.metric("Inlet pressure", f"{gas_inlet_pressure_bar_abs:.3f} bar(a)")
                with ac3: st.metric("Water vapor", f"{cb_water_pct:.2f} vol%")
                if cb_actual_warn:
                    st.warning(cb_actual_warn)

    else:
        st.markdown("### 2. Measured cathode gas balance")
        st.caption("Use dry, standard-basis inlet and outlet flows referenced to the same STP/SATP basis. GC composition should correspond to the outlet-flow measurement interval.")

        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            exp_v1_display = gas_flow_number_input("V₁ — CO₂ inlet", default_slpm=0.100, step_slpm=0.005, key="exp_v1")
            exp_v1_slpm = display_to_slpm(exp_v1_display)
            exp_v2_display = gas_flow_number_input("V₂ — total cathode outlet", default_slpm=0.090, step_slpm=0.005, key="exp_v2")
            exp_v2_slpm = display_to_slpm(exp_v2_display)
        with ex2:
            exp_co_pct = st.number_input("CO in cathode outlet (mol%)", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="exp_co_pct")
            exp_h2_pct = st.number_input("H₂ in cathode outlet (mol%)", min_value=0.0, max_value=100.0, value=2.0, step=0.5, key="exp_h2_pct")
        with ex3:
            st.metric("Total current used", f"{cb_core['I_total_A']:,.3f} A")
            st.metric("Current density", f"{cb_j:,.2f} {cb_j_unit}")
            st.caption("GC-derived FE values below use this total current and the global stack count.")

        with st.expander("Advanced products and corrections", expanded=False):
            st.markdown("**Additional measured outlet species**")
            a1, a2, a3 = st.columns(3)
            with a1:
                exp_ch4_pct = st.number_input("CH₄ in cathode outlet (mol%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="exp_ch4_pct")
            with a2:
                exp_c2h4_pct = st.number_input("C₂H₄ in cathode outlet (mol%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="exp_c2h4_pct")
            with a3:
                exp_other_gas_pct = st.number_input("Other non-CO₂ outlet gases (mol%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="exp_other_gas_pct", help="For example N₂, O₂, or an internal standard. These gases are excluded when CO₂ leftover is calculated.")

            st.markdown("**CO₂-equivalent carbon corrections**")
            st.caption(f"These entries use {GAS_FLOW_UNIT} CO₂-equivalent, not physical liquid volume. Convert measured carbon moles to the selected standard gas-flow equivalent.")
            b1, b2, b3 = st.columns(3)
            with b1:
                exp_liquid_carbon_display = gas_flow_number_input("Liquid-product carbon", default_slpm=0.0, step_slpm=0.001, key="exp_liquid_carbon")
                exp_salt_display = gas_flow_number_input("Carbon retained as salt", default_slpm=0.0, step_slpm=0.001, key="exp_salt_carbon")
            with b2:
                exp_dissolved_display = gas_flow_number_input("Dissolved-carbon accumulation", default_slpm=0.0, step_slpm=0.001, key="exp_dissolved_carbon")
                exp_product_xover_display = gas_flow_number_input("Measured product crossover", default_slpm=0.0, step_slpm=0.001, key="exp_product_crossover")
            with b3:
                exp_other_loss_display = gas_flow_number_input("Other identified carbon loss", default_slpm=0.0, step_slpm=0.001, key="exp_other_loss")
                exp_anode_co2_display = gas_flow_number_input("Directly measured anode CO₂", default_slpm=0.0, step_slpm=0.001, key="exp_anode_co2")

        # Defaults are required because Streamlit only instantiates advanced widgets after the expander content executes.
        exp_ch4_pct = float(st.session_state.get("exp_ch4_pct", 0.0))
        exp_c2h4_pct = float(st.session_state.get("exp_c2h4_pct", 0.0))
        exp_other_gas_pct = float(st.session_state.get("exp_other_gas_pct", 0.0))
        exp_liquid_carbon_slpm = display_to_slpm(float(st.session_state.get("exp_liquid_carbon", 0.0)))
        exp_salt_slpm = display_to_slpm(float(st.session_state.get("exp_salt_carbon", 0.0)))
        exp_dissolved_slpm = display_to_slpm(float(st.session_state.get("exp_dissolved_carbon", 0.0)))
        exp_product_xover_slpm = display_to_slpm(float(st.session_state.get("exp_product_crossover", 0.0)))
        exp_other_loss_slpm = display_to_slpm(float(st.session_state.get("exp_other_loss", 0.0)))
        exp_anode_co2_slpm = display_to_slpm(float(st.session_state.get("exp_anode_co2", 0.0)))

        c_co = exp_co_pct / 100.0
        c_h2 = exp_h2_pct / 100.0
        c_ch4 = exp_ch4_pct / 100.0
        c_c2h4 = exp_c2h4_pct / 100.0
        c_other = exp_other_gas_pct / 100.0
        measured_nonco2_fraction = c_co + c_h2 + c_ch4 + c_c2h4 + c_other

        if measured_nonco2_fraction > 1.0 + 1e-9:
            st.error(f"Measured non-CO₂ outlet composition sums to {100*measured_nonco2_fraction:.1f}%. It must be ≤100%.")
        elif exp_v1_slpm <= EPS or exp_v2_slpm <= EPS:
            st.error("Both V₁ and V₂ must be greater than zero.")
        else:
            co2_leftover_slpm = exp_v2_slpm * max(0.0, 1.0 - measured_nonco2_fraction)
            co_slpm = exp_v2_slpm * c_co
            h2_slpm = exp_v2_slpm * c_h2
            ch4_slpm = exp_v2_slpm * c_ch4
            c2h4_slpm = exp_v2_slpm * c_c2h4
            gas_product_carbon_slpm = co_slpm + ch4_slpm + 2.0 * c2h4_slpm
            total_product_carbon_slpm = gas_product_carbon_slpm + exp_liquid_carbon_slpm
            residual_loss_slpm = exp_v1_slpm - co2_leftover_slpm - total_product_carbon_slpm
            identified_non_xover_slpm = exp_salt_slpm + exp_dissolved_slpm + exp_product_xover_slpm + exp_other_loss_slpm
            corrected_inorganic_xover_slpm = residual_loss_slpm - identified_non_xover_slpm

            productive_conversion = total_product_carbon_slpm / max(exp_v1_slpm, EPS)
            gross_co2_consumption_exp = (total_product_carbon_slpm + residual_loss_slpm) / max(exp_v1_slpm, EPS)
            residual_carbon_selectivity = total_product_carbon_slpm / max(total_product_carbon_slpm + residual_loss_slpm, EPS)
            r1 = residual_loss_slpm / max(total_product_carbon_slpm, EPS)
            r2 = residual_loss_slpm / max(total_product_carbon_slpm + h2_slpm, EPS)
            r1_corrected = corrected_inorganic_xover_slpm / max(total_product_carbon_slpm, EPS)
            stoi_actual = (exp_v1_slpm - residual_loss_slpm) / max(total_product_carbon_slpm, EPS)
            stoi_gross = exp_v1_slpm / max(total_product_carbon_slpm, EPS)

            if residual_loss_slpm < -1e-9:
                st.error("The inferred carbon loss is negative. Check inlet/outlet flow calibration, wet-versus-dry basis, GC normalization, leakage, and time alignment.")
            if corrected_inorganic_xover_slpm < -1e-9:
                st.warning("Entered corrections exceed the inferred residual loss. The corrected inorganic-carbon crossover is negative, indicating inconsistent measurements or double counting.")

            st.markdown("### 3. Experimental crossover results")
            x1, x2, x3, x4 = st.columns(4)
            with x1: st.metric(f"CO₂ leftover ({GAS_FLOW_UNIT})", format_gas_flow(co2_leftover_slpm))
            with x2: st.metric(f"Product carbon ({GAS_FLOW_UNIT} CO₂-eq)", format_gas_flow(total_product_carbon_slpm))
            with x3: st.metric(f"Residual CO₂ loss ({GAS_FLOW_UNIT})", format_gas_flow(residual_loss_slpm))
            with x4: st.metric(f"Corrected inorganic crossover ({GAS_FLOW_UNIT})", format_gas_flow(corrected_inorganic_xover_slpm))

            x5, x6, x7, x8 = st.columns(4)
            with x5: st.metric("R₁ = loss / product carbon", f"{r1:.3f}")
            with x6: st.metric("R₂ = loss / (product + H₂)", f"{r2:.3f}")
            with x7: st.metric("Actual stoichiometry", f"{stoi_actual:.3f}")
            with x8: st.metric("Gross CO₂ consumption", f"{100*gross_co2_consumption_exp:.1f}%")

            x9, x10, x11, x12 = st.columns(4)
            with x9: st.metric("Productive CO₂ conversion", f"{100*productive_conversion:.1f}%")
            with x10: st.metric("Residual carbon selectivity", f"{100*residual_carbon_selectivity:.1f}%")
            with x11: st.metric("Gross stoichiometry", f"{stoi_gross:.3f}")
            with x12: st.metric("Corrected crossover / product", f"{r1_corrected:.3f}")

            if exp_anode_co2_slpm > EPS:
                anode_recovery_fraction = exp_anode_co2_slpm / max(corrected_inorganic_xover_slpm, EPS)
                st.info(
                    f"Direct anode CO₂ is {format_gas_flow(exp_anode_co2_slpm)} {GAS_FLOW_UNIT}, equal to "
                    f"{100*anode_recovery_fraction:.1f}% of the corrected inferred inorganic-carbon crossover. "
                    "Differences can arise from dissolved anolyte carbon, gas holdup, temporal lag, or incomplete anode degassing."
                )

            # GC-derived FE check for measured gas products.
            measured_product_flows = {"CO": co_slpm, "H₂": h2_slpm, "CH₄": ch4_slpm, "C₂H₄": c2h4_slpm}
            fe_check_rows = []
            for product, flow_slpm in measured_product_flows.items():
                n_dot = slpm_to_mol_s(flow_slpm, mv_L_per_mol)
                fe_gc = 100.0 * n_dot * PRODUCT_MAP[product]["nₑ⁻ to product"] * F / max(cb_core["I_total_A"], EPS)
                fe_check_rows.append({
                    "Product": product,
                    "Shared system FE (%)": cb_fe_map.get(product, 0.0),
                    "GC-derived FE (%)": fe_gc,
                    "Difference (points)": fe_gc - cb_fe_map.get(product, 0.0),
                    f"Measured flow ({GAS_FLOW_UNIT})": slpm_to_display(flow_slpm),
                })
            fe_check_df = pd.DataFrame(fe_check_rows)
            with st.expander("Compare GC-derived FE with the shared system FE", expanded=True):
                st.dataframe(
                    fe_check_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Shared system FE (%)": st.column_config.NumberColumn(format="%.2f"),
                        "GC-derived FE (%)": st.column_config.NumberColumn(format="%.2f"),
                        "Difference (points)": st.column_config.NumberColumn(format="%+.2f"),
                        f"Measured flow ({GAS_FLOW_UNIT})": st.column_config.NumberColumn(format="%.4f"),
                    },
                )
                st.caption("A systematic mismatch can indicate outlet-flow calibration, GC normalization, standard-condition mismatch, leaks, liquid products, or current/flow time misalignment.")

            if residual_loss_slpm >= -1e-9 and corrected_inorganic_xover_slpm >= -1e-9:
                corrected_for_plot = max(0.0, corrected_inorganic_xover_slpm)
                sankey_labels_exp = [
                    "CO₂ inlet", "Product carbon", "CO₂ leftover", "Residual carbon loss",
                    "Corrected inorganic crossover", "Salt carbon", "Dissolved accumulation",
                    "Product crossover", "Other identified loss",
                ]
                sankey_source_exp = [0, 0, 0, 3, 3, 3, 3, 3]
                sankey_target_exp = [1, 2, 3, 4, 5, 6, 7, 8]
                sankey_values_exp = [
                    total_product_carbon_slpm, co2_leftover_slpm, max(0.0, residual_loss_slpm),
                    corrected_for_plot, exp_salt_slpm, exp_dissolved_slpm,
                    exp_product_xover_slpm, exp_other_loss_slpm,
                ]
                sankey_exp = make_sankey_figure(
                    labels=sankey_labels_exp,
                    source=sankey_source_exp,
                    target=sankey_target_exp,
                    values_slpm=sankey_values_exp,
                    node_colors=[
                        "#176B68", "#2E8B57", "#90A4AE", "#E3A33B", "#C17C00",
                        "#8D6E63", "#7E57C2", "#D95F59", "#7F8C8D",
                    ],
                    link_colors=[
                        "rgba(46,139,87,0.52)", "rgba(144,164,174,0.48)", "rgba(227,163,59,0.52)",
                        "rgba(193,124,0,0.52)", "rgba(141,110,99,0.48)", "rgba(126,87,194,0.45)",
                        "rgba(217,95,89,0.48)", "rgba(127,140,141,0.45)",
                    ],
                    title="Experimentally decoded cathode carbon balance",
                )
                st.plotly_chart(sankey_exp, use_container_width=True, config={"displayModeBar": False, "responsive": True})

            experimental_rows = [
                ("CO₂ inlet V₁", exp_v1_slpm),
                ("Total cathode outlet V₂", exp_v2_slpm),
                ("CO₂ leftover", co2_leftover_slpm),
                ("Gas-product carbon", gas_product_carbon_slpm),
                ("Liquid-product carbon", exp_liquid_carbon_slpm),
                ("Total product carbon", total_product_carbon_slpm),
                ("Residual CO₂ loss", residual_loss_slpm),
                ("Salt carbon", exp_salt_slpm),
                ("Dissolved-carbon accumulation", exp_dissolved_slpm),
                ("Product crossover", exp_product_xover_slpm),
                ("Other identified carbon loss", exp_other_loss_slpm),
                ("Corrected inorganic-carbon crossover", corrected_inorganic_xover_slpm),
                ("Directly measured anode CO₂", exp_anode_co2_slpm),
            ]
            experimental_df = pd.DataFrame({
                "Measured or inferred pathway": [r[0] for r in experimental_rows],
                f"CO₂-equivalent flow ({GAS_FLOW_UNIT})": [slpm_to_display(r[1]) for r in experimental_rows],
                "mol CO₂-eq/s": [slpm_to_mol_s(r[1], mv_L_per_mol) for r in experimental_rows],
            })
            with st.expander("View equations, table, and download", expanded=False):
                st.markdown(r"""
                For the generalized gas case used here:

                \[
                V_{CO_2,leftover}=V_2\left(1-c_{CO}-c_{H_2}-c_{CH_4}-c_{C_2H_4}-c_{other}\right)
                \]
                \[
                V_{product\ carbon}=V_2\left(c_{CO}+c_{CH_4}+2c_{C_2H_4}\right)+V_{liquid\ carbon}
                \]
                \[
                V_{residual\ loss}=V_1-V_{CO_2,leftover}-V_{product\ carbon}
                \]
                \[
                V_{corrected\ inorganic\ crossover}=V_{residual\ loss}-V_{salt}-V_{dissolved}-V_{product\ crossover}-V_{other}
                \]
                """)
                st.dataframe(experimental_df, hide_index=True, use_container_width=True)
                st.download_button(
                    "Download experimental crossover balance (CSV)",
                    data=experimental_df.to_csv(index=False).encode("utf-8"),
                    file_name="cheese_experimental_crossover_decoder.csv",
                    mime="text/csv",
                    key="exp_download",
                )

    st.markdown("---")
    st.markdown("### Product-specific energy")
    st.caption("This section uses the shared FE split and electrochemical operating point above, independent of which carbon workflow is open.")
    energy_df = product_specific_energy_rows(cb_core, cb_fe_map, cb_V)
    if energy_df.empty:
        st.info("Enter a nonzero FE to calculate product-specific energy metrics.")
    else:
        total_lhv_eff = energy_df["LHV efficiency contribution (%)"].sum(skipna=True)
        total_hhv_eff = energy_df["HHV efficiency contribution (%)"].sum(skipna=True)
        e1, e2, e3 = st.columns(3)
        with e1: st.metric("Stack power", f"{cb_core['I_total_A']*cb_V/1000.0:,.3f} kW")
        with e2: st.metric("Total LHV efficiency", f"{total_lhv_eff:.1f}%")
        with e3: st.metric("Total HHV efficiency", f"{total_hhv_eff:.1f}%")
        st.dataframe(
            energy_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "FE (%)": st.column_config.NumberColumn(format="%.1f"),
                "Production (kg/h)": st.column_config.NumberColumn(format="%.5f"),
                "Specific electricity (kWh/kg)": st.column_config.NumberColumn(format="%.2f"),
                "Thermodynamic efficiency contribution (%)": st.column_config.NumberColumn(format="%.1f"),
                "LHV efficiency contribution (%)": st.column_config.NumberColumn(format="%.1f"),
                "HHV efficiency contribution (%)": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        st.caption("Specific electricity assigns the full electrochemical voltage requirement to formation of each product at its own FE. LHV/HHV columns show each product's contribution to total stack energy efficiency.")

        with st.expander("Explore voltage × FE energy sensitivity", expanded=False):
            energy_product = st.selectbox("Product", PRODUCT_LIST, index=0, key="cb_energy_product")
            es1, es2, es3, es4 = st.columns(4)
            with es1:
                V_min_s = st.number_input("Voltage min (V)", min_value=0.1, value=2.0, step=0.1, key="cb_Vmin")
                V_max_s = st.number_input("Voltage max (V)", min_value=0.1, value=4.0, step=0.1, key="cb_Vmax")
            with es2:
                V_step_s = st.number_input("Voltage step (V)", min_value=0.01, value=0.1, step=0.05, key="cb_Vstep")
            with es3:
                FE_min_s = st.number_input("FE min (%)", min_value=1.0, max_value=100.0, value=20.0, step=5.0, key="cb_FEmin")
                FE_max_s = st.number_input("FE max (%)", min_value=1.0, max_value=100.0, value=100.0, step=5.0, key="cb_FEmax")
            with es4:
                FE_step_s = st.number_input("FE step (%)", min_value=1.0, value=10.0, step=1.0, key="cb_FEstep")

            if V_max_s < V_min_s or FE_max_s < FE_min_s:
                st.warning("Maximum sensitivity values must be greater than or equal to minimum values.")
            else:
                props = PRODUCT_MAP[energy_product]
                n_e = props["nₑ⁻ to product"]
                MW_kg_mol = props["MW (g/mol)"] / 1000.0
                rows_energy_sens = []
                for V_s in np.arange(V_min_s, V_max_s + 1e-9, V_step_s):
                    for FE_s in np.arange(FE_min_s, FE_max_s + 1e-9, FE_step_s):
                        sec = n_e * F * V_s / ((FE_s / 100.0) * MW_kg_mol * 3.6e6)
                        rows_energy_sens.append({"Cell voltage (V)": V_s, "FE (%)": FE_s, "kWh/kg": sec})
                df_energy_sens = pd.DataFrame(rows_energy_sens)
                energy_heat = alt.Chart(df_energy_sens).mark_rect().encode(
                    x=alt.X("Cell voltage (V):O", title="Cell voltage (V)"),
                    y=alt.Y("FE (%):O", title=f"{energy_product} FE (%)"),
                    color=alt.Color("kWh/kg:Q", title="kWh/kg"),
                    tooltip=["Cell voltage (V)", "FE (%)", alt.Tooltip("kWh/kg:Q", format=".2f")],
                ).properties(height=390, title=f"{energy_product} specific electricity — lower is better")
                st.altair_chart(energy_heat, use_container_width=True)

# -------------------- Tab: Calc — Size Active Area from CO₂ Inlet & Stoich --------------------
with tab_size:
    st.subheader("Calc: Size Active Area from CO₂ Inlet & Stoich (with per-product outputs)")

    units_used = n_units_global if use_stack_global else 1
    col1, col2, col3 = st.columns(3)
    with col1:
        co2_in_display_sz = gas_flow_number_input(
            "CO₂ Inlet",
            default_slpm=50.0,
            step_slpm=1.0,
            key="sz_inlet",
        )
        co2_in_slpm_sz = display_to_slpm(co2_in_display_sz)
        S_sz = st.number_input("Stoich S (inlet/min)", min_value=1.0, value=2.0, step=0.1, key="sz_S")
    with col2:
        j_val_sz = st.number_input("Current density", min_value=0.0, value=200.0, step=10.0, key="sz_j")
        j_unit_sz = st.selectbox("j units", ["mA/cm²", "A/cm²", "A/m²"], index=0, key="sz_j_unit")
        V_cell_sz = st.number_input("Cell voltage (V)", min_value=0.0, value=3.2, step=0.1, key="sz_V")
    with col3:
        fe_map_sz = fe_grid_inputs("sz", PRODUCT_LIST, title="FE split (%)", per_row=3)

    j_A_m2_sz = to_A_per_m2(j_val_sz, j_unit_sz)
    co2_min_slpm_sz = co2_in_slpm_sz / max(S_sz, EPS)
    co2_min_mol_s_sz = slpm_to_mol_s(co2_min_slpm_sz, mv_L_per_mol)

    denom = 0.0  # Σ FE_i * (CO2_per_i / n_e_i)
    for p in PRODUCT_LIST:
        fe_frac = fe_to_frac(fe_map_sz.get(p, 0.0))
        denom += fe_frac * PRODUCT_MAP[p]["co2_per_mol"] / max(PRODUCT_MAP[p]["nₑ⁻ to product"], EPS)

    if denom <= EPS:
        st.error("FE split yields zero carbon products (Σ FE_i·CO₂_per_i/n_e_i = 0). Increase FEs.")
    else:
        I_total_sz = co2_min_mol_s_sz * F / denom
        A_total_m2 = I_total_sz / max(j_A_m2_sz, EPS)
        A_total_cm2 = A_total_m2 * 1e4
        A_per_unit_m2 = A_total_m2 / max(units_used, 1)
        A_per_unit_cm2 = A_per_unit_m2 * 1e4

        # Product rates at this size (gas/liquid separated)
        gas_rows, liq_rows = [], []
        gas_total_slpm = 0.0
        for p in PRODUCT_LIST:
            fe_frac = fe_to_frac(fe_map_sz.get(p, 0.0))
            n_e = PRODUCT_MAP[p]["nₑ⁻ to product"]
            n_p = prod_mol_s(I_total_sz, fe_frac, n_e)
            if PRODUCT_MAP[p]["Phase"].lower() == "gas":
                slpm_p = mol_s_to_slpm(n_p, mv_L_per_mol)
                gas_rows.append((p, slpm_p))
                gas_total_slpm += slpm_p
            else:
                kg_h, L_h = mflow_to_mass_and_vol(n_p, PRODUCT_MAP[p]["MW (g/mol)"], PRODUCT_MAP[p]["ρ_liq (kg/L)"])
                liq_rows.append((p, n_p, kg_h, L_h if L_h else 0.0))

        P_total_kW_sz = (I_total_sz * V_cell_sz) / 1000.0
        util_sz = 1.0 / max(S_sz, EPS)

        show_cm2 = st.toggle("Display resultant area in cm²", value=False, key="sz_area_toggle")
        if show_cm2:
            area_unit_label = "cm²"; total_area_display = A_total_cm2; per_unit_area_display = A_per_unit_cm2
            fmt_total, fmt_per_unit = "{:,.0f}", "{:,.0f}"
        else:
            area_unit_label = "m²"; total_area_display = A_total_m2; per_unit_area_display = A_per_unit_m2
            fmt_total, fmt_per_unit = "{:.3f}", "{:.4f}"

        st.subheader("Sizing Results")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(f"Total active area ({area_unit_label})", fmt_total.format(total_area_display))
            st.metric("Total current (A)", f"{I_total_sz:.1f}")
        with c2:
            st.metric(f"Per-unit area ({area_unit_label})", fmt_per_unit.format(per_unit_area_display))
            st.metric("Power (kW)", f"{P_total_kW_sz:.2f}")
        with c3:
            st.metric(f"CO₂ Minimum ({GAS_FLOW_UNIT})", format_gas_flow(co2_min_slpm_sz))
            st.metric("Utilization (%)", f"{util_sz*100:.1f}")

        st.subheader(f"Resulting Gas Products ({GAS_FLOW_UNIT})")
        if gas_rows:
            colsG = st.columns(len(gas_rows))
            for i, (p, slpm_p) in enumerate(gas_rows):
                with colsG[i]: st.metric(p, format_gas_flow(slpm_p))
            st.write(f"**Gas products total ({GAS_FLOW_UNIT})**: {format_gas_flow(gas_total_slpm)}")

        with st.expander("Actual wet inlet at measured gas conditions", expanded=False):
            actual_sz_in, water_sz_pct, actual_sz_warn = standard_dry_to_actual_wet_lpm(
                co2_in_slpm_sz, gas_temperature_C, gas_inlet_pressure_bar_abs, gas_relative_humidity_pct
            )
            as1, as2, as3 = st.columns(3)
            with as1: st.metric("Actual wet inlet", f"{actual_sz_in:,.3f} L/min")
            with as2: st.metric("Inlet pressure", f"{gas_inlet_pressure_bar_abs:.3f} bar(a)")
            with as3: st.metric("Water vapor", f"{water_sz_pct:.2f} vol%")
            if actual_sz_warn:
                st.warning(actual_sz_warn)

        st.subheader(f"Resulting Liquid Products ({LIQUID_FLOW_UNIT})")
        if liq_rows:
            df_liq_sz = pd.DataFrame([
                {
                    "Product": p,
                    f"Production rate ({LIQUID_FLOW_UNIT})": liquid_rate_to_display(kg, Lh),
                    "mol/s": n,
                }
                for (p, n, kg, Lh) in liq_rows
            ])
            st.dataframe(
                df_liq_sz, hide_index=True, use_container_width=True,
                column_config={f"Production rate ({LIQUID_FLOW_UNIT})": st.column_config.NumberColumn(format="%.5g")},
            )

# -------------------- Tab: Sensitivity — CO₂ Utilization (Gas Only) --------------------
with tab_s2:
    st.subheader("Sensitivity: CO₂ Utilization (Gas flows & composition only)")

    col1, col2, col3 = st.columns(3)
    with col1:
        area_u = st.number_input("Area per unit (cm²)", min_value=0.0, value=100.0, step=5.0, key="u_area")
        j_u    = st.number_input("Current density (mA/cm²)", min_value=0.0, value=200.0, step=10.0, key="u_j")
        V_u    = st.number_input("Cell voltage (V)", min_value=0.0, value=3.2, step=0.1, key="u_V")
    with col2:
        fe_map_u = fe_grid_inputs("u", PRODUCT_LIST, title="FE split (%)", per_row=3)
        units_u = n_units_global if use_stack_global else 1
        st.info(f"Using global stack setting: {units_u} unit(s).")
    with col3:
        st.write(f"**Gas basis:** {mv_L_per_mol:.3f} L/mol")

    Umin = st.number_input("Utilization min (%)", min_value=1.0, value=20.0, step=1.0, key="u_min")
    Umax = st.number_input("Utilization max (%)", min_value=1.0, value=100.0, step=1.0, key="u_max")
    Ustep = st.number_input("Utilization step (%)", min_value=1.0, value=5.0, step=1.0, key="u_step")

    inp_u = ElectrolyzerInputs(
        area_value=area_u, area_unit="cm²", j_value=j_u, j_unit="mA/cm²",
        V_cell=V_u, fe_map_pct=fe_map_u,
        n_units=units_u, molar_vol_L=mv_L_per_mol,
    )
    core_u = compute_core_products(inp_u)

    df_util = build_sensitivity_table_U(core_u, Umin, Umax, Ustep)

    value_vars_flows = [f"CO2 Outlet ({GAS_FLOW_UNIT})"] + [f"{p} ({GAS_FLOW_UNIT})" for p in GASES]
    df_flows = df_util.melt(
        id_vars=["Utilization (%)"],
        value_vars=value_vars_flows,
        var_name="Stream",
        value_name=GAS_FLOW_UNIT,
    )
    chart_flows = alt.Chart(df_flows).mark_line(point=True).encode(
        x=alt.X("Utilization (%):Q"),
        y=alt.Y(f"{GAS_FLOW_UNIT}:Q", title=f"Gas flow ({GAS_FLOW_UNIT})"),
        color="Stream:N",
        tooltip=["Utilization (%)", "Stream", GAS_FLOW_UNIT],
    ).properties(title=f"Outlet gas flows vs Utilization (%) [{GAS_FLOW_UNIT}]", height=320)

    df_sens_S = build_sensitivity_table_S(core_u, S_min=1.0, S_max=max(1.0, 1.0/(Umin/100.0)), S_step=0.5)
    value_vars_comp = ["CO2 vol%"] + [f"{p} vol%" for p in GASES]
    df_comp = df_sens_S.melt(id_vars=["Stoich S (inlet/min)"], value_vars=value_vars_comp, var_name="Species", value_name="vol%")
    chart_comp = alt.Chart(df_comp).mark_line(point=True).encode(
        x=alt.X("Stoich S (inlet/min):Q"),
        y=alt.Y("vol%:Q"),
        color="Species:N",
        tooltip=["Stoich S (inlet/min)","Species","vol%"]
    ).properties(title="Gas composition vs Stoich S", height=320)

    left, right = st.columns(2)
    with left:  st.altair_chart(chart_flows, use_container_width=True)
    with right: st.altair_chart(chart_comp,  use_container_width=True)

    st.download_button(
        "Download utilization sweep (CSV)",
        data=df_util.to_csv(index=False).encode("utf-8"),
        file_name="utilization_sweep_gas.csv",
        mime="text/csv",
        key="u_dl"
    )

# -------------------- Tab: Sensitivity — Area × Stack / CO₂ Cap --------------------
with tab_s3:
    st.subheader(f"Sensitivity: Area × Stack (gas {GAS_FLOW_UNIT} + liquid {LIQUID_FLOW_UNIT}) + CO₂ Cap")

    col1, col2, col3 = st.columns(3)
    with col1:
        j_value1 = st.number_input("Current density", min_value=0.0, value=200.0, step=10.0, key="axs_j")
        j_unit1  = st.selectbox("j units", ["mA/cm²","A/cm²","A/m²"], index=0, key="axs_j_unit")
        V_cell1  = st.number_input("Cell voltage (V)", min_value=0.0, value=3.2, step=0.1, key="axs_V")
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

            gas_slpm_map = {}
            liq_rate_map = {}
            co2_min_mol_s = 0.0

            for p in PRODUCT_LIST:
                fe_frac = fe_to_frac(fe_map1.get(p, 0.0))
                n_e = PRODUCT_MAP[p]["nₑ⁻ to product"]
                n_p = prod_mol_s(I_total, fe_frac, n_e)

                if PRODUCT_MAP[p]["Phase"].lower() == "gas":
                    gas_slpm_map[p] = mol_s_to_slpm(n_p, mv_L_per_mol)
                else:
                    kg_h, L_h = mflow_to_mass_and_vol(
                        n_p, PRODUCT_MAP[p]["MW (g/mol)"], PRODUCT_MAP[p]["ρ_liq (kg/L)"]
                    )
                    liq_rate_map[p] = liquid_rate_to_display(kg_h, L_h if L_h is not None else 0.0)

                co2_min_mol_s += n_p * PRODUCT_MAP[p]["co2_per_mol"]

            co2_min_slpm = mol_s_to_slpm(co2_min_mol_s, mv_L_per_mol)
            co2_in_slpm = S1 * co2_min_slpm
            P_total_kW = (I_total * V_cell1) / 1000.0

            row = {
                "Area_cm2": area_cm2,
                "Units": int(n_units1),
                f"CO2_min_{GAS_FLOW_UNIT}": slpm_to_display(co2_min_slpm),
                f"CO2_in_{GAS_FLOW_UNIT}": slpm_to_display(co2_in_slpm),
                "Power_kW": P_total_kW,
            }
            for p in GASES:
                row[f"{p}_{GAS_FLOW_UNIT}"] = slpm_to_display(gas_slpm_map.get(p, 0.0))
            for p in LIQUIDS:
                row[f"{p}_liquid_rate"] = liq_rate_map.get(p, 0.0)
            rows.append(row)

    df_grid = pd.DataFrame(rows)

    co2_min_grid_col = f"CO2_min_{GAS_FLOW_UNIT}"
    co2_in_grid_col = f"CO2_in_{GAS_FLOW_UNIT}"
    metric_choices = (
        [f"{p}_{GAS_FLOW_UNIT}" for p in GASES]
        + [f"{p}_liquid_rate" for p in LIQUIDS]
        + [co2_in_grid_col, co2_min_grid_col, "Power_kW"]
    )
    metric_labels = {f"{p}_{GAS_FLOW_UNIT}": f"{p} ({GAS_FLOW_UNIT})" for p in GASES}
    metric_labels.update({f"{p}_liquid_rate": f"{p} ({LIQUID_FLOW_UNIT})" for p in LIQUIDS})
    metric_labels.update({
        co2_in_grid_col: f"CO₂ inlet ({GAS_FLOW_UNIT})",
        co2_min_grid_col: f"CO₂ minimum ({GAS_FLOW_UNIT})",
        "Power_kW": "Power (kW)",
    })
    if st.session_state.get("axs_metric") not in metric_choices:
        st.session_state["axs_metric"] = metric_choices[0]
    metric = st.selectbox(
        "Heatmap metric", metric_choices, index=0, key="axs_metric",
        format_func=lambda key: metric_labels.get(key, key),
    )
    heat = alt.Chart(df_grid).mark_rect().encode(
        x=alt.X("Area_cm2:O", title="Area per unit (cm²)"),
        y=alt.Y("Units:O", title="# of Units"),
        color=alt.Color(f"{metric}:Q", title=metric_labels.get(metric, metric)),
        tooltip=["Area_cm2","Units"] + metric_choices
    ).properties(height=420)
    st.altair_chart(heat, use_container_width=True)

    st.download_button(
        "Download Area×Stack grid (CSV)",
        data=df_grid.to_csv(index=False).encode("utf-8"),
        file_name="area_stack_grid.csv",
        mime="text/csv",
        key="axs_dl"
    )

    st.markdown("---")
    st.subheader("CO₂ Supply Cap (quick check)")
    co2_cap = gas_flow_number_input(
        "CO₂ supply cap",
        default_slpm=50.0,
        step_slpm=1.0,
        key="cap_cap",
    )

    if not df_grid.empty:
        row0 = df_grid.iloc[0]
        co2_min_display2 = float(row0[co2_min_grid_col])
    else:
        co2_min_display2 = 0.0

    S_min_cap = (co2_cap / max(co2_min_display2, EPS)) if co2_min_display2 > 0 else np.inf
    util_max_cap = min(1.0, 1.0 / max(S_min_cap, EPS))

    c1, c2, c3 = st.columns(3)
    with c1: st.metric(f"CO₂ Minimum ({GAS_FLOW_UNIT})", f"{co2_min_display2:,.3f}")
    with c2: st.metric(f"CO₂ Cap ({GAS_FLOW_UNIT})", f"{co2_cap:,.3f}")
    with c3: st.metric("Max Utilization allowed", f"{100*util_max_cap:.1f}%")

    if np.isinf(S_min_cap) or co2_cap < co2_min_display2:
        st.warning("Cap is below the theoretical minimum CO₂ required at these operating conditions. Reduce current (j), area, units, or adjust FE split.")
    else:
        st.success("Feasible. You may increase utilization up to the shown maximum by reducing S accordingly.")

# -------------------- Tab: Durability, degradation, and stack replacement --------------------
with tab_durability:
    st.subheader("Durability, Degradation & Stack Replacement")
    st.caption("Translate measured degradation rates into replacement intervals, lifetime production, and lifetime-average energy demand.")

    with st.expander("How to use this model", expanded=True):
        st.markdown("""
        1. Choose the product whose FE will be tracked.
        2. Enter beginning-of-life voltage, FE, and carbon efficiency.
        3. Enter linear degradation rates per 1,000 operating hours.
        4. Define replacement thresholds. The earliest threshold becomes the predicted stack life.
        5. Set plant life and availability to estimate replacements and cumulative production.

        This is a **screening model**. It assumes constant current density and linear degradation within each stack cycle, followed by full performance reset after replacement.
        """)

    st.markdown("### 1. Beginning-of-life operating point")
    if "dur_product" not in st.session_state:
        st.session_state["dur_product"] = PRODUCT_LIST[0]
    if "dur_FE0" not in st.session_state:
        st.session_state["dur_FE0"] = float(st.session_state[_global_fe_key(st.session_state["dur_product"])])

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        dur_product = st.selectbox(
            "Tracked product", PRODUCT_LIST, index=0, key="dur_product",
            on_change=load_durability_fe_for_product,
        )
        dur_area = st.number_input("Active area per unit (cm²)", min_value=0.0, value=100.0, step=5.0, key="dur_area")
    with d2:
        dur_j = st.number_input("Current density (mA/cm²)", min_value=0.0, value=200.0, step=10.0, key="dur_j")
        dur_units = n_units_global if use_stack_global else 1
        st.info(f"Using **{dur_units} unit(s)** from Global Settings")
    with d3:
        dur_V0 = st.number_input("Initial cell voltage (V)", min_value=0.0, value=3.0, step=0.05, key="dur_V0")
        dur_FE0 = st.number_input(
            "Initial product FE (%)", min_value=0.0, max_value=100.0, step=1.0, key="dur_FE0",
            on_change=sync_durability_fe_to_system,
            help="Shared with the selected product FE in every other analysis tab.",
        )
    with d4:
        dur_CE0 = st.number_input(
            "Initial fresh-feed carbon efficiency (%)",
            min_value=0.0, max_value=100.0,
            value=80.0 if PRODUCT_MAP[dur_product]["co2_per_mol"] > 0 else 100.0,
            step=1.0,
            key="dur_CE0",
            disabled=PRODUCT_MAP[dur_product]["co2_per_mol"] <= 0,
        )
        st.caption("Carbon efficiency is not applied to H₂ because H₂ contains no carbon.")

    st.markdown("### 2. Degradation rates and replacement limits")
    rate1, rate2, rate3, limit1 = st.columns(4)
    with rate1:
        voltage_rise_mV_1000h = st.number_input("Voltage rise (mV / 1,000 h)", min_value=0.0, value=50.0, step=5.0, key="dur_dV")
        max_voltage = st.number_input("Maximum cell voltage (V)", min_value=0.0, value=3.5, step=0.05, key="dur_Vmax")
    with rate2:
        fe_loss_pp_1000h = st.number_input("FE loss (percentage points / 1,000 h)", min_value=0.0, value=2.0, step=0.5, key="dur_dFE")
        min_FE = st.number_input("Minimum acceptable FE (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0, key="dur_FEmin")
    with rate3:
        ce_loss_pp_1000h = st.number_input(
            "Carbon-efficiency loss (points / 1,000 h)", min_value=0.0, value=2.0, step=0.5, key="dur_dCE",
            disabled=PRODUCT_MAP[dur_product]["co2_per_mol"] <= 0,
        )
        min_CE = st.number_input(
            "Minimum carbon efficiency (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0, key="dur_CEmin",
            disabled=PRODUCT_MAP[dur_product]["co2_per_mol"] <= 0,
        )
    with limit1:
        scheduled_stack_hours = st.number_input("Scheduled maximum stack hours", min_value=1.0, value=10000.0, step=500.0, key="dur_sched_hours")
        replacement_downtime_h = st.number_input("Downtime per replacement (h)", min_value=0.0, value=24.0, step=4.0, key="dur_downtime")

    st.markdown("### 3. Deployment horizon")
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        plant_life_years = st.number_input("Plant life (years)", min_value=0.1, value=10.0, step=1.0, key="dur_years")
    with h2:
        base_capacity_factor_pct = st.number_input("Base capacity factor (%)", min_value=0.0, max_value=100.0, value=90.0, step=1.0, key="dur_cf")
    with h3:
        replacement_stack_cost = st.number_input("Replacement stack cost ($, optional)", min_value=0.0, value=0.0, step=1000.0, key="dur_repl_cost")
    with h4:
        st.caption("Base capacity factor excludes the additional replacement downtime entered above.")

    voltage_rate_V_h = voltage_rise_mV_1000h / 1e6
    fe_rate_pp_h = fe_loss_pp_1000h / 1000.0
    ce_rate_pp_h = ce_loss_pp_1000h / 1000.0

    t_voltage = (max_voltage - dur_V0) / voltage_rate_V_h if voltage_rate_V_h > EPS and max_voltage > dur_V0 else np.inf
    t_fe = (dur_FE0 - min_FE) / fe_rate_pp_h if fe_rate_pp_h > EPS and dur_FE0 > min_FE else np.inf
    if PRODUCT_MAP[dur_product]["co2_per_mol"] > 0:
        t_ce = (dur_CE0 - min_CE) / ce_rate_pp_h if ce_rate_pp_h > EPS and dur_CE0 > min_CE else np.inf
    else:
        t_ce = np.inf

    trigger_name, cycle_life_h = first_positive_minimum({
        "Voltage limit": t_voltage,
        "FE limit": t_fe,
        "Carbon-efficiency limit": t_ce,
        "Scheduled stack-hour limit": scheduled_stack_hours,
    })

    if not np.isfinite(cycle_life_h) or cycle_life_h <= EPS:
        st.error("No valid positive stack-life limit was found. Check degradation rates and thresholds.")
    elif max_voltage <= dur_V0 or min_FE >= dur_FE0 or (PRODUCT_MAP[dur_product]["co2_per_mol"] > 0 and min_CE >= dur_CE0):
        st.error("At least one replacement threshold is already reached or exceeded at beginning of life.")
    else:
        total_calendar_h = plant_life_years * 8760.0
        planned_operating_h = total_calendar_h * base_capacity_factor_pct / 100.0

        # Treat the capacity-factor-adjusted time as the available deployment window.
        # Each completed replacement block contains one stack life plus its downtime;
        # the final stack does not incur end-of-project replacement downtime.
        if replacement_downtime_h > EPS:
            completed_replacement_blocks = int(np.floor(planned_operating_h / (cycle_life_h + replacement_downtime_h)))
            remaining_window_h = planned_operating_h - completed_replacement_blocks * (cycle_life_h + replacement_downtime_h)
            actual_operating_h = completed_replacement_blocks * cycle_life_h + min(cycle_life_h, max(0.0, remaining_window_h))
            replacements = completed_replacement_blocks
        else:
            actual_operating_h = planned_operating_h
            replacements = max(0, int(np.ceil(actual_operating_h / cycle_life_h - 1e-12)) - 1)

        area_m2_dur = dur_area * 1e-4
        j_A_m2_dur = dur_j * 10.0
        I_total_dur = area_m2_dur * j_A_m2_dur * dur_units
        props_dur = PRODUCT_MAP[dur_product]
        n_e_dur = props_dur["nₑ⁻ to product"]
        MW_kg_mol_dur = props_dur["MW (g/mol)"] / 1000.0
        carbon_per_product = props_dur["co2_per_mol"]

        # Lifetime totals are evaluated from a representative degradation cycle and
        # scaled analytically, so very short stack lives cannot create enormous tables.
        mass_rate_per_FE_point = I_total_dur * MW_kg_mol_dur * 3600.0 / (n_e_dur * F * 100.0)

        def product_integral_kg(duration_h: float) -> float:
            d = max(0.0, min(float(duration_h), cycle_life_h))
            return mass_rate_per_FE_point * max(0.0, dur_FE0 * d - 0.5 * fe_rate_pp_h * d * d)

        def energy_integral_kWh(duration_h: float) -> float:
            d = max(0.0, min(float(duration_h), cycle_life_h))
            return I_total_dur / 1000.0 * (dur_V0 * d + 0.5 * voltage_rate_V_h * d * d)

        def fresh_co2_integral_kg(duration_h: float) -> float:
            if carbon_per_product <= 0:
                return 0.0
            d = max(0.0, min(float(duration_h), cycle_life_h))
            if d <= EPS:
                return 0.0
            age = np.linspace(0.0, d, 300)
            fe_pct_local = np.maximum(0.0, dur_FE0 - fe_rate_pp_h * age)
            ce_pct_local = np.maximum(EPS, dur_CE0 - ce_rate_pp_h * age)
            n_product_local = I_total_dur * (fe_pct_local / 100.0) / (n_e_dur * F)
            fresh_co2_kg_h_local = (
                n_product_local * carbon_per_product / np.maximum(ce_pct_local / 100.0, EPS)
                * MW_CO2_G_MOL * 3600.0 / 1000.0
            )
            return trapezoid_integral(fresh_co2_kg_h_local, age)

        full_cycles = int(np.floor(actual_operating_h / cycle_life_h + 1e-12))
        partial_cycle_h = max(0.0, actual_operating_h - full_cycles * cycle_life_h)
        if partial_cycle_h < 1e-8:
            partial_cycle_h = 0.0

        full_cycle_product_kg = product_integral_kg(cycle_life_h)
        full_cycle_energy_kWh = energy_integral_kWh(cycle_life_h)
        full_cycle_fresh_co2_kg = fresh_co2_integral_kg(cycle_life_h)

        total_product_kg = full_cycles * full_cycle_product_kg + product_integral_kg(partial_cycle_h)
        total_energy_kWh = full_cycles * full_cycle_energy_kWh + energy_integral_kWh(partial_cycle_h)
        total_fresh_co2_kg = full_cycles * full_cycle_fresh_co2_kg + fresh_co2_integral_kg(partial_cycle_h)

        initial_n_product_mol_s = I_total_dur * (dur_FE0 / 100.0) / (n_e_dur * F)
        initial_mass_rate_kg_h = initial_n_product_mol_s * MW_kg_mol_dur * 3600.0
        ideal_product_kg = initial_mass_rate_kg_h * actual_operating_h
        production_loss_pct = 100.0 * (1.0 - total_product_kg / ideal_product_kg) if ideal_product_kg > EPS else 0.0
        lifetime_sec = total_energy_kWh / total_product_kg if total_product_kg > EPS else np.nan
        kg_co2_per_kg = total_fresh_co2_kg / total_product_kg if total_product_kg > EPS and carbon_per_product > 0 else np.nan
        total_replacement_cost = replacements * replacement_stack_cost

        # Keep the browser responsive: plot at most the first 20 stack cycles.
        total_cycles_operated = full_cycles + (1 if partial_cycle_h > EPS else 0)
        cycles_to_plot = min(total_cycles_operated, 20)
        profile_truncated = total_cycles_operated > cycles_to_plot
        segments = []
        cycle_starts = []
        for cycle_zero_based in range(cycles_to_plot):
            cycle_start = cycle_zero_based * cycle_life_h
            if cycle_zero_based < full_cycles:
                duration = cycle_life_h
            else:
                duration = partial_cycle_h
            if duration <= EPS:
                continue
            ages = np.linspace(0.0, duration, 80)
            cumulative_h = cycle_start + ages
            voltage = dur_V0 + voltage_rate_V_h * ages
            fe_pct = np.maximum(0.0, dur_FE0 - fe_rate_pp_h * ages)
            if carbon_per_product > 0:
                ce_pct = np.maximum(EPS, dur_CE0 - ce_rate_pp_h * ages)
            else:
                ce_pct = np.full_like(ages, np.nan)

            fe_frac = fe_pct / 100.0
            n_product_mol_s = I_total_dur * fe_frac / (n_e_dur * F)
            mass_rate_kg_h = n_product_mol_s * MW_kg_mol_dur * 3600.0
            power_kW = I_total_dur * voltage / 1000.0
            if carbon_per_product > 0:
                product_carbon_mol_s = n_product_mol_s * carbon_per_product
                fresh_co2_mol_s = product_carbon_mol_s / np.maximum(ce_pct / 100.0, EPS)
                fresh_co2_kg_h = fresh_co2_mol_s * MW_CO2_G_MOL * 3600.0 / 1000.0
            else:
                fresh_co2_kg_h = np.zeros_like(ages)

            cumulative_product_kg = cycle_zero_based * full_cycle_product_kg + (
                mass_rate_per_FE_point * np.maximum(0.0, dur_FE0 * ages - 0.5 * fe_rate_pp_h * ages * ages)
            )
            segments.append(pd.DataFrame({
                "Operating hour": cumulative_h,
                "Cycle": cycle_zero_based + 1,
                "Cycle age (h)": ages,
                "Cell voltage (V)": voltage,
                "Product FE (%)": fe_pct,
                "Carbon efficiency (%)": ce_pct,
                "Product rate (kg/h)": mass_rate_kg_h,
                "Power (kW)": power_kW,
                "Fresh CO2 rate (kg/h)": fresh_co2_kg_h,
                "Cumulative product (kg)": cumulative_product_kg,
            }))
            cycle_starts.append(cycle_start)

        dur_df = pd.concat(segments, ignore_index=True) if segments else pd.DataFrame()

        r1, r2, r3, r4 = st.columns(4)
        with r1: st.metric("Predicted stack life", f"{cycle_life_h:,.0f} h")
        with r2: st.metric("Limiting trigger", trigger_name)
        with r3: st.metric("Stack replacements", f"{replacements:,d}")
        with r4: st.metric("Actual operating hours", f"{actual_operating_h:,.0f} h")

        r5, r6, r7, r8 = st.columns(4)
        with r5: st.metric(f"Lifetime {dur_product}", f"{total_product_kg:,.1f} kg")
        with r6: st.metric("Lifetime-average electricity", "—" if np.isnan(lifetime_sec) else f"{lifetime_sec:,.2f} kWh/kg")
        with r7: st.metric("Production lost to FE decay", f"{production_loss_pct:.1f}%")
        with r8: st.metric("Replacement-stack cost", f"${total_replacement_cost:,.0f}")

        if carbon_per_product > 0:
            cc1, cc2 = st.columns(2)
            with cc1: st.metric("Lifetime fresh CO₂", f"{total_fresh_co2_kg:,.1f} kg")
            with cc2: st.metric("Fresh CO₂ / product", "—" if np.isnan(kg_co2_per_kg) else f"{kg_co2_per_kg:,.3f} kg/kg")

        replacement_df = pd.DataFrame({"Operating hour": cycle_starts[1:]})
        voltage_line = alt.Chart(dur_df).mark_line().encode(
            x=alt.X("Operating hour:Q", title="Cumulative operating hours"),
            y=alt.Y("Cell voltage (V):Q", title="Cell voltage (V)"),
            tooltip=[alt.Tooltip("Operating hour:Q", format=",.0f"), "Cycle:Q", alt.Tooltip("Cell voltage (V):Q", format=".3f")],
        ).properties(height=330, title="Voltage degradation and reset after replacement")
        if not replacement_df.empty:
            voltage_line = voltage_line + alt.Chart(replacement_df).mark_rule(strokeDash=[5, 5]).encode(x="Operating hour:Q")

        perf_long_vars = ["Product FE (%)"] + (["Carbon efficiency (%)"] if carbon_per_product > 0 else [])
        perf_long = dur_df.melt(
            id_vars=["Operating hour", "Cycle"],
            value_vars=perf_long_vars,
            var_name="Performance metric",
            value_name="Percent",
        )
        performance_line = alt.Chart(perf_long).mark_line().encode(
            x=alt.X("Operating hour:Q", title="Cumulative operating hours"),
            y=alt.Y("Percent:Q", title="Performance (%)", scale=alt.Scale(domain=[0, 100])),
            color="Performance metric:N",
            tooltip=[alt.Tooltip("Operating hour:Q", format=",.0f"), "Cycle:Q", "Performance metric:N", alt.Tooltip("Percent:Q", format=".1f")],
        ).properties(height=330, title="FE and carbon-efficiency degradation")

        pc1, pc2 = st.columns(2)
        with pc1: st.altair_chart(voltage_line, use_container_width=True)
        with pc2: st.altair_chart(performance_line, use_container_width=True)

        if profile_truncated:
            st.caption(
                f"Plots show the first {cycles_to_plot} of {total_cycles_operated:,} operating cycles to keep the dashboard responsive. Lifetime metrics above include the full deployment horizon."
            )

        with st.expander("Cumulative production and downloadable displayed profile", expanded=False):
            dur_df_download = dur_df.copy()
            cumulative_chart = alt.Chart(dur_df_download).mark_line().encode(
                x=alt.X("Operating hour:Q", title="Cumulative operating hours"),
                y=alt.Y("Cumulative product (kg):Q", title=f"Cumulative {dur_product} (kg)"),
                tooltip=[alt.Tooltip("Operating hour:Q", format=",.0f"), alt.Tooltip("Cumulative product (kg):Q", format=",.1f")],
            ).properties(height=330)
            st.altair_chart(cumulative_chart, use_container_width=True)
            st.dataframe(dur_df_download, hide_index=True, use_container_width=True)
            st.download_button(
                "Download displayed durability profile (CSV)",
                data=dur_df_download.to_csv(index=False).encode("utf-8"),
                file_name="cheese_durability_profile.csv",
                mime="text/csv",
                key="dur_download",
            )

        st.info("Interpretation: the replacement interval is controlled by the first threshold reached. Tightening any threshold can increase replacements and downtime even when beginning-of-life performance is unchanged.")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("© 2025 Aditya Prajapati · CHEESE")
