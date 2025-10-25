# CHEESE — CO₂ Handling & Electrolysis Efficiency Scaling Evaluator
# Tagline: Because scaling electrolysis shouldn’t be this gouda! 🧀
# Author: Aditya Prajapati + ChatGPT (GPT-5 Thinking)
# Copyright (c) 2025 Aditya Prajapati

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# -------------------- Page setup --------------------
st.set_page_config(
    page_title="CHEESE — CO₂ Handling & Electrolysis",
    page_icon="🧀",
    layout="wide",
)

st.title("🧀 CHEESE — CO₂ Handling & Electrolysis Efficiency Scaling Evaluator")
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
    help="Used to compute gas molar flows from SLPM and gas densities from MW.",
)

mv_L_per_mol = MV_OPTIONS[basis_label]  # L/mol
mv_m3_per_mol = mv_L_per_mol / 1000.0  # m³/mol

use_stack_global = st.sidebar.checkbox("Use a stack (multiple identical units)?", value=True, key="gs_stack")
n_units_global = st.sidebar.number_input("Number of units in stack", min_value=1, value=10, step=1, key="gs_units")

# -------------------- Helper: numeric sanitizer (Arrow-safe) --------------------
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

# -------------------- Product properties (single source of truth) --------------------

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

def mflow_to_mass_and_vol(n_mol_s: float, MW_g_mol: float, rho_liq_kg_L: Optional[float]) -> Tuple[float, Optional[float]]:
    kg_h = n_mol_s * MW_g_mol * 3600.0 / 1000.0
    if rho_liq_kg_L is None or rho_liq_kg_L <= 0:
        return kg_h, None
    L_h = kg_h / rho_liq_kg_L
    return kg_h, L_h

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
    st.markdown(f"#### {title}")
    fe_map: Dict[str, float] = {}
    defaults = {p: (90.0 if p == "CO" else 0.0) for p in products}
    if default_map:
        defaults.update({k: float(default_map.get(k, defaults[k])) for k in products})
    for i in range(0, len(products), per_row):
        row = products[i:i+per_row]
        cols = st.columns(len(row), gap="small")
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
    gas_prod_total = sum(gas_prod_slpm.values())

    S_vals = np.arange(S_min, S_max + 1e-9, S_step)
    rows = []
    for S in S_vals:
        S = max(1.0, float(S))
        util = 1.0 / S
        co2_in = S * co2_min_slpm
        co2_out = co2_in - co2_min_slpm
        gas_total_out = max(1e-9, co2_out + gas_prod_total)

        row = {
            "Stoich S (inlet/min)": S,
            "CO2 Utilization (frac)": util,
            "CO2 Inlet (SLPM)": co2_in,
            "CO2 Outlet (SLPM)": co2_out,
            "Gas Total Outlet (SLPM)": gas_total_out,
            "CO2 vol%": 100 * co2_out / gas_total_out,
        }
        for p in GASES:
            row[f"{p} (SLPM)"] = gas_prod_slpm[p]
            row[f"{p} vol%"] = 100 * gas_prod_slpm[p] / gas_total_out
        rows.append(row)
    return pd.DataFrame(rows)

def build_sensitivity_table_U(core: Dict[str, float], Umin_pct: float, Umax_pct: float, Ustep_pct: float) -> pd.DataFrame:
    co2_min_slpm = core["CO2_min_slpm"]
    gas_prod_slpm = {p: core.get(f"{p}_slpm", 0.0) for p in GASES}
    gas_prod_total = sum(gas_prod_slpm.values())

    U_vals_pct = np.arange(Umin_pct, Umax_pct + 1e-9, Ustep_pct)
    rows = []
    for U_pct in U_vals_pct:
        U = max(1e-6, min(1.0, U_pct / 100.0))
        S = 1.0 / U
        co2_in = S * co2_min_slpm
        co2_out = co2_in - co2_min_slpm
        gas_total_out = max(1e-9, co2_out + gas_prod_total)

        row = {
            "Utilization (%)": U_pct,
            "Stoich S (inlet/min)": S,
            "CO2 Inlet (SLPM)": co2_in,
            "CO2 Outlet (SLPM)": co2_out,
            "Gas Total Outlet (SLPM)": gas_total_out,
            "CO2 vol%": 100 * co2_out / gas_total_out,
        }
        for p in GASES:
            row[f"{p} (SLPM)"] = gas_prod_slpm[p]
            row[f"{p} vol%"] = 100 * gas_prod_slpm[p] / gas_total_out
        rows.append(row)
    return pd.DataFrame(rows)

# -------------------- Tabs --------------------
tab_instructions, tab_calc, tab_size, tab_s2, tab_s3 = st.tabs([
    "Instructions",
    "Calculator",
    "Calc: Area from Inlet & Stoich",
    "Sensitivity: CO₂ Utilization",
    "Sensitivity: Area × Stack / CO₂ Cap",
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
            
            - Stoich is the "Stoichiometry". It is the ratio of actual CO₂ fed to the theoretical minimum CO₂ required to produce the observed products.
               
                • S = 1 means 100% CO₂ utilization (no excess feed).
               
                • S > 1 means excess CO₂ feed and lower utilization (e.g., S = 2 → 50% utilization).
         
        
        - **Calc: Area from Inlet & Stoich:**  
          Provides the **required electrode area** per unit and total area for a given CO₂ inlet and stoichiometric ratio (S).  
          Includes per-product outputs (gas SLPM, liquid kg/h, etc.).
    
        - **Sensitivity: CO₂ Utilization:**  
          Sweeps utilization (%) to show gas outlet composition and flowrate trends.
    
        - **Sensitivity: Area × Stack:**  
          Visualizes scaling trade-offs between cell area and number of units in the stack using a heatmap.
    
        - **Sensitivity: CO₂ Supply Cap:**  
          Determines the maximum achievable utilization given a CO₂ feed limitation.
    
        - **Constants & Reference (this tab):**  
          Lists all physical constants, product properties, and data sources.
    
        💡**Tips:**  
        - You can download any result table via the “Download CSV” buttons.  
        - Hover over plots for tooltips showing precise data points.  
        - Adjust **molar volume basis (STP/SATP)** in the sidebar to update gas volumetric conversions.
        - If you find any mistakes please feel free to reach out!
    
        ---
        """)

    st.subheader("Constants & Properties")
    st.markdown(f"""
- **Faraday constant (F):** `{F:.5f}` C·mol⁻¹ e⁻  
- **Molar volume bases:**  
  • STP = `{MV_OPTIONS['STP (0°C, 1 atm) — 22.414 L/mol']:.3f}` L·mol⁻¹  
  • SATP = `{MV_OPTIONS['SATP (25°C, 1 atm) — 24.465 L/mol']:.3f}` L·mol⁻¹  
- **Current basis:** `{basis_label}`  
- **Stacking:** `{'ON' if use_stack_global else 'OFF'}` — Units: `{n_units_global}`  
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
    mode = st.radio("CO₂ feed input mode", ["Stoich (S)", "Inlet flow (SLPM)"], index=0, horizontal=True, key="calc_mode")
    if mode == "Stoich (S)":
        S = st.number_input("CO₂ Stoich S (inlet/min)", min_value=1.0, value=2.0, step=0.1, key="calc_S")
        co2_in_slpm_input = None
    else:
        co2_in_slpm_input = st.number_input("CO₂ Inlet flow (SLPM)", min_value=0.0, value=10.0, step=0.5, key="calc_inlet")
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
        co2_in_slpm = 0.0 if (mode == "Stoich (S)") else float(co2_in_slpm_input or 0.0)
        Stoich_S = (np.inf if co2_in_slpm > 0 else 1.0) if mode != "Stoich (S)" else float(S or 1.0)
        util = 0.0 if co2_in_slpm > 0 else 1.0
        warn = "Total FE to carbon products is zero; CO₂ minimum is 0."
    else:
        if mode == "Stoich (S)":
            Stoich_S = max(1.0, float(S or 1.0))
            co2_in_slpm = Stoich_S * core["CO2_min_slpm"]
            util = 1.0 / Stoich_S
            warn = None
        else:
            co2_in_slpm = max(0.0, float(co2_in_slpm_input or 0.0))
            Stoich_S = co2_in_slpm / max(core["CO2_min_slpm"], EPS)
            util = min(1.0, 1.0 / max(Stoich_S, EPS))
            warn = None if co2_in_slpm >= core["CO2_min_slpm"] else f"Inlet ({co2_in_slpm:.3f} SLPM) < minimum ({core['CO2_min_slpm']:.3f} SLPM)."

    # GAS metrics
    st.subheader("Gas-side Results (true outlet)")
    g1, g2, g3, g4 = st.columns(4)
    with g1: st.metric("CO₂ Minimum (SLPM)", f"{core['CO2_min_slpm']:.3f}")
    with g2: st.metric("CO₂ Inlet (SLPM)", f"{co2_in_slpm:.3f}")
    with g3: st.metric("Stoich S (inlet/min)", "∞" if not np.isfinite(Stoich_S) else f"{Stoich_S:.3f}")
    with g4: st.metric("Utilization (%)", f"{util*100:.1f}")

    g5, g6, g7 = st.columns(3)
    with g5: st.metric("Per-Unit Current (A)", f"{core['I_unit_A']:.2f}")
    with g6: st.metric("Total Current (A)", f"{core['I_total_A']:.2f}")
    with g7: st.metric("Power (kW)", f"{(core['I_total_A']*V_cell)/1000.0:.2f}")

    st.markdown("#### Gas product flowrates (SLPM)")
    gas_cols = st.columns(max(1, len(GASES)))
    for i, p in enumerate(GASES):
        with gas_cols[i]:
            st.metric(p, f"{core.get(f'{p}_slpm', 0.0):.3f}")
    gas_total_out = sum(core.get(f"{p}_slpm", 0.0) for p in GASES) + max(co2_in_slpm - core["CO2_min_slpm"], 0.0)
    st.metric("Gas Total Outlet (SLPM)", f"{gas_total_out:.3f}")

    # LIQUID metrics
    st.subheader("Liquid production (true condensed)")
    liq_rows = []
    for p in LIQUIDS:
        n_p = core.get(f"{p}_mol_s", 0.0)
        MW = PRODUCT_MAP[p]["MW (g/mol)"]
        rho = PRODUCT_MAP[p]["ρ_liq (kg/L)"]
        kg_h, L_h = mflow_to_mass_and_vol(n_p, MW, rho)
        liq_rows.append({
            "Product": p,
            "mol/s": n_p,
            "kg/h": kg_h,
            "L/h": (L_h if L_h is not None else 0.0),
            "ρ (kg/L)": (rho if rho else 0.0),
            "MW (g/mol)": MW
        })
    if liq_rows:
        df_liq = pd.DataFrame(liq_rows)
        st.dataframe(df_liq, hide_index=True, use_container_width=True)

    if warn:
        st.warning(warn)

# -------------------- Tab: Calc — Size Active Area from CO₂ Inlet & Stoich --------------------
with tab_size:
    st.subheader("Calc: Size Active Area from CO₂ Inlet & Stoich (with per-product outputs)")

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
            st.metric("CO₂ Minimum (SLPM)", f"{co2_min_slpm_sz:.3f}")
            st.metric("Utilization (%)", f"{util_sz*100:.1f}")

        st.subheader("Resulting Gas Products (SLPM)")
        if gas_rows:
            colsG = st.columns(len(gas_rows))
            for i, (p, slpm_p) in enumerate(gas_rows):
                with colsG[i]: st.metric(p, f"{slpm_p:.3f}")
            st.write(f"**Gas products total (SLPM)**: {gas_total_slpm:.3f}")

        st.subheader("Resulting Liquid Products")
        if liq_rows:
            df_liq_sz = pd.DataFrame([{"Product": p, "mol/s": n, "kg/h": kg, "L/h": Lh} for (p, n, kg, Lh) in liq_rows])
            st.dataframe(df_liq_sz, hide_index=True, use_container_width=True)

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

    value_vars_flows = ["CO2 Outlet (SLPM)"] + [f"{p} (SLPM)" for p in GASES]
    df_flows = df_util.melt(id_vars=["Utilization (%)"], value_vars=value_vars_flows, var_name="Stream", value_name="SLPM")
    chart_flows = alt.Chart(df_flows).mark_line(point=True).encode(
        x=alt.X("Utilization (%):Q"),
        y=alt.Y("SLPM:Q"),
        color="Stream:N",
        tooltip=["Utilization (%)","Stream","SLPM"]
    ).properties(title="Outlet gas flows vs Utilization (%)", height=320)

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
    st.subheader("Sensitivity: Area × Stack (gas SLPM + liquid kg/h) + CO₂ Cap")

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
            liq_kg_h_map = {}
            co2_min_mol_s = 0.0

            for p in PRODUCT_LIST:
                fe_frac = fe_to_frac(fe_map1.get(p, 0.0))
                n_e = PRODUCT_MAP[p]["nₑ⁻ to product"]
                n_p = prod_mol_s(I_total, fe_frac, n_e)

                if PRODUCT_MAP[p]["Phase"].lower() == "gas":
                    gas_slpm_map[p] = mol_s_to_slpm(n_p, mv_L_per_mol)
                else:
                    kg_h, _ = mflow_to_mass_and_vol(n_p, PRODUCT_MAP[p]["MW (g/mol)"], PRODUCT_MAP[p]["ρ_liq (kg/L)"])
                    liq_kg_h_map[p] = kg_h

                co2_min_mol_s += n_p * PRODUCT_MAP[p]["co2_per_mol"]

            co2_min_slpm = mol_s_to_slpm(co2_min_mol_s, mv_L_per_mol)
            co2_in_slpm = S1 * co2_min_slpm
            P_total_kW = (I_total * V_cell1) / 1000.0

            row = {"Area_cm2": area_cm2, "Units": int(n_units1), "CO2_min_SLPM": co2_min_slpm, "CO2_in_SLPM": co2_in_slpm, "Power_kW": P_total_kW}
            for p in GASES:   row[f"{p}_SLPM"] = gas_slpm_map.get(p, 0.0)
            for p in LIQUIDS: row[f"{p}_kg_h"]  = liq_kg_h_map.get(p, 0.0)
            rows.append(row)

    df_grid = pd.DataFrame(rows)

    metric_choices = [f"{p}_SLPM" for p in GASES] + [f"{p}_kg_h" for p in LIQUIDS] + ["CO2_in_SLPM", "CO2_min_SLPM", "Power_kW"]
    metric = st.selectbox("Heatmap metric", metric_choices, index=0, key="axs_metric")
    heat = alt.Chart(df_grid).mark_rect().encode(
        x=alt.X("Area_cm2:O", title="Area per unit (cm²)"),
        y=alt.Y("Units:O", title="# of Units"),
        color=alt.Color(f"{metric}:Q", title=metric),
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
    co2_cap = st.number_input("CO₂ supply cap (SLPM)", min_value=0.0, value=50.0, step=1.0, key="cap_cap")

    if not df_grid.empty:
        row0 = df_grid.iloc[0]
        co2_min_slpm2 = float(row0["CO2_min_SLPM"])
    else:
        co2_min_slpm2 = 0.0

    S_min_cap = (co2_cap / max(co2_min_slpm2, EPS)) if co2_min_slpm2 > 0 else np.inf
    util_max_cap = min(1.0, 1.0 / max(S_min_cap, EPS))

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("CO₂ Minimum (SLPM)", f"{co2_min_slpm2:.3f}")
    with c2: st.metric("CO₂ Cap (SLPM)", f"{co2_cap:.3f}")
    with c3: st.metric("Max Utilization allowed", f"{100*util_max_cap:.1f}%")

    if np.isinf(S_min_cap) or co2_cap < co2_min_slpm2:
        st.warning("Cap is below the theoretical minimum CO₂ required at these operating conditions. Reduce current (j), area, units, or adjust FE split.")
    else:
        st.success("Feasible. You may increase utilization up to the shown maximum by reducing S accordingly.")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("© 2025 Aditya Prajapati · CHEESE")
