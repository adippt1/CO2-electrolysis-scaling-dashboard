# 🧀 CHEESE — CO₂ Handling & Electrolysis Efficiency Scaling Evaluator

**Tagline:** Because scaling electrolysis shouldn’t be this gouda! 🧀

[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9–3.12-blue.svg)](https://www.python.org/)


CHEESE is an interactive Streamlit dashboard (or Cheeseboard) for CO₂ handling and electrolysis scaling. It provides calculators and several sensitivity analyses to translate lab‑scale performance to stack‑level requirements: gas handling, stoichiometry/utilization, active area sizing, and power. It is simple..but it's got heart.

**Made by:** [Aditya Prajapati (Adi)](https://people.llnl.gov/prajapati3) + ChatGPT (GPT‑5 Thinking): LLM was really helpful in making the codes looks pretty and modular. Also, it takes care of subscripting the molecular formulas which is nice.

---

## Features
- **Calculator:** Current, power, product rates (CO, C₂H₄), theoretical CO₂ minimum, energy‑efficiency metric.
- **Sizing from Inlet & Stoich:** Determine **total and per‑unit active area** to process a given CO₂ feed at a selected stoichiometric ratio **S**.
- **Sensitivity — CO₂ Utilization:** Sweep U (%) and visualize outlet flows and composition vs. S.
- **Sensitivity — Area × Stack:** Heatmap for production/power vs. per‑unit area and number of units.
- **Sensitivity — CO₂ Supply Cap:** Feasibility under a supply constraint with recommendations.
- **Single source of truth:** **Global settings** for STP/SATP gas basis and stack size shared across all tabs.

---

## Installation

```bash
# 1) Create a fresh environment (recommended)
python -m venv .venv && source .venv/bin/activate # Windows: .venv\\Scripts\\activate

# 2) Install dependencies
pip install streamlit numpy pandas altair

# 3) (Optional) Pin versions for reproducibility
pip install "streamlit==1.*" "numpy==1.*" "pandas==2.*" "altair==5.*"

## Other comments
The author will not be apologizing for the use of pun(s)