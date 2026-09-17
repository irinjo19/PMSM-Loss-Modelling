# PMSM Loss Modelling — Redesigned & Benchmarked

This repository contains code and models for evaluating **Permanent Magnet Synchronous Motor (PMSM)** loss estimation across operating conditions. It benchmarks physical material-derived models, reduced-order regression models, purely data-driven machine learning models (Gradient Boosting, MLPs), and hybrid physics-guided neural networks (PGNN).

The central research question addressed by this codebase is:
> **When does physics actually help data-driven PMSM loss estimation?**

---

## Features & Methodologies

- **Physics Baselines**:
  - Material-derived iron loss ($P_{Fe} = K_h f B^\alpha + K_e f^2 B^2$) fitted from stator steel measurements.
  - Stator copper loss ($P_{Cu} = 1.5 R_s I_q^2$) derived from equivalent circuit electrical parameters.
  - Parasitic/unmodelled loss speed-dependent correction ($P_{\text{parasitic}} = c_1 |\omega| + c_2 \omega^2$).
- **Data-Driven & Empirical Baselines**:
  - Reduced-Order Model (ROM): Interpretable polynomial ($a_0 + a_1 |\omega| + a_2 \omega^2 + a_3 T^2 + a_4 |\omega T|$).
  - Polynomial Ridge Regression.
  - Histogram-based Gradient Boosting (`HistGradientBoostingRegressor`).
  - Pure Multi-Layer Perceptron (`Data_MLP`).
- **Hybrid & Physics-Guided Neural Networks**:
  - Residual MLP (`Residual_MLP`): Trains an MLP to learn the residual $P_{\text{measured}} - P_{\text{physics}}$.
  - Physics-Guided Neural Network (`Physics_guided_NN`): PyTorch model trained with joint loss $L = L_{\text{data}} + \lambda L_{\text{physics}}$.
- **Comprehensive Evaluation**:
  - **Random vs. Blocked Splits**: Evaluates standard random train/test splits vs. contiguous time-block splits.
  - **Leave-One-Cycle-Out Cross-Validation**: Holds out complete vehicle drive cycles to test generalization.
  - **Operating-Region Extrapolation**: Evaluates models under high-speed and high-torque extrapolation (top 25%).
  - **Low-Data Efficiency Studies**: Benchmarks data efficiency across operating-map-aware sampling budgets.
  - **Winding Temperature Sensitivity**: Analyzes sensitivity to stator copper resistance temperature variations.
  - **Cycle-Integrated Energy Error**: Evaluates cumulative energy estimation error ($Wh$) across drive cycles.

---

## Requirements & Environment Setup

### Prerequisites
- Python 3.8+
- Recommended packages:
  - `numpy`
  - `pandas`
  - `scipy`
  - `scikit-learn`
  - `torch`
  - `matplotlib`
  - `openpyxl`

### Quick Start using `uv` (Recommended)

1. **Create virtual environment and install dependencies**:
   ```bash
   uv venv
   uv pip install --python .venv\Scripts\python.exe pandas numpy scipy scikit-learn torch matplotlib openpyxl
   ```

2. **Run the loss modelling script**:
   ```bash
   .\.venv\Scripts\python.exe -u pmsm_loss_modelling_redesigned_v2.py
   ```

### Standard `pip` Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python pmsm_loss_modelling_redesigned_v2.py
```

---

## Dataset Location & Structure

The script automatically detects local datasets or downloads them from the CREATOR benchmark repository (TU Graz):
- Dataset path: `PM_synchronous_motor/PM_synchronous_motor`
- Subdirectories expected:
  - `Design_parameters/` (Electrical, Motor geometry, Material properties, Winding properties)
  - `Measurement_results/` (Equivalent circuit, No-load tests, Drive cycle speed/torque & measurement data)

---

## Summary of Experiments

| Experiment | Description | Key Findings |
|---|---|---|
| **Exp 1: Random vs. Blocked Split** | In-distribution generalization across standard random and contiguous block splits. | GBT & PGNN deliver high accuracy, with ROM providing a strong interpretable baseline. |
| **Exp 2: Leave-One-Cycle-Out** | Group-level generalization by holding out entire drive cycle runs. | Tests whether physics aids out-of-cycle transfer. |
| **Exp 3: Extrapolation** | Training on lower 75% speed/torque and testing on top 25%. | Physics-guided models prevent unphysical divergence outside training range. |
| **Exp 4: Low-Data Efficiency** | Performance under restricted budgets (20 to 5000 operating points). | Physics-guided and residual models dominate in low-data regimes. |
| **Exp 5: Residual Diagnostics** | Analysis of $P_{\text{measured}} - P_{\text{physics}}$ against operating variables. | Reveals unmodelled speed and torque dynamics. |
| **Exp 6: Temperature Sensitivity** | Impact of temperature variations ($20^\circ C - 120^\circ C$) on copper loss. | Quantifies error bounds when winding temperature is unmeasured. |
| **Exp 7: Cycle-Integrated Energy** | Evaluation of integrated cycle energy error ($Wh$). | Confirms energy-conservation properties of physical models over long cycles. |

---

## Citation & License

This project is licensed under the [MIT License](LICENSE).
DataSet courtesy of the CREATOR Project (TU Graz).
