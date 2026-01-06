
# Quantum Stability Analysis of the Duffing Oscillator via Spectral Curvature

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)
![JAX](https://img.shields.io/badge/Accelerated_by-JAX-orange?logo=google&logoColor=white)
![Physics](https://img.shields.io/badge/Field-Quantum_Physics-blueviolet)
![License](https://img.shields.io/badge/license-MIT-green)

## 1. Abstract
This repository provides a high-performance computational framework for analyzing the structural stability of a quantum Duffing oscillator. Utilizing **JAX** for accelerated linear algebra and automatic vectorization, the engine evaluates the stability of the system’s ground state by constructing a 3x3 curvature matrix derived from spectral moments. The framework identifies phase boundaries between stable and unstable regimes in the parameter space of harmonic frequency ($\omega$) and quartic non-linearity ($\alpha$), ensuring convergence through high-precision (float64) calculations.

## 2. Project Structure (Root Tree)
The repository is organized following standard scientific computing practices:

```text
Duffing-Oscillator-via-Spectral-Curvature/
├── .gitignore               # Excludes temporary Python & JAX files
├── LICENSE                  # MIT License full text
├── README.md                # Project documentation (Journal-style)
├── requirements.txt         # Dependencies (JAX, NumPy)
├── results/                 # Output logs and simulation data
│   └── .gitkeep
└── src/                     # Source code directory
    └── duffing_q1_curvature_jax.py  # Main simulation engine
```

---

<p align="center">
  <b>Author:</b> Hari Hardiyan <br>
  <b>Email:</b> <a href="mailto:lorozloraz@gmail.com">lorozloraz@gmail.com</a> <br><br>
  <b>Lead AI Development:</b> AI Tamer <br>
  <b>Assistant:</b> Microsoft Copilot
</p>

---

## 3. Physical Model: The Quantum Duffing Oscillator
The system investigates a particle in a non-linear potential field, defined by the sextic Hamiltonian:
$$\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2 + \alpha\hat{x}^4 + \gamma\hat{x}^6$$

Where:
*   **$m$**: Particle mass (normalized to 1.0).
*   **$\omega$**: Natural frequency.
*   **$\alpha$**: Quartic non-linearity (determines stability threshold).
*   **$\gamma$**: Sextic term for high-energy stabilization.

## 4. Methodology
### 4.1 Spectral Representation
Operators are represented in a truncated Hilbert space of dimension $N$. Position ($\hat{x}$) and momentum ($\hat{p}$) operators are constructed using ladder operators ($a, a^\dagger$) in Fock space to maintain quantum fidelity.

### 4.2 Curvature Matrix ($K$)
Structural stability is assessed via a 3x3 curvature matrix $K$. The system is deemed **stable** if $K$ is Positive Definite ($K \succ 0$), evaluated using:
1.  **Sylvester’s Criterion:** Positive principal minors.
2.  **Eigen-spectrum Analysis:** $\lambda_{min}(K) > 0$.

## 5. Implementation Details
*   **JAX Optimization:** Uses `jax.vmap` for parallel parameter scanning and `jax.lax.while_loop` for high-precision boundary root-finding.
*   **Precision:** Forced `x64` (double precision) for accurate eigenvalue decomposition of the Hamiltonian.
*   **Robustness:** Automated L2-distance benchmarking between different basis sizes ($N$) to verify convergence.

## 6. Example Output & Results
The following results were obtained using $N=128$ for the coarse map and $N=160$ for the robustness benchmark, with $\gamma=0.05$:

```text
[Coarse] PD3 fraction=100.00% | eigen-min: min=2.2951e-01, mean=8.4582e-01, q01=2.5062e-01, q05=2.9377e-01
[Boundary] Samples=80 | alpha* range=[-2.000, 0.131]
[Boundary observables] <x^2>: min=3.2203e-01, max=1.2797e+02 | kurtosis: mean=4.3361
[Robustness] Boundary L2 deviation across N: 7.0356e-01
```

## 7. Installation & Usage
To replicate the simulation:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/harihardiyan/Duffing-Oscillator-via-Spectral-Curvature.git
   cd Duffing-Oscillator-via-Spectral-Curvature
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the simulation:**
   ```bash
   python src/duffing_q1_curvature_jax.py
   ```

## 8. License
This project is licensed under the **MIT License**.

Copyright (c) 2026 Hari Hardiyan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions... (See LICENSE file for full text).

## 9. Citation
If you utilize this code for academic research, please cite:
> **Hardiyan, H. (2026).** *Stability Mapping of Non-Linear Quantum Oscillators using JAX-Accelerated Spectral Methods.* GitHub: harihardiyan/Duffing-Oscillator-via-Spectral-Curvature.

---
*Facilitated by AI Tamer and Microsoft Copilot as part of a computational physics development initiative.*
```

