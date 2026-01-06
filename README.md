

# Quantum Stability Analysis of the Duffing Oscillator via Spectral Curvature

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)
![JAX](https://img.shields.io/badge/Accelerated_by-JAX-orange?logo=google&logoColor=white)
![Physics](https://img.shields.io/badge/Field-Quantum_Physics-blueviolet)
![License](https://img.shields.io/badge/license-MIT-green)

## 1. Abstract
This repository provides a high-performance computational framework for analyzing the structural stability of a quantum Duffing oscillator. Utilizing **JAX** for accelerated linear algebra and automatic vectorization, the code evaluates the stability of the system’s ground state by constructing a 3x3 curvature matrix derived from spectral moments. The framework identifies phase boundaries between stable and unstable regimes in the parameter space of harmonic frequency ($\omega$) and quartic non-linearity ($\alpha$), while ensuring convergence through high-precision (float64) calculations.

## 2. Physical Model: The Quantum Duffing Oscillator
The system is defined by a sextic Hamiltonian of the form:
$$\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2 + \alpha\hat{x}^4 + \gamma\hat{x}^6$$

Where:
*   $m$: Particle mass (set to 1.0).
*   $\omega$: Reference harmonic frequency.
*   $\alpha$: Quartic non-linearity parameter.
*   $\gamma$: Sextic stabilization parameter.

---

### **Development & Authorship**
**Author:** Hari Hardiyan  
**Email:** [lorozloraz@gmail.com](mailto:lorozloraz@gmail.com)  

**Lead AI Development:** AI Tamer  
**Assistant:** Microsoft Copilot  

---

## 3. Methodology

### 3.1 Fock Space Representation
The code employs a spectral method to represent operators in a truncated Hilbert space of dimension $N$. Position ($\hat{x}$) and momentum ($\hat{p}$) operators are constructed using ladder operators ($a, a^\dagger$):
$$\hat{x} = \sqrt{\frac{\hbar}{2m\omega_{ref}}} (a + a^\dagger), \quad \hat{p} = i\sqrt{\frac{m\hbar\omega_{ref}}{2}} (a^\dagger - a)$$

### 3.2 The Curvature Matrix ($K$)
Stability is assessed via a curvature matrix $K$, representing the second-order variations of the energy surface. The matrix elements are calculated from ground-state expectation values $\langle \hat{x}^2 \rangle$ and $\langle \hat{x}^4 \rangle$:
*   $K_{xx} = m\omega^2 + 12\alpha\langle x^2 \rangle + 30\gamma\langle x^4 \rangle$
*   $K_{ww} = m\langle x^2 \rangle$
*   $K_{pp} = 1/m$

### 3.3 Stability Criteria
A state is considered **structurally stable** if the curvature matrix $K$ is Positive Definite ($K \succ 0$). The implementation checks this using:
1.  **Sylvester’s Criterion:** Ensuring all leading principal minors are positive.
2.  **Eigenvalue Analysis:** Verifying that the minimum eigenvalue $\lambda_{min}(K) > 0$.

## 4. Computational Implementation
The script leverages **JAX** for several critical advantages:
*   **X64 Precision:** Enabled via `jax_enable_x64` to avoid numerical drift.
*   **Vectorization (`vmap`):** Parallel evaluation of the stability map across grids.
*   **Root-Finding (`while_loop`):** A JAX-native bisection algorithm to precisely locate the stability boundary.
*   **Robustness Benchmarking:** Comparative analysis between different basis sizes ($N=128$ vs $N=160$).

## 5. License
**MIT License**

Copyright (c) 2024 Hari Hardiyan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## 6. Citation and References
If you use this framework in your research, please cite:

1.  **Hardiyan, H. (2026).** *Stability Analysis of Non-Linear Quantum Systems via JAX-Accelerated Curvature Matrices.*
2.  **Duffing, G. (1918).** *Erzwungene Schwingungen bei veränderlicher Eigenfrequenz.*
3.  **Bradbury, J., et al. (2018).** *JAX: Composable transformations of Python+NumPy programs.* [http://github.com/google/jax](http://github.com/google/jax).

---
*Facilitated by AI Tamer and Microsoft Copilot as part of a computational physics development initiative.*
