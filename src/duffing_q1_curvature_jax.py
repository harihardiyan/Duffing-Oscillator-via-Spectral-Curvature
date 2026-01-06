
# duffing_q1_curvature_jax.py (bagian 1/2)
# Pure-physics stability for Duffing oscillator via curvature matrix (JAX, x64)

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax

hbar = 1.0
m    = 1.0

# =========================
# Operators and Hamiltonian
# =========================
def create_fock_ops(N):
    n = jnp.arange(N)
    s = jnp.sqrt(n[1:])
    a    = jnp.zeros((N, N), dtype=jnp.complex128)
    adag = jnp.zeros((N, N), dtype=jnp.complex128)
    a    = a.at[1:, :-1].set(jnp.diag(s).astype(jnp.complex128))
    adag = adag.at[:-1, 1:].set(jnp.diag(s).astype(jnp.complex128))
    num  = adag @ a
    return a, adag, num

def x_op(N, omega_ref=1.0):
    a, adag, _ = create_fock_ops(N)
    scale = jnp.sqrt(hbar/(2*m*omega_ref))
    return scale * (a + adag)

def p_op(N, omega_ref=1.0):
    a, adag, _ = create_fock_ops(N)
    scale = 1j*jnp.sqrt(m*hbar*omega_ref/2.0)
    return scale * (adag - a)

def H_duffing(N, omega, alpha, gamma):
    X = x_op(N)
    P = p_op(N)
    X2 = X @ X
    X4 = X2 @ X2
    X6 = X4 @ X2
    H_kin  = (P @ P)/(2*m)
    H_quad = 0.5*m*(omega**2)*X2
    H_nl   = alpha*X4 + gamma*X6
    H      = H_kin + H_quad + H_nl
    return H, X, P, X2, X4

def ground_state(N, omega, alpha, gamma):
    H, X, P, X2, X4 = H_duffing(N, omega, alpha, gamma)
    E, U = jnp.linalg.eigh(H)
    psi0 = U[:, 0]
    psi0 = psi0 / jnp.linalg.norm(psi0)
    return H, X, P, X2, X4, E, psi0

# =========================
# Curvature matrix K from spectral moments
# =========================
def curvature_matrix(N, omega, alpha, gamma):
    H, X, P, X2, X4, E, psi0 = ground_state(N, omega, alpha, gamma)
    x2 = jnp.real(jnp.vdot(psi0, (X @ X) @ psi0))
    x4 = jnp.real(jnp.vdot(psi0, X4 @ psi0))
    K_xx = m*(omega**2) + 12.0*alpha*x2 + 30.0*gamma*x4
    K_pp = 1.0/m
    K_xp = 0.0
    K_ww = m * x2
    K_xw = 0.0
    K_pw = 0.0
    K = jnp.array([[K_xx, K_xp, K_xw],
                   [K_xp, K_pp, K_pw],
                   [K_xw, K_pw, K_ww]], dtype=jnp.float64)
    return K, x2, x4

def sylvester_pd3(H, tol=1e-12):
    m11  = H[0,0]
    det2 = H[0,0]*H[1,1] - H[0,1]*H[1,0]
    det3 = jnp.linalg.det(H)
    return (m11 > tol) & (det2 > tol) & (det3 > tol)

def eigen_min(H):
    return jnp.min(jnp.linalg.eigvalsh(H))

# =========================
# Coarse phase map (JAX vmap)
# =========================
def coarse_map(N=96, gamma=0.05,
               omega_range=(0.2, 2.0), alpha_range=(-2.0, 0.2),
               n_omega=60, n_alpha=60):
    omegas = jnp.linspace(omega_range[0], omega_range[1], n_omega)
    alphas = jnp.linspace(alpha_range[0], alpha_range[1], n_alpha)

    def eval_point(w, a):
        K, _, _ = curvature_matrix(N, w, a, gamma)
        pd   = sylvester_pd3(K)
        emin = eigen_min(K)
        return pd, emin

    vm_w = jax.vmap(eval_point, in_axes=(0, None))
    vm_a = jax.vmap(vm_w, in_axes=(None, 0))
    pd_map, emin_map = vm_a(omegas, alphas)
    return pd_map, emin_map, omegas, alphas
# duffing_q1_curvature_jax.py (bagian 2/2)

# =========================
# Refined boundary (JAX while_loop)
# =========================
def find_boundary_alpha_for_omega(N, gamma, omega, alpha_lo, alpha_hi, target=0.0):
    def cond_fun(state):
        alo, ahi, _ = state
        return (ahi - alo) > 1e-4

    def body_fun(state):
        alo, ahi, _ = state
        amid = 0.5*(alo + ahi)
        K, _, _ = curvature_matrix(N, omega, amid, gamma)
        emin  = eigen_min(K)
        choose_hi = emin >= target
        new_alo = jnp.where(choose_hi, alo, amid)
        new_ahi = jnp.where(choose_hi, amid, ahi)
        return (new_alo, new_ahi, emin)

    alo0, ahi0 = alpha_lo, alpha_hi
    alo, ahi, emin_last = lax.while_loop(cond_fun, body_fun, (alo0, ahi0, jnp.nan))
    alpha_star = 0.5*(alo + ahi)
    return alpha_star, emin_last

def refined_boundary(N=128, gamma=0.05,
                     omega_range=(0.2, 2.0), n_omega=80,
                     alpha_bracket=(-2.0, 0.2)):
    omegas = jnp.linspace(omega_range[0], omega_range[1], n_omega)
    vm = jax.vmap(lambda w: find_boundary_alpha_for_omega(
        N, gamma, w, alpha_bracket[0], alpha_bracket[1], target=0.0
    ))
    alpha_star, emin_at_star = vm(omegas)
    return omegas, alpha_star, emin_at_star

# =========================
# Observables along boundary
# =========================
def boundary_observables(N, gamma, omegas, alphas):
    def obs_at(w, a):
        H, X, P, X2, X4, E, psi0 = ground_state(N, w, a, gamma)
        x2 = jnp.real(jnp.vdot(psi0, (X @ X) @ psi0))
        x4 = jnp.real(jnp.vdot(psi0, X4 @ psi0))
        kurt = x4 / jnp.maximum(x2**2, 1e-18)
        return x2, kurt
    vm = jax.vmap(obs_at, in_axes=(0, 0))
    x2, kurt = vm(omegas, alphas)
    return x2, kurt

# =========================
# Metrics and summary
# =========================
def curve_l2_distance(a1, a2):
    mask = (~jnp.isnan(a1)) & (~jnp.isnan(a2))
    return jnp.sqrt(jnp.mean((a1[mask] - a2[mask])**2))

def summarize_maps(pd_map, emin_map):
    frac = float(jnp.mean(pd_map.astype(jnp.float64)))
    e_min = float(jnp.nanmin(emin_map))
    e_mean = float(jnp.nanmean(emin_map))
    e_q01 = float(jnp.nanquantile(emin_map, 0.01))
    e_q05 = float(jnp.nanquantile(emin_map, 0.05))
    return frac, e_min, e_mean, e_q01, e_q05

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Coarse overview
    pd_map, emin_map, omegas_c, alphas_c = coarse_map()
    frac, e_min, e_mean, e_q01, e_q05 = summarize_maps(pd_map, emin_map)
    print(f"[Coarse] PD3 fraction={frac*100:.2f}% | "
          f"eigen-min: min={e_min:.4e}, mean={e_mean:.4e}, "
          f"q01={e_q01:.4e}, q05={e_q05:.4e}")

    # Refined boundary
    omegas_f, alpha_star_f, emin_star_f = refined_boundary()
    print(f"[Boundary] Samples={alpha_star_f.shape[0]} | "
          f"alpha* range=[{float(jnp.nanmin(alpha_star_f)):.3f}, "
          f"{float(jnp.nanmax(alpha_star_f)):.3f}]")

    # Observables along boundary
    x2_f, kurt_f = boundary_observables(128, 0.05, omegas_f, alpha_star_f)
    print(f"[Boundary observables] <x^2>: min={float(jnp.nanmin(x2_f)):.4e}, "
          f"max={float(jnp.nanmax(x2_f)):.4e} | "
          f"kurtosis: mean={float(jnp.nanmean(kurt_f)):.4f}")

    # Robustness check
    omegas_g, alpha_star_g, _ = refined_boundary(N=160, gamma=0.05)
    dist = curve_l2_distance(alpha_star_f, alpha_star_g)
    print(f"[Robustness] Boundary L2 deviation across N: {float(dist):.4e}")
