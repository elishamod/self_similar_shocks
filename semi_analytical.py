import rk_solve as rk
import numpy as np


def delta1(u, c, n, omega, gamma, d):
    return u*(1-u)*(1-d-u) - c**2 * ((n+1)*u - (omega-2*d)/gamma)


def delta2(u, c, n, omega, gamma, d):
    aux = c*((1-u)*(1-d-u) - 0.5*(gamma-1)*u*(n*(1-u)+d))
    return aux - c**3 * (1 - ((gamma-1)*omega+2*d)/(2*gamma*(1-u)))


def solve_given_d(top_point, flip_deriv, n, omega, gamma, d, rk_eps=1e-5):
    U0, C0 = 2.0/(gamma+1), np.sqrt(2*gamma*(gamma-1))/(gamma+1)
    if n == 0:
        Cs = gamma * d / (omega + (gamma - 2) * d)
    else:
        h = 0.5 - (omega+(gamma-2)*d) / (2*n*gamma)
        sgt = 1
        if not top_point:
            sgt = -1
        Cs = h + sgt * np.sqrt(h**2 + d/n)
    Us = 1 - Cs

    a = (-n*Cs**2-2*Us*Cs+d*(1-2*Cs))
    b = 2*Cs*((omega-2*d)/gamma - (n+1)*Us)
    c = Cs * (d+d/gamma-2*Cs + 0.5*(gamma-1)*(n*(1-2*Cs)-d+omega/gamma))
    e = Cs*((gamma-1)*omega+2*d)/gamma - 2*Cs**2
    aux = (a - e) / (2 * c)
    sgt = 1
    if flip_deriv:
        sgt = -1
    dUdC_s = aux + sgt * np.sqrt(aux**2 + b / c)

    eps = sgt * 1e-8
    if True:
        Cs, Us = Cs + eps, Us + eps * dUdC_s
        dCdU = lambda uw, cw: delta2(uw, cw, n, omega, gamma, d) / delta1(uw, cw, n, omega, gamma, d)
        U, C = rk.rk4(dCdU, Us, U0, Cs, rk_eps, 2000)
    else:
        Cs, Us = Cs + eps, Us + eps * dUdC_s
        dUdC = lambda cw, uw: delta1(uw, cw, n, omega, gamma, d) / delta2(uw, cw, n, omega, gamma, d)
        C, U = rk.rk4(dUdC, Cs, C0, Us, rk_eps, 2000)
    return C, U


def solve_downwards(top_point, flip_deriv, n, omega, gamma, d, rk_eps=1e-5, C_final=1e-6):
    if n == 0:
        Cs = gamma * d / (omega + (gamma - 2) * d)
    else:
        h = 0.5 - (omega+(gamma-2)*d) / (2*n*gamma)
        sgt = 1
        if not top_point:
            sgt = -1
        Cs = h + sgt * np.sqrt(h ** 2 + d / n)
    Us = 1 - Cs

    a = (-n*Cs**2-2*Us*Cs+d*(1-2*Cs))
    b = 2*Cs*((omega-2*d)/gamma - (n+1)*Us)
    c = Cs * (d+d/gamma-2*Cs + 0.5*(gamma-1)*(n*(1-2*Cs)-d+omega/gamma))
    e = Cs*((gamma-1)*omega+2*d)/gamma - 2*Cs**2
    aux = (a - e) / (2 * c)
    sgt = 1
    if flip_deriv:
        sgt = -1
    dUdC_s = aux + sgt * np.sqrt(aux**2 + b / c)

    eps = -1e-7 * sgt
    Cs, Us = Cs + eps, Us + eps * dUdC_s
    dUdC = lambda cw, uw: delta1(uw, cw, n, omega, gamma, d) / delta2(uw, cw, n, omega, gamma, d)
    C, U = rk.rk4(dUdC, Cs, C_final, Us, rk_eps)
    return C, U


def solve(n, omega, gamma, d_eps=1e-6, rk_eps=1e-5):
    U0 = 2.0 / (gamma + 1)
    dmin, dmax = 0.0, 1.0
    d = 0.5 * (dmin + dmax)
    while dmax - dmin > d_eps:
        C, U = solve_given_d(True, False, n, omega, gamma, d, rk_eps)
        err = U[-1] - U0
        if err > 0:
            dmin = d
        else:
            dmax = d
        d = 0.5 * (dmin + dmax)
    C, U = solve_given_d(True, False, n, omega, gamma, d, rk_eps)
    return d, C, U


def solve_for_xi(U, C, n, omega, gamma, d, xi0=1.0):
    dzdc = lambda uw, cw: (cw**2 - (1-uw)**2) / delta2(uw, cw, n, omega, gamma, d)
    z = [np.log(xi0)]
    for j in range(1, len(U)):
        dC = C[j] - C[j - 1]
        z.append(z[-1] + dC * 0.5 * (dzdc(U[j], C[j]) + dzdc(U[j - 1], C[j - 1])))
    xi = [np.exp(zt) for zt in z]
    return xi


def solve_for_zeta(U, C, n, omega, gamma, d, zeta0=0.0):
    dzdc = lambda uw, cw: (cw**2 - (1-uw)**2) / delta2(uw, cw, n, omega, gamma, d)
    z = [zeta0]
    for j in range(1, len(U)):
        dC = C[j] - C[j - 1]
        z.append(z[-1] + dC * 0.5 * (dzdc(U[j], C[j]) + dzdc(U[j - 1], C[j - 1])))
    return z


def solve_given_d2(top_point, flip_deriv, n, omega, gamma, d, rk_eps=1e-5):
    U0, C0 = 2.0 / (gamma + 1), np.sqrt(2 * gamma * (gamma - 1)) / (gamma + 1)
    if n == 0:
        Cs = gamma * d / (omega + (gamma - 2) * d)
    else:
        h = 0.5 - (omega + (gamma - 2) * d) / (2 * n * gamma)
        sgt = 1
        if not top_point:
            sgt = -1
        Cs = h + sgt * np.sqrt(h ** 2 + d / n)
    Us = 1 - Cs

    a = (-n * Cs ** 2 - 2 * Us * Cs + d * (1 - 2 * Cs))
    b = 2 * Cs * ((omega - 2 * d) / gamma - (n + 1) * Us)
    c = Cs * (d + d / gamma - 2 * Cs + 0.5 * (gamma - 1) * (n * (1 - 2 * Cs) - d + omega / gamma))
    e = Cs * ((gamma - 1) * omega + 2 * d) / gamma - 2 * Cs ** 2
    aux = (a - e) / (2 * c)
    sgt = 1
    if flip_deriv:
        sgt = -1
    dUdC_s = aux + sgt * np.sqrt(aux ** 2 + b / c)

    eps = sgt * 1e-8
    Cs, Us = Cs + eps, Us + eps * dUdC_s
    dUdC = lambda cw, uw: delta1(uw, cw, n, omega, gamma, d) / delta2(uw, cw, n, omega, gamma, d)
    C, U = rk.flat_solve(dUdC, Cs, Us, C0, U0)
    return C, U


def flexible_solve(n, omega, gamma, d, Cs, Us, Cf, Uf):
    dUdC = lambda cw, uw: delta1(uw, cw, n, omega, gamma, d) / delta2(uw, cw, n, omega, gamma, d)
    C, U = rk.flat_solve(dUdC, Cs, Us, Cf, Uf)
    return C, U


def solve_downwards2(top_point, flip_deriv, n, omega, gamma, d, rk_eps=1e-5):
    U0, C0 = 0.0, 0.0
    if n == 0:
        Cs = gamma * d / (omega + (gamma - 2) * d)
    else:
        h = 0.5 - (omega + (gamma - 2) * d) / (2 * n * gamma)
        sgt = 1
        if not top_point:
            sgt = -1
        Cs = h + sgt * np.sqrt(h ** 2 + d / n)
    Us = 1 - Cs

    a = (-n * Cs ** 2 - 2 * Us * Cs + d * (1 - 2 * Cs))
    b = 2 * Cs * ((omega - 2 * d) / gamma - (n + 1) * Us)
    c = Cs * (d + d / gamma - 2 * Cs + 0.5 * (gamma - 1) * (n * (1 - 2 * Cs) - d + omega / gamma))
    e = Cs * ((gamma - 1) * omega + 2 * d) / gamma - 2 * Cs ** 2
    aux = (a - e) / (2 * c)
    sgt = 1
    if flip_deriv:
        sgt = -1
    dUdC_s = aux + sgt * np.sqrt(aux ** 2 + b / c)

    eps = sgt * 1e-8
    Cs, Us = Cs + eps, Us + eps * dUdC_s
    dUdC = lambda cw, uw: delta1(uw, cw, n, omega, gamma, d) / delta2(uw, cw, n, omega, gamma, d)
    C, U = rk.flat_solve(dUdC, Cs, Us, C0, U0)
    return C, U


def solve_phase2(n, omega, gamma, d, U0, C0, Cf, rk_eps=1e-5):
    dUdC = lambda cw, uw: delta1(uw, cw, n, omega, gamma, d) / delta2(uw, cw, n, omega, gamma, d)
    C, U = rk.rk4(dUdC, C0, Cf, U0, rk_eps)
    return C, U


def solve_post_strong_shock(n, omega, gamma, d, Cf, rk_eps=1e-5):
    Ustr = 2 / (gamma + 1)
    Cstr = np.sqrt(2 * gamma * (gamma - 1)) / (gamma + 1)
    dUdC = lambda cw, uw: delta1(uw, cw, n, omega, gamma, d) / delta2(uw, cw, n, omega, gamma, d)
    C, U = rk.rk4(dUdC, Cstr, Cf, Ustr, rk_eps)
    return C, U
