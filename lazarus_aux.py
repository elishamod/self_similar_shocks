import numpy as np
import rk_solve as rk
from collections.abc import Iterable


def D(V, C):
    return (1 + V)**2 - C**2


def F(V, C, lmb, n, gamma, omega=0.0):
    ttt = (1+V)**2 + n*(gamma-1)*V*(1+V)/2 + (lmb-1)*((3-gamma)*V+2)/2
    return C**3 * (1 + (lmb-1-0.5*(gamma-1)*omega)/(gamma*(1+V))) - C * ttt


def G(V, C, lmb, n, gamma, omega=0.0):
    return C**2*((n+1)*V + (omega+2*(lmb-1))/gamma) - V*(1+V)*(lmb+V)


def F1(V, C, gamma, eta):
    return -C**3 * (eta+0.5*(gamma-1))/(gamma*(1+V)) + C * eta*((3-gamma)*V/2+1)


def G1(V, C, gamma, eta):
    return C**2*(1-2*eta)/gamma + eta*V*(1+V)


def func(y, x, lmb, n, gamma, omega=0.0):
    sss = np.array([G(y[0], y[1], lmb, n, gamma, omega), F(y[0], y[1], lmb, n, gamma, omega)])
    return sss / (-lmb*x*D(y[0], y[1]))


def func_eta(y, x, gamma, eta):
    sss = np.array([G1(y[0], y[1], gamma, eta), F1(y[0], y[1], gamma, eta)])
    return sss / (-eta*x*D(y[0], y[1]))


def func2(y, w, lmb, n, gamma, sigma, omega=0.0):
    sss = np.array([G(y[0], y[1], lmb, n, gamma, omega), F(y[0], y[1], lmb, n, gamma, omega)])
    return sss / (lmb*sigma*w*D(y[0], y[1]))


def logfunc2(y, logw, lmb, n, gamma, sigma, omega=0.0):
    sss = np.array([G(y[0], y[1], lmb, n, gamma, omega), F(y[0], y[1], lmb, n, gamma, omega)])
    return sss / (lmb*sigma*D(y[0], y[1]))


def solve1(n, gamma, lmb, omega=0.0, x_end=1e4, prec1=1e-4, prec2=1e-6, switch=True):
    foo = lambda xw, yw: func(yw, xw, lmb, n, gamma, omega)
    Vm1, Cm1 = -2. / (gamma + 1), np.sqrt(2*gamma*(gamma-1)) / (gamma + 1)
    lil_x = 2e-2
    if switch:
        x, y = rk.rk4(foo, -1.0, lil_x, np.array([Vm1, Cm1]), prec1, max_n=2000)
        x2, y2 = rk.rk4(foo, lil_x, x_end, np.array([y[-1][0], y[-1][1]]), prec2, max_n=2000)
        return x + x2, y + y2
    else:
        x, y = rk.rk4(foo, 1.0, 1e2, np.array([Vm1, Cm1]), prec1, max_n=2000)
        return x, y


def solve1cont(n, gamma, lmb, omega, y_start, x_start=1e0, x_end=1e4, prec=1e-6):
    foo = lambda xw, yw: func(yw, xw, lmb, n, gamma, omega)
    x, y = rk.rk4(foo, x_start, x_end, y_start, prec, max_n=2000)
    return x, y


def solve2(n, gamma, lmb, omega=0.0, w_end=2e3, prec=1e-2):
    V0 = (-2*(lmb-1)-omega)/((n+1)*gamma)
    sigma = 1./lmb * (1 + (lmb-1 - 0.5*(gamma-1)*omega)/(gamma*(1+V0)))  # Correct this
    foo = lambda ww, yw: func2(yw, ww, lmb, n, gamma, sigma, omega)
    w0 = 1e-3
    w, y = rk.rk4(foo, w0, w_end, np.array([V0, -1./w0]), prec, max_n=2000)
    return w, y


def solve2cont(n, gamma, lmb, omega, y_start, w_start=1e0, w_end=2e3, prec=1e-2):
    V0 = (-2*(lmb-1)-omega)/((n+1)*gamma)
    sigma = 1./lmb * (1 + (lmb-1 - 0.5*(gamma-1)*omega)/(gamma*(1+V0)))
    foo = lambda ww, yw: func2(yw, ww, lmb, n, gamma, sigma, omega)
    w, y = rk.rk4(foo, w_start, w_end, y_start, prec, max_n=2000)
    return w, y


def solve_eta(gamma, eta, x_end=1e4, prec1=1e-4, prec2=1e-6, switch=True):
    foo = lambda xw, yw: func_eta(yw, xw, gamma, eta)
    Vm1, Cm1 = -2. / (gamma + 1), np.sqrt(2*gamma*(gamma-1)) / (gamma + 1)
    lil_x = 2e-2
    if switch:
        x, y = rk.rk4(foo, -1.0, lil_x, np.array([Vm1, Cm1]), prec1, max_n=2000)
        x2, y2 = rk.rk4(foo, lil_x, x_end, np.array([y[-1][0], y[-1][1]]), prec2, max_n=2000)
        return x + x2, y + y2
    else:
        x, y = rk.rk4(foo, 1.0, 1e2, np.array([Vm1, Cm1]), prec1, max_n=2000)
        return x, y


def solve_from_sonic(n, gamma, lmb, omega=0.0, s=-1, x_plus=0.5, prec=1e-7, max_n=1000):
    foo = lambda xw, yw: func(yw, xw, lmb, n, gamma, omega)
    Cs = sonic_point_C(n, gamma, lmb, omega, guess_correct_point=True)
    Us = 1 - Cs

    dUdC = get_dUdC(Us, Cs, n, gamma, lmb, omega)
    dC = 1e-6

    x1_start = 1.0 * np.sign(lmb * s)
    x1_end = x1_start * 1e1 ** np.sign(-lmb * s)
    x1, y1 = rk.rk4(foo, x1_start, x1_end, np.array([-Us - dUdC * dC, Cs + dC]), prec, max_n=max_n)
    shock_ind = find_closest_to_point(y1, strong_shock_point(gamma))
    x1 = [np.sign(lmb * s) * xp / x1[shock_ind] for xp in x1[:shock_ind + 1]]
    y1 = y1[:shock_ind + 1]
    x1, y1 = x1[-1::-1], y1[-1::-1]

    x2_start = x1[-1] + (x1[-1] - x1[-2]) * dC / (y1[-1][1] - y1[-2][1])
    y2_start = np.array([-Us + dUdC * dC, Cs - dC])
    if lmb * s > 0:
        x2_end = 10 * x2_start
    else:
        x2_end = 1.0
    x2, y2 = rk.rk4(foo, x2_start, x2_end, y2_start, prec, max_n=max_n)

    x, y = x1 + x2, y1 + y2
    return x, y


def get_dUdC(Us, Cs, n, gamma, lmb, omega=0.0):
    """Returns dU/dC at the sonic point"""
    dD1dU = -(n + 1) * Cs ** 2 + (1 - Us) * (lmb - Us) - Us * (lmb - Us) - Us * (1 - Us)
    dD1dC = ((omega + 2 * (lmb - 1)) / gamma - (n + 1) * Us) * 2 * Cs
    dD2dU = (2 * (Us - 1) - 0.5 * n * (gamma - 1) * (1 - 2 * Us) + (lmb - 1) * (gamma - 3) / 2) * Cs
    dD2dU -= (2 * (lmb - 1) - (gamma - 1) * omega) / (2 * gamma * (1 - Us) ** 2) * Cs ** 3
    dD2dC = (1 - Us) ** 2 - 0.5 * n * (gamma - 1) * Us * (1 - Us) + (lmb - 1) * ((gamma - 3) * Us / 2 + 1)
    dD2dC -= (1 + (2 * (lmb - 1) - (gamma - 1) * omega) / (2 * gamma * (1 - Us))) * 3 * Cs ** 2
    # A * (dU/dC)**2 + B * dU/dC + C = 0
    A, B, C = dD2dU, dD2dC - dD1dU, -dD1dC
    discriminant = np.sqrt(B ** 2 - 4 * A * C)
    dUdC = (-B + np.array([-1, 1]) * discriminant) / (2 * A)
    Vshock, Cshock = strong_shock_point(gamma)
    sonic_to_shock_angle = np.arctan((-Vshock - Us) / (Cshock - Cs))
    sol_line_angle = np.arctan(dUdC)
    correct_ind = np.argmin(np.abs(sonic_to_shock_angle - sol_line_angle))
    return dUdC[correct_ind]


def sonic_point_C(n, gamma, lmb, omega=0.0, guess_correct_point=False):
    if n == 0:
        Cs = gamma * (1 - lmb) / (omega + (gamma - 2) * (1 - lmb))
    else:
        h = 0.5 - (omega + (gamma - 2) * (1 - lmb)) / (2 * n * gamma)
        Cs = h + np.array([1, -1]) * np.sqrt(h ** 2 + (1 - lmb) / n)
        if guess_correct_point:
            probably_correct_index = find_closest_to_point([(ccc - 1, ccc) for ccc in Cs], strong_shock_point(gamma))
            Cs = Cs[probably_correct_index]
    return Cs


def strong_shock_point(gamma):
    return [-2 / (gamma + 1), np.sqrt(2 * gamma * (gamma - 1)) / (gamma + 1)]


def find_closest_to_point(y_set, y_point):
    min_dist, closest_ind = np.inf, None
    for i in range(len(y_set)):
        curr_dist = (y_set[i][0] - y_point[0]) ** 2 + (y_set[i][1] - y_point[1]) ** 2
        if curr_dist < min_dist:
            min_dist = curr_dist
            closest_ind = i
    return closest_ind
