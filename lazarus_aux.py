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
    Um1, Cm1 = 2. / (gamma + 1), np.sqrt(2 * gamma * (gamma - 1)) / (gamma + 1)
    aux = 3 - lmb + (omega + 2 * (lmb - 1)) / gamma
    # Us = 0.25 * (aux - np.sqrt(aux ** 2 - 8 * (omega + 2 * (lmb - 1)) / gamma))
    # Cs = 1 - Us
    Cs = sonic_point_C(n, gamma, lmb, omega)
    if isinstance(Cs, Iterable):
        Cs = Cs[-1]
    Us = 1 - Cs
    A1 = ((omega + 2 * (lmb - 1)) / gamma - 3 * Us) * 2 * Cs
    B1 = -3 * Cs ** 2 + 3 * Us ** 2 - 2 * (lmb + 1) * Us + lmb
    aux = (2 * (lmb - 1) - (gamma - 1) * omega) / (2 * gamma)
    A2 = -2 * Cs ** 2 - 0.5 * n * (gamma - 1) * Us * Cs + (lmb - 1) * (0.5 * (gamma - 3) * Us + 1) - 3 * Cs * aux
    B2 = (-2 * Cs - 0.5 * n * (gamma - 1) * (1 - 2 * Us) + (lmb - 1) * (gamma - 3) / 2 - aux) * Cs
    dUdC = (B1 - A2 + np.sqrt((A2 - B1) ** 2 + 4 * A1 * B2)) / (2 * B2)
    dUdC = -dUdC        # dubious...
    dC = 1e-6

    # x_start = -1.0
    # x, y = rk.rk4(foo, x_start + x_plus, x_start, np.array([-Us-dUdC*dC, Cs+dC]), prec, max_n=max_n)
    # x2, y2 = rk.rk4(foo, -x_start + 1e1, -x_start, np.array([-Us+dUdC*dC, Cs-dC]), prec, max_n=max_n)
    # x = np.array([z for z in reversed(x)] + list(x2))
    # y = np.array([z for z in reversed(y)] + list(y2))

    x1_start = 1.0 * np.sign(lmb * s)
    x1_end = x1_start * 1e1 ** np.sign(-lmb * s)
    x1, y1 = rk.rk4(foo, x1_start, x1_end, np.array([-Us - dUdC * dC, Cs + dC]), prec, max_n=max_n)
    shock_ind = find_closest_to_point(y1, strong_shock_point(gamma))
    x1 = [-xp / x1[shock_ind] for xp in x1[:shock_ind + 1]]
    y1 = y1[:shock_ind + 1]
    x1, y1 = x1[-1::-1], y1[-1::-1]

    x2_start = x1[-1] + (x1[-1] - x1[-2]) * dC / (y1[-1][1] - y1[-2][1])
    y2_start = np.array([-Us + dUdC * dC, Cs - dC])
    x2, y2 = rk.rk4(foo, x2_start, 1.0, y2_start, prec, max_n=max_n)

    x, y = x1 + x2, y1 + y2
    return x, y


def sonic_point_C(n, gamma, lmb, omega=0.0):
    if n == 0:
        Cs = gamma * (1 - lmb) / (omega + (gamma - 2) * (1 - lmb))
    else:
        h = 0.5 - (omega + (gamma - 2) * (1 - lmb)) / (2 * n * gamma)
        Cs = h + np.array([1, -1]) * np.sqrt(h ** 2 + (1 - lmb) / n)
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
