import copy
from math import sqrt, atan2, pi
import numpy as np


def rk4_one_step(f, tn, yn, h):
    k1 = h * f(tn, yn)
    k2 = h * f(tn + h / 2, yn + k1 / 2)
    k3 = h * f(tn + h / 2, yn + k2 / 2)
    k4 = h * f(tn + h, yn + k3)
    yn = yn + (k1 + 2 * (k2 + k3) + k4) / 6.0
    return yn


def rk4(f, a, b, y0, eps, max_n=-1, dx0=-1.0):
    help_sign = (b - a) / abs(b - a)
    h = float(b - a) / 10
    if dx0 > 0:
        h = dx0 * help_sign
    tn = a
    t = [tn]
    yn_prev = copy.deepcopy(y0)
    if not hasattr(y0, '__iter__'):
        yn_prev = np.array([y0])
    y = [yn_prev]
    while tn != b:
        if len(y) == max_n:
            break
        yn = rk4_one_step(f, tn, yn_prev, h)
        yn2 = rk4_one_step(f, tn, yn_prev, h / 2)
        yn2 = rk4_one_step(f, tn + h / 2, yn2, h / 2)
        while max(abs(yn - yn2)) > eps * abs(h / (b - a)):
            h /= 2
            yn = rk4_one_step(f, tn, yn_prev, h)
            yn2 = rk4_one_step(f, tn, yn_prev, h / 2)
            yn2 = rk4_one_step(f, tn + h / 2, yn2, h / 2)
        if help_sign * (tn + h) > help_sign * b:
            h = b - tn
            yn2 = rk4_one_step(f, tn, yn_prev, h)
            yn = yn2
        tn += h
        t.append(tn)
        y.append(yn2)
        yn_prev = yn2
        if max(abs(yn - yn2)) < 0.4 * eps * abs(h / (b - a)):
            h *= 1.1
        elif max(abs(yn - yn2)) > 0.6 * eps * abs(h / (b - a)):
            h *= 0.9
    return [t, y]


def angle_diff(ax, ay, bx, by):       # angle between two directions
    phi_a = atan2(ay, ax)
    phi_b = atan2(by, bx)
    dphi = abs(phi_a - phi_b)
    if dphi > pi:
        dphi = 2 * pi - dphi
    return dphi


def flat_solve(f, x0, y0, xf, yf, aprox_n=10000):
    max_n = int(aprox_n * 2.5)
    dr = sqrt((xf-x0)**2 + (yf-y0)**2) / aprox_n
    x, y = [x0], [y0]
    while len(x) < max_n:
        target_dirx, target_diry = xf - x[-1], yf - y[-1]
        deriv = f(x[-1], y[-1])
        dx = dr
        if angle_diff(target_dirx, target_diry, 1, deriv) > angle_diff(target_dirx, target_diry, -1, -deriv):
            dx *= -1
        dx /= sqrt(1 + deriv ** 2)
        dy = dx * deriv
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
    return [x, y]
