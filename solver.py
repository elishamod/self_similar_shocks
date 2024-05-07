"""
A tool that tries to find the self-similar solution automatically.
Use -h to get help.
"""
import numpy as np
import matplotlib.pyplot as plt
from utils import get_parser, init_styled_plot, finish_styled_plot
import lazarus_aux as lz
from collections import namedtuple
PhysicalParams = namedtuple('PhysicalParams', ['omega', 'gamma', 'n', 's'])


def main():
    params, user_lmb, profile_plot_flag = read_input()
    if user_lmb is None:
        lmb = lambda_initial_guess(params)
        sol = solve_given_lambda(params, lmb)
    else:
        lmb = user_lmb
        sol = solve_given_lambda(params, lmb)
    plot_UC_diagram([sol], params=params, lmb=lmb, show=False)
    if profile_plot_flag:
        profile_plot([sol], params=params, lmb=lmb, show=False)
    plt.show()


def lambda_initial_guess(params: PhysicalParams) -> float:
    """Using the analytical approximation from https://doi.org/10.1063/5.0047518"""
    gap_coefficients = {0: 0, 1: 0.5905, 2: 1.5148}
    omega, gamma, s, n = params.omega, params.gamma, params.s, params.n
    if s > 0:  # Diverging shock
        if n in gap_coefficients.keys():
            gap_coefficient = gap_coefficients[n]
        else:
            # Exotic geometry - I never studied this, so this is likely a very bad guess
            gap_coefficient = 1.0
        gap_size = gap_coefficient * (gamma - 1) / (gamma + 3)
        d_omega = omega - (n + 1)
        if d_omega <= 0:
            return 1.0 - d_omega / 2
        elif d_omega <= gap_size:
            return 1.0
        else:
            return 1.0 - (d_omega - gap_size) / (2 + np.sqrt(2 * gamma / (gamma - 1)))
    elif s < 0:  # converging shock
        lmb_at_0 = 1 + n / (1 + 2 / gamma + np.sqrt(2 * gamma / (gamma - 1)))
        eta1 = 1 / (2 + np.sqrt(2 * gamma / (gamma - 1)))
        eta2 = 1 - 0.4 * (1 - 1 / gamma) ** 0.3
        omega_b = n + 2 + (1 - n) / 2 * np.sqrt(1 - 1 / gamma)
        omega_s = (eta2 * omega_b - lmb_at_0) / (eta2 - eta1)
        if omega < omega_s:
            return lmb_at_0 - eta1 * omega
        else:
            return eta2 * (omega_b - omega)
    else:
        raise Exception("s should be either 1 or -1. s = 0 is meaningless.")


def solve_given_lambda(params: PhysicalParams, lmb):
    # x, y = lz.solve1(params.n, params.gamma, lmb, params.omega,
    #                  prec1=1e-4, x_end=1e1, prec2=1e-7, switch=(params.s * lmb < 0))
    x, y = lz.solve_from_sonic(params.n, params.gamma, lmb, params.omega, params.s)
    V = [y[i][0] for i in range(len(x))]
    C = [y[i][1] for i in range(len(x))]
    x, V, C = np.array(x), np.array(V), np.array(C)
    return x, V, C


def profile_plot(solutions, params: PhysicalParams, lmb, show=True):
    init_styled_plot(font_type='David')
    fig, axes = plt.subplots(2, 1)
    for x, V, C in solutions:
        axes[0].plot(x, -V)
        axes[1].plot(x, C)
    for ax in axes:
        ax.set_xlabel('x')
    axes[0].set_ylabel('U')
    axes[1].set_ylabel('C')
    finish_styled_plot()
    if show:
        plt.show()


def plot_UC_diagram(solutions, params: PhysicalParams, lmb=None, show=True):
    omega, gamma, s, n = params.omega, params.gamma, params.s, params.n
    init_styled_plot(font_type='David')
    fig, ax = plt.subplots(1, 1)
    vvv = np.linspace(-1.0, 0.0, 10)
    ax.plot(-vvv, 1 + vvv, label='Sonic line', color='C1')
    for x, V, C in solutions:
        ax.plot(-V, C, label='Solution')
    V_shock, C_shock = -2 / (gamma + 1), np.sqrt(2 * gamma * (gamma - 1)) / (gamma + 1)
    ax.plot(-V_shock, C_shock, 'rx', markersize=8, label='Strong shock')
    if lmb is not None:
        Cs = lz.sonic_point_C(n=n, gamma=gamma, lmb=lmb, omega=omega)
        ax.plot(1 - Cs, Cs, 'g*', markersize=8, label='Singular point')
    ax.plot(0, 0, 'k.', markersize=8, label='_origin')
    set_UC_diagram_axes_limits(ax, solutions)
    ax.set_xlabel('U')
    ax.set_ylabel('C')
    gamma_str = str(gamma)
    if gamma == 5. / 3.:
        gamma_str = '5/3'
    ttl = rf's={s}, n={n}, $\gamma$={gamma_str}, $\omega$={omega}, $\lambda$={lmb}'
    plt.title(ttl)
    finish_styled_plot()
    if show:
        plt.show()


def set_UC_diagram_axes_limits(ax, solutions):
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0
    for _, V, C in solutions:
        x_vec, y_vec = -V, C
        x_vec_min_trunc = 0.5 * np.floor(2 * min(x_vec))
        y_vec_min_trunc = 0.5 * np.floor(2 * min(y_vec))
        y_vec_max_trunc = 0.5 * np.ceil(2 * max(y_vec))
        xmin = max(-1.5, min(xmin, x_vec_min_trunc))
        ymin = max(-1.5, min(ymin, y_vec_min_trunc))
        xmax = min(1.5, max(ymax, y_vec_max_trunc))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def read_input():
    ap = get_parser(__doc__)
    ap.add_argument('-w', '--omega', type=float, default=0.0,
                    help='initial density exponent')
    ap.add_argument('-g', '--gamma', type=float, default=5/3,
                    help='gas adiabatic index')
    ap.add_argument('-s', type=int, default=1, metavar='direction',
                    help='s = 1 for diverging, s = -1 for converging')
    ap.add_argument('-n', type=int, default=0, metavar='geometry',
                    help='n = 0 - plane, n = 1 - cylinder, n = 2 - sphere')
    ap.add_argument('-p', '--profileplot', action='store_true')
    ap.add_argument('-l', '--lambda', type=float, default=None, dest='lmb',
                    help='specify a specific lambda and do not look for a solution automatically')
    args = ap.parse_args()
    return PhysicalParams(omega=args.omega, gamma=args.gamma, s=args.s, n=args.n), args.lmb, args.profileplot


if __name__ == '__main__':
    main()
