"""A tool that tries to find the self-similar solution automatically"""
import numpy as np
import matplotlib.pyplot as plt
from utils import get_parser, init_styled_plot, finish_styled_plot
import lazarus_aux as lz
gap_coefficients = {0: 0, 1: 0.5905, 2: 1.5148}


def main():
    omega, gamma, s, n = read_input()
    sol = solve_given_lambda(omega, gamma, s, n, lambda_initial_guess(omega, gamma, s, n))
    plot_CU_diagram([sol])


def lambda_initial_guess(omega: float, gamma: float, s: int, n: int) -> float:
    """Using the analytical approximation from https://doi.org/10.1063/5.0047518"""
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


def solve_given_lambda(omega, gamma, s, n, lmb):
    x, y = lz.solve1(n, gamma, lmb, omega, prec1=1e-4, x_end=1e1, prec2=1e-7, switch=(s * lmb < 0))
    V = [y[i][0] for i in range(len(x))]
    C = [y[i][1] for i in range(len(x))]
    x, V, C = np.array(x), np.array(V), np.array(C)
    return x, V, C


def plot_CU_diagram(solutions, gamma=None, s=None, n=None):
    init_styled_plot()
    fig, ax = plt.subplots(1 , 1)
    for x, V, C in solutions:
        ax.plot(C, -V)
    ax.set_xlabel('C')
    ax.set_ylabel('U')
    finish_styled_plot()
    plt.show()


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
    args = ap.parse_args()
    return args.omega, args.gamma, args.s, args.n


if __name__ == '__main__':
    main()
