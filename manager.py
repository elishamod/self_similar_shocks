import matplotlib.pyplot as plt
import numpy as np
import semi_analytical
import hydro_sim
import sonic_singular_point
import time

show_type1 = False
diverging = True
gamma = 5.0 / 3.0
n = 1
omega = 10.0
# omega = 3.255469351
# omega = 2.090149
d = 1.74

num_of_cells = 200
L = 3e0


def CU_diagram(C, U, C1=0, U1=0):
    U0, C0 = 2.0 / (gamma + 1), np.sqrt(2 * gamma * (gamma - 1)) / (gamma + 1)
    plt.plot(U, C)
    plt.plot(U0, C0, 'rx')
    plt.plot(np.linspace(0, 1, 100), np.linspace(1, 0, 100))
    plt.xlabel('U')
    plt.xlim([0, 1])
    plt.ylabel('C')
    plt.ylim([0, 1])
    plt.title('C-U Diagram')
    if n == 0:
        Cs = gamma * d / (omega + (gamma - 2) * d)
    else:
        h = 0.5 - (omega+(gamma-2)*d) / (2*n*gamma)
        sgt = 1
        if not diverging:
            sgt = -1
        Cs = h + sgt * np.sqrt(h ** 2 + d / n)
        Cs = h + np.sqrt(h**2 + d/n)
    Us = 1 - Cs
    plt.plot(Us, Cs, '*')
    if C1 != 0:
        plt.plot(U1, C1)

    if show_type1:
        Ua = np.linspace(1.0/gamma+1e-5, 1.0, 300)
        Ca = Ua * np.sqrt(0.5 * (gamma - 1) * (1.0 - Ua) / (Ua - 1.0/gamma))
        plt.plot(Ua, Ca)

    plt.show()


start_time = time.time()
print('hi')

C, U = semi_analytical.solve_given_d(True, False, n, omega, gamma, d)
print(len(C))
print('Finished type II part 1')
C1, U1 = [], []
#C1, U1 = semi_analytical.solve_downwards(True, False, n, omega, gamma, d, 1e-8)
C = [x for x in reversed(C)]
U = [x for x in reversed(U)]
C2 = C + C1
U2 = U + U1
print(len(C))
print('Finished type II part 2')
print('delta = ' + str(d))
print(str(time.time() - start_time) + ' sec')
CU_diagram(C2, U2)

# C3, U3 = sonic_singular_point.semi_analytical(diverging, n, omega, gamma, 0.0, 1e-8)
# print 'Finished gap solution'
# print str(time.time() - start_time) + ' sec'
# CU_diagram(C3, U3)

# CU_diagram(C2, U2, C3, U3)

# xi = semi_analytical.solve_for_xi(U2, C2, n, omega, gamma, d)
# plt.figure()
# plt.plot(xi, C2, '.-')
# # plt.plot(xi[0:-2], C2[0:-2], '.-')
# plt.title('C vs. xi')
# plt.xlabel('xi')
# plt.ylabel('C')
# plt.show()

# plt.figure()
# x, u, xm, um, V, p, q, e, c, R = hydro_sim.solve(diverging, n, omega, gamma, num_of_cells, L)
# plt.plot(xm, 1 / V, '--')
# x, u, xm, um, V, p, q, e, c, R = hydro_sim.solve(diverging, n, omega, gamma, num_of_cells / 2, L)
# plt.plot(xm, 1 / V, '--')
# plt.title('Density vs. Radius')
# plt.xlabel('Radius')
# plt.ylabel('Density')
# plt.show()
