import matplotlib.pyplot as plt
import numpy as np
import lazarus_aux as lz
from os import mkdir

n = 2
gamma = 5./3.
omega = 2.1
lmb = 0.810136
reflection = True

plot_bv = False
plot_profiles = False
save_results = False

# x, y = lz.solve_from_sonic(n, gamma, lmb, omega, 0.33875)
# x, y = lz.solve1(n, gamma, lmb, omega, prec1=1e-3, x_end=1e2, prec2=1e-7, switch=reflection)
# x, y = lz.solve1(n, gamma, lmb, omega, prec1=1e-4, x_end=0.92184, prec2=1e-7, switch=reflection)
x, y = lz.solve1(n, gamma, lmb, omega, prec1=1e-4, x_end=3.9298, prec2=1e-7, switch=reflection)

V = [y[i][0] for i in range(len(x))]
C = [y[i][1] for i in range(len(x))]
x, V, C = np.array(x), np.array(V), np.array(C)
print(len(C))

xc, yc = lz.solve1cont(n, gamma, lmb, omega, y[-1], x[-1], 1e1)
Vc = [yc[i][0] for i in range(len(xc))]
Cc = [yc[i][1] for i in range(len(xc))]
xc, Vc, Cc = np.array(xc), np.array(Vc), np.array(Cc)

if plot_bv:
    indices = np.where(x > 0)
    xp, Vp, Cp = x[indices], V[indices], C[indices]

    Vq = -1 + ((gamma-1)*(1+Vp) + 2*Cp**2/(1+Vp))/(gamma+1)
    Cq = -np.sqrt(Cp**2 + 0.5*(gamma-1)*((1+Vp)**2 - (1+Vq)**2))
    Vqc = -1 + ((gamma-1)*(1+Vc) + 2*Cc**2/(1+Vc))/(gamma+1)
    Cqc = -np.sqrt(Cc**2 + 0.5*(gamma-1)*((1+Vc)**2 - (1+Vqc)**2))

    # w, yr = lz.solve2(n, gamma, lmb, omega, w_end=1e-1, prec=1e-3)
    # w, yr = lz.solve2(n, gamma, lmb, omega, w_end=1.346, prec=1e-3)
    w, yr = lz.solve2(n, gamma, lmb, omega, w_end=0.6241, prec=1e-3)
    Vr = np.array([yr[i][0] for i in range(len(w))])
    Cr = np.array([yr[i][1] for i in range(len(w))])
    print(len(Cr))

    wc, yrc = lz.solve2cont(n, gamma, lmb, omega, yr[-1], w[-1], 1e4, prec=1e-2)
    Vrc = np.array([yrc[i][0] for i in range(len(wc))])
    Crc = np.array([yrc[i][1] for i in range(len(wc))])

s = 1
if reflection ^ (lmb < 0):
    s = -1

# plt.rcParams["font.family"] = "Times New Roman"
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('legend', fontsize=13)
plt.rc('axes', titlesize=16)

fig1 = plt.figure(1)
plt.plot(-V, C, label='(a)')
if plot_bv:
    plt.plot(-Vc, Cc, 'C0--')
    plt.plot(-Vq, Cq, 'g--', label='(v)')
    # plt.plot(-Vqc, Cqc, 'g--')
    plt.ylim(plt.ylim())
    plt.plot(-Vr, Cr, 'r', label='(b)')
    plt.plot(-Vrc, Crc, 'r--')

    aux = (gamma-1)/(gamma+1.) * (1+V[-1])/(1+Vq[-1])
    M = np.sqrt(2./(gamma-1) * aux / (1-aux))
    print('M = ' + str(M))

Vm1, Cm1 = -2. / (gamma + 1), np.sqrt(2 * gamma * (gamma - 1)) / (gamma + 1)
plt.plot(-Vm1, Cm1, 'rx', label='Strong shock')
vvv = np.linspace(-1.0, 0.5, 10)
plt.plot(-vvv, 1+vvv, label='Sonic line', color='C1')
plt.plot(-vvv, -(1+vvv), 'C1')
plt.plot(0.0, 0.0, 'k.')

if n == 0:
    Cs = np.array([gamma * (1-lmb) / (omega + (gamma - 2) * (1-lmb))])
else:
    h = 0.5 - (omega+(gamma-2)*(1-lmb)) / (2*n*gamma)
    Cs = np.array([h + np.sqrt(h ** 2 + (1-lmb) / n), h - np.sqrt(h ** 2 + (1-lmb) / n)])
plt.plot(1-Cs, Cs, 'g*', label='Singular point')

gamma_str = str(gamma)
if gamma == 5./3.:
    gamma_str = '5/3'

ttl = r's=' + str(s) + ', n=' + str(n) + ', $\gamma=$' + gamma_str\
      + ', $\omega=$' + str(omega) + r', $\lambda=$' + str(lmb)
plt.title(ttl)
plt.xlabel('U')
plt.ylabel('C')
plt.grid(True)
plt.ylim(0, 1)
plt.xlim(0, 1)
# plt.legend()
fig1.tight_layout()
plt.show()

if plot_profiles:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('axes', labelsize=24)
    plt.rc('xtick', labelsize=19)
    plt.rc('ytick', labelsize=19)
    plt.rc('legend', fontsize=20)
    # First find k
    U0 = (2 * (lmb - 1) + omega) / ((n + 1) * gamma)
    sigma = (1 + (lmb - 1 - 0.5*(gamma-1)*omega) / (gamma * (1 - U0))) / lmb
    xt, wt = x[-1], w[-1]
    k = wt * xt**sigma
    xb = (k / np.array(w)) ** (1/sigma)
    xfull = np.concatenate((x, np.flip(xb)))
    Vfull = np.concatenate((V, np.flip(Vr)))
    Cfull = np.concatenate((C, np.flip(Cr)))
    # Calculate G
    q = (2*(1-lmb) + (n+1)*(gamma-1)) / (omega - n - 1)
    Gfac = 2*gamma/(gamma+1) * ((gamma-1)/(gamma+1))**gamma
    G = (Gfac * (x/C)**2 * (1+V)**(1-q-gamma)) ** (1/q)
    Gr0 = G[-1] * (1 + V[-1]) / (1 + Vr[-1])
    Gr = Gr0 * ((xb/xb[-1])**2 * (Cr[-1]/Cr)**2 * ((1+Vr[-1])/(1+Vr))**(q+gamma-1)) ** (1/q)
    Gfull = np.concatenate((G, np.flip(Gr)))
    # Plot
    fig2 = plt.figure(2)
    plt.plot([-10, -1], [0, 0], 'b')
    plt.plot([-1, -1], [0, 2/(gamma+1)], 'b--')
    plt.plot(x, -V, 'b')
    plt.plot(xb, -Vr, 'b')
    plt.plot([x[-1], xb[-1]], [-V[-1], -Vr[-1]], 'b--')
    plt.grid(True)
    # plt.title(ttl)
    plt.xlabel('x')
    plt.ylabel('U')
    plt.xlim(-2, 2)
    plt.ylim(-0.2, 0.8)
    fig2.tight_layout()

    fig3 = plt.figure(3)
    plt.plot([-10, -1], [0, 0], 'r')
    plt.plot([-1, -1], [0, np.sqrt(2*gamma*(gamma-1))/(gamma+1)], 'r--')
    plt.plot(x, C, 'r')
    plt.plot(xb, Cr, 'r')
    plt.plot([x[-1], xb[-1]], [C[-1], Cr[-1]], 'r--')
    plt.grid(True)
    # plt.title(ttl)
    plt.xlabel('x')
    plt.ylabel('C')
    plt.xlim(-2, 2)
    plt.ylim(-1.5, 0.75)
    fig3.tight_layout()

    fig4 = plt.figure(4)
    plt.plot([-10, -1], [1, 1], 'm')
    plt.plot([-1, -1], [1, (gamma+1)/(gamma-1)], 'm--')
    plt.plot(x, G, 'm')
    plt.plot(xb, Gr, 'm')
    plt.plot([x[-1], xb[-1]], [G[-1], Gr[-1]], 'm--')
    plt.grid(True)
    # plt.title(ttl)
    plt.xlabel('x')
    plt.ylabel('G')
    plt.xlim(-2, 10)
    plt.ylim(0, 12)
    fig4.tight_layout()

    # Plot P(r)
    ind1, ind2 = np.where(x < 0), np.where(x > 0)
    x1, x2 = x[ind1], x[ind2]
    C1, C2 = C[ind1], C[ind2]
    G1, G2 = G[ind1], G[ind2]
    r1, r2, r3 = (-x1) ** (-1/lmb), x2 ** (-1/lmb), xb ** (-1/lmb)
    p1 = r1**(2-omega) * G1*C1**2 / (gamma*lmb**2)
    p2 = r2**(2-omega) * G2*C2**2 / (gamma*lmb**2)
    p3 = r3**(2-omega) * Gr*Cr**2 / (gamma*lmb**2)

    fig5 = plt.figure(5)
    plt.loglog(r1, p1, 'b', label='t=-1')
    plt.plot([0, 1], [0, 0], 'b')
    plt.plot([1, 1], [0, p1[0]], 'b--')
    plt.plot(r2, p2, 'r', label='t=1')
    plt.plot([r2[-1], r2[-1]], [p2[-1], p3[-1]], 'r--')
    plt.plot(r3, p3, 'r')
    plt.grid(True)
    plt.legend()
    # plt.title(ttl)
    plt.xlabel('r')
    plt.ylabel('p')
    plt.xlim([1e-1, 1e2])
    plt.ylim([1e-1, 1e1])
    fig5.tight_layout()

    plt.show()

if save_results:
    name = 'n' + str(n) + '_om' + str(omega)
    drct = 'semi_analytical_results/' + name + '/'
    mkdir(drct)
    f = ''
    for i in range(len(x)):
        f += str(x[i]) + ', ' + str(-V[i]) + ', ' + str(C[i]) + '\n'
    g = ''
    for i in range(len(w)):
        g += str(w[i]) + ', ' + str(-Vr[i]) + ', ' + str(Cr[i]) + '\n'
    open(drct + 'p1.txt', 'w').write(f)
    open(drct + 'p2.txt', 'w').write(g)
