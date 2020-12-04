import taichi as ti
import math
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.cuda)

nS = 300
nW = 500

T = 24
Sbar = 1.0
rho = 0.99
sigma = 0.01
lamb = 0.01

W_scale = 10
W_offset = -W_scale / 2 + 0.5

J = ti.field(dtype=ti.f32, shape=(T + 1, nS + 1, nW + 1))
opt = ti.field(dtype=ti.f32, shape=(T + 1, nS + 1, nW + 1))

p = ti.field(dtype=ti.f32, shape=(nS + 1, nS + 1))


@ti.kernel
def compute_p():  # transition probability of S
    for i in range(nS + 1):
        tot = 0.0
        for j in range(nS + 1):
            Sprev = i / nS * 2
            Snext = j / nS * 2
            Snext_AR = (Sprev - Sbar) * rho + Sbar
            d = Snext - Snext_AR
            pdf = 1 / (sigma * ti.sqrt(2 * math.pi)) * ti.exp(-0.5 *
                                                              (d / sigma)**2)
            p[i, j] = pdf
            tot += pdf

        for j in range(nS + 1):  # normalize
            p[i, j] /= tot


@ti.kernel
def compute_Jt(t: ti.i32):
    for i, j in ti.ndrange(nS + 1, nW + 1):
        J[t, i, j] = -1e30
        S = i / nS * 2
        for k in range(nW + 1):
            x = (j - k) / nW * W_scale
            E = 0.0  # expectation
            for l in range(nS + 1):
                val = x * (rho * (S - Sbar) + Sbar - lamb * x) + J[t + 1, l, k]
                E += p[i, l] * val
            if E > J[t, i, j]:
                opt[t, i, j] = x
                J[t, i, j] = E


@ti.kernel
def compute_JT():
    for i, j in ti.ndrange(nS + 1, nW + 1):
        S = i / nS * 2
        x = j / nW * W_scale + W_offset
        val = x * (rho * (S - Sbar) + Sbar - lamb * x)
        J[T, i, j] = val


opt.fill(-1)
compute_p()

compute_JT()
for t in reversed(range(T)):
    compute_Jt(t)


def plot_J(t):
    colors = 'rgbcym'
    for i, c in zip([nS // 6, nS // 3, nS // 2, 2 * nS // 3, nS * 5 // 6, nS],
                    range(6)):
        S = i / nS * 2
        analytical = []
        Ws = []
        dp = []
        for j in range(0, nW, 1):
            W = j / nW * W_scale + W_offset
            Ws.append(W)
            if t == T - 1:
                J_ana = -lamb * W * W / 2 + (
                    (rho * (1 + rho) / 2) * (S - Sbar) + Sbar) * W + rho**2 * (
                        1 - rho)**2 * (S - Sbar)**2 / (8 * lamb)
            elif t == T - 2:
                D = (S - Sbar)
                J_ana = -lamb * W * W / 6 + 1 / 3 * (
                    3 * Sbar + rho * D *
                    (1 + rho *
                     (1 + rho))) * W + rho**2 * (1 - rho)**2 * D**2 * (
                         rho**4 - rho**2 - 2 * rho + 2) / (4 * lamb)
            else:
                raise ValueError()

            analytical.append(J_ana)
            dp.append(J[t, i, j])
        plt.plot(Ws, dp, colors[c] + '.', label=f'DP  S={S:.3f}')
        plt.plot(Ws, analytical, colors[c] + '-', label=f'ANA S={S:.3f}')
    print(dp)
    print(analytical)
    plt.title('DP v.s. analytical')
    plt.xlabel('x')
    plt.ylabel('J')
    plt.legend()
    plt.show()


def plot_policy(t):
    colors = 'rgbcym'
    for W, c in zip([0.0, 0.4, 0.8, 1.2, 1.6, -0.4], range(6)):
        analytical = []
        Ss = []
        dp = []
        for i in range(nS // 4, 3 * nS // 4, 1):
            S = i / nS * 2
            Ss.append(S)

            D = (S - Sbar)
            if t == 22:
                x_ana = W / 3 + (rho * D * (2 - rho * (1 + rho))) / (6 * lamb)
                # x_ana = W / 2 + ((rho * (1-rho) * D)) / (4 * lamb)
                analytical.append(x_ana)
            dp.append(opt[t, i, int((W - W_offset) / W_scale * nW)])
        if t == 22:
            plt.plot(Ss, analytical, colors[c] + '-', label=f'ANA W={W:.3f}')
        f = '.' if t == 22 else '-'
        plt.plot(Ss, dp, colors[c] + f, label=f'DP  W={W:.3f}')
    if t == 22:
        plt.title('DP v.s. analytical optimal policy')
    else:
        plt.title('Optimal policy at t = T/2')
    plt.ylabel('x^*')
    plt.xlabel('S')
    plt.legend()
    plt.show()


# plot_a(t=23)
# plot_policy(t=T - 2)
# plot_policy(t=T // 2)


def plot_policy_w():
    Ws = []
    dp = []
    for i in range(0, nW, nW // 30):
        W = i / nW * W_scale + W_offset
        Ws.append(W)
        dp.append(opt[T // 2, nS // 2, i])

    plt.plot(Ws, dp)
    plt.ylabel('x^*')
    plt.xlabel('W')
    plt.legend()
    plt.show()


# plot_policy_w()


def draw():
    S = 1
    W_dp = 1
    W_uniform = 1
    P_dp = 0
    P_uniform = 0

    for t in range(1, T + 1):
        S = Sbar + (S - Sbar) * rho + np.random.normal() * sigma
        i = min(max(0, int(S / 2 * nS + 0.5)), nS)
        j = max(min(int((W_dp - W_offset) / W_scale * nW + 0.5), nW), 0)
        if t == T:
            x = W_dp
        else:
            x = opt[t, i, j]
        W_dp -= x
        W_uniform -= 1 / T
        P_dp += x * (S - lamb * x)
        P_uniform += 1 / T * (S - lamb * 1 / T)

    print(P_dp, P_uniform)
    return P_dp - P_uniform


def simulate():
    diff = 0
    for i in range(10000):
        diff += draw()
    print(diff / 10000)


simulate()
# print(J[0, nS // 2, int((1 - W_offset) / W_scale * nW)])
