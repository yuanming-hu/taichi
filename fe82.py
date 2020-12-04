import taichi as ti
import math
import matplotlib.pyplot as plt

ti.init(arch=ti.cuda, debug=True)

nS = 100
nW = 900

T = 24
Sbar = 1.0
rho = 0.99
sigma = 0.01
lamb = 0.01



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
            x = (j - k) / nW * 5
            E = 0.0  # expectation
            for l in range(nS + 1):
                val = x * (rho *
                           (S - Sbar) + Sbar - lamb * x) + J[t + 1, l, k]
                E += p[i, l] * val
            if E > J[t, i, j]:
                opt[t, i, j] = x
                J[t, i, j] = E


@ti.kernel
def compute_JT():
    for i, j in ti.ndrange(nS + 1, nW + 1):
        S = i / nS * 2
        x = j / nW * 5 - 2
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
            W = j / nW * 5 - 2
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
    
def plot_policy():
    colors = 'rgbcym'
    for j, c in zip([nW // 6, nW // 3, nW // 2, 2 * nW // 3, nW * 5 // 6, nW],
                    range(6)):
        W = j / nW * 5 - 2
        analytical = []
        Ss = []
        dp = []
        for i in range(0, nS, 1):
            S = i / nS * 2
            Ss.append(S)
            
            D = (S - Sbar)
            x_ana = W / 3 + (rho * D * (2 - rho * (1+rho))) / (6 * lamb)
            # x_ana = W / 2 + ((rho * (1-rho) * D)) / (4 * lamb)
            
            analytical.append(x_ana)
            dp.append(opt[22, i, j])
        plt.plot(Ss, dp, colors[c] + '.', label=f'DP  W={W:.3f}')
        plt.plot(Ss, analytical, colors[c] + '-', label=f'ANA W={W:.3f}')
    plt.title('DP v.s. analytical optimal policy')
    plt.ylabel('x^*')
    plt.xlabel('S')
    plt.legend()
    plt.show()


# plot_a(t=23)
plot_policy()
