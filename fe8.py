import taichi as ti
import math

ti.init(arch=ti.cuda)

nS = 300
nW = 300

T = 24
Sbar = 1.0
rho = 0.99
sigma = 0.01
lamb = 0.01

J = ti.field(dtype=ti.f32, shape=(T + 1, nW + 1, nS + 1))

p = ti.field(dtype=ti.f32, shape=(nS + 1, nS + 1))

@ti.kernel
def compute_p(): # transition probability of S
    for i in range(nS + 1):
        tot = 0.0
        for j in range(nS + 1):
            Sprev = i / nS
            Snext = j / nS
            Snext_AR = (Sprev - Sbar) * rho + Sbar
            d = Snext - Snext_AR
            pdf = 1 / (sigma * ti.sqrt(2 * math.pi)) * ti.exp(-0.5 * (d / sigma) ** 2)
            p[i, j] = pdf
            tot += pdf

        for j in range(nS + 1): # normalize
            p[i, j] /= tot

@ti.kernel
def compute_Jt(t: ti.i32):
    for i, j in ti.ndrange(nS + 1, nW + 1):
        J[t, i, j] = -1e30
        S = i / nS
        for k in range(j + 1):
            x = k / nW
            E = 0.0 # expectation
            for l in range(nS + 1):
                val = x * (rho * (S - Sbar) + Sbar - lamb * x) + J[t + 1, l, j - k]
                E += p[j, l] * val
            J[t, i, j] = max(J[t, i, j], E)

@ti.kernel
def compute_JT():
    for i, j in ti.ndrange(nS + 1, nW + 1):
        S = i / nS
        x = j / nW
        val = x * (rho * (S - Sbar) + Sbar - lamb * x)
        J[T, i, j] = val
        
compute_p()

compute_JT()
for t in reversed(range(T)):
    compute_Jt(t)

print(J.to_numpy())
