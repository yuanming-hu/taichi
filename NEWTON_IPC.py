import sys, os, time
import taichi as ti
import time
import numpy as np
from distance import *

ti.init(arch=ti.cpu, print_ir=True)

real = ti.f32
scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(dim, dt=real)
mat = lambda: ti.Matrix(dim, dim, dt=real)

dim = 3
dt = 0.01
E = 1e4
nu = 0.4
la = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
density = 100
n_particles = 1000
n_elements = 1000
n_boundary_points = 1000
n_boundary_edges = 1000
n_boundary_triangles = 1000

x, x0, xPrev, xTilde, xn, v, m = vec(), vec(), vec(), vec(), vec(), vec(), scalar()
zero = vec()
restT = mat()
vertices = ti.var(ti.i32)
boundary_points = ti.var(ti.i32)
boundary_edges = ti.var(ti.i32)
boundary_triangles = ti.var(ti.i32)
ti.root.dense(ti.k, n_particles).place(x, x0, xPrev, xTilde, xn, v, m)
ti.root.dense(ti.k, n_particles).place(zero)
ti.root.dense(ti.i, n_elements).place(restT)
ti.root.dense(ti.ij, (n_elements, dim + 1)).place(vertices)
ti.root.dense(ti.i, n_boundary_points).place(boundary_points)
ti.root.dense(ti.ij, (n_boundary_edges, 2)).place(boundary_edges)
ti.root.dense(ti.ij, (n_boundary_triangles, 3)).place(boundary_triangles)

data_rhs = ti.var(real, shape=20000)
data_mat = ti.var(real, shape=(3, 2000000))
data_sol = ti.var(real, shape=20000)
cnt = ti.var(dt=ti.i32, shape=())

PP = ti.var(ti.i32, shape=(100000, 2))
n_PP = ti.var(dt=ti.i32, shape=())
PE = ti.var(ti.i32, shape=(100000, 3))
n_PE = ti.var(dt=ti.i32, shape=())
PT = ti.var(ti.i32, shape=(100000, 4))
n_PT = ti.var(dt=ti.i32, shape=())
EE = ti.var(ti.i32, shape=(100000, 4))
n_EE = ti.var(dt=ti.i32, shape=())
EEM = ti.var(ti.i32, shape=(100000, 4))
n_EEM = ti.var(dt=ti.i32, shape=())
PPM = ti.var(ti.i32, shape=(100000, 4))
n_PPM = ti.var(dt=ti.i32, shape=())
PEM = ti.var(ti.i32, shape=(100000, 4))
n_PEM = ti.var(dt=ti.i32, shape=())

dHat2 = 1e-5
dHat = dHat2 ** 0.5
kappa = 1e4


@ti.func
def fill_vec6(v):
    idx = ti.static([0, 1, 2, 6, 7, 8])
    vec = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in ti.static(range(6)):
        vec[idx[i]] = v[i]
    return vec


@ti.func
def fill_vec9(v):
    idx = ti.static([0, 1, 2, 6, 7, 8, 9, 10, 11])
    vec = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in ti.static(range(9)):
        vec[idx[i]] = v[i]
    return vec


@ti.func
def fill_mat6(m):
    idx = ti.static([0, 1, 2, 6, 7, 8])
    Z = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mat = ti.Matrix.rows([Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z])
    for i in ti.static(range(6)):
        for j in ti.static(range(6)):
            mat[idx[i], idx[j]] = m[i, j]
    return mat


@ti.func
def fill_mat9(m):
    idx = ti.static([0, 1, 2, 6, 7, 8, 9, 10, 11])
    Z = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mat = ti.Matrix.rows([Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z])
    for i in ti.static(range(9)):
        for j in ti.static(range(9)):
            mat[idx[i], idx[j]] = m[i, j]
    return mat


@ti.func
def load_hessian_and_gradient2(H, g, idx, c):
    for i in ti.static(range(2)):
        for d in ti.static(range(dim)):
            for j in ti.static(range(2)):
                for e in ti.static(range(dim)):
                    data_mat[0, c], data_mat[1, c], data_mat[2, c] = idx[i] * dim + d, idx[j] * dim + e, H[i, j]
                    c += 1
    for i in ti.static(range(2)):
        for d in ti.static(range(dim)):
            data_rhs[idx[i] * dim + d] -= g[i]


@ti.func
def load_hessian_and_gradient3(H, g, idx, c):
    for i in ti.static(range(3)):
        for d in ti.static(range(dim)):
            for j in ti.static(range(3)):
                for e in ti.static(range(dim)):
                    data_mat[0, c], data_mat[1, c], data_mat[2, c] = idx[i] * dim + d, idx[j] * dim + e, H[i, j]
                    c += 1
    for i in ti.static(range(3)):
        for d in ti.static(range(dim)):
            data_rhs[idx[i] * dim + d] -= g[i]


@ti.func
def load_hessian_and_gradient4(H, g, idx, c):
    for i in ti.static(range(4)):
        for d in ti.static(range(dim)):
            for j in ti.static(range(4)):
                for e in ti.static(range(dim)):
                    data_mat[0, c], data_mat[1, c], data_mat[2, c] = idx[i] * dim + d, idx[j] * dim + e, H[i, j]
                    c += 1
    for i in ti.static(range(4)):
        for d in ti.static(range(dim)):
            data_rhs[idx[i] * dim + d] -= g[i]


@ti.kernel
def compute_hessian_and_gradient():
    '''
    cnt[None] = 0
    # ipc
    for r in range(n_PP[None]):
        p0, p1 = x[PP[r, 0]], x[PP[r, 1]]
        dist2 = PP_3D_E(p0, p1)
        dist2g = PP_3D_g(p0, p1)
        bg = barrier_g(dist2, dHat2, kappa)
        g = bg * dist2g
        H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PP_3D_H(p0, p1)
        load_hessian_and_gradient2(H, g, ti.Vector([PP[r, 0], PP[r, 1]]), cnt[None] + r * 36)
    cnt[None] += n_PP[None] * 36
    for r in range(n_PE[None]):
        p, e0, e1 = x[PE[r, 0]], x[PE[r, 1]], x[PE[r, 2]]
        dist2 = PE_3D_E(p, e0, e1)
        dist2g = PE_3D_g(p, e0, e1)
        bg = barrier_g(dist2, dHat2, kappa)
        g = bg * dist2g
        H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PE_3D_E(p, e0, e1)
        load_hessian_and_gradient3(H, g, ti.Vector([PE[r, 0], PE[r, 1], PE[r, 2]]), cnt[None] + r * 81)
    cnt[None] += n_PE[None] * 81
    for r in range(n_PT[None]):
        p, t0, t1, t2 = x[PT[r, 0]], x[PT[r, 1]], x[PT[r, 2]], x[PT[r, 3]]
        dist2 = PT_3D_E(p, t0, t1, t2)
        dist2g = PT_3D_g(p, t0, t1, t2)
        bg = barrier_g(dist2, dHat2, kappa)
        g = bg * dist2g
        H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PT_3D_H(p, t0, t1, t2)
        load_hessian_and_gradient4(H, g, ti.Vector([PT[r, 0], PT[r, 1], PT[r, 2], PT[r, 3]]), cnt[None] + r * 144)
    cnt[None] += n_PT[None] * 144
    for r in range(n_EE[None]):
        a0, a1, b0, b1 = x[EE[r, 0]], x[EE[r, 1]], x[EE[r, 2]], x[EE[r, 3]]
        dist2 = EE_3D_E(a0, a1, b0, b1)
        dist2g = EE_3D_g(a0, a1, b0, b1)
        bg = barrier_g(dist2, dHat2, kappa)
        g = bg * dist2g
        H = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * EE_3D_H(a0, a1, b0, b1)
        load_hessian_and_gradient4(H, g, ti.Vector([EE[r, 0], EE[r, 1], EE[r, 2], EE[r, 3]]), cnt[None] + r * 144)
    cnt[None] += n_EE[None] * 144
    for r in range(n_EEM[None]):
        a0, a1, b0, b1 = x[EEM[r, 0]], x[EEM[r, 1]], x[EEM[r, 2]], x[EEM[r, 3]]
        _a0, _a1, _b0, _b1 = x0[EEM[r, 0]], x0[EEM[r, 1]], x0[EEM[r, 2]], x0[EEM[r, 3]]
        eps_x = M_threshold(_a0, _a1, _b0, _b1)
        dist2 = EE_3D_E(a0, a1, b0, b1)
        dist2g = EE_3D_g(a0, a1, b0, b1)
        b = barrier_E(dist2, dHat2, kappa)
        bg = barrier_g(dist2, dHat2, kappa)
        lg = bg * dist2g
        lH = barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * EE_3D_H(a0, a1, b0, b1)
        M = M_E(a0, a1, b0, b1, eps_x)
        Mg = M_g(a0, a1, b0, b1, eps_x)
        g = lg * M + b * Mg
        H = lH * M + 2 * lg.outer_product(Mg) + b * M_H(a0, a1, b0, b1, eps_x)
        load_hessian_and_gradient4(H, g, ti.Vector([EEM[r, 0], EEM[r, 1], EEM[r, 2], EEM[r, 3]]), cnt[None] + r * 144)
    cnt[None] += n_EEM[None] * 144
    for r in range(n_PPM[None]):
        a0, a1, b0, b1 = x[PPM[r, 0]], x[PPM[r, 1]], x[PPM[r, 2]], x[PPM[r, 3]]
        _a0, _a1, _b0, _b1 = x0[PPM[r, 0]], x0[PPM[r, 1]], x0[PPM[r, 2]], x0[PPM[r, 3]]
        eps_x = M_threshold(_a0, _a1, _b0, _b1)
        dist2 = PP_3D_E(a0, b0)
        dist2g = PP_3D_g(a0, b0)
        b = barrier_E(dist2, dHat2, kappa)
        bg = barrier_g(dist2, dHat2, kappa)
        lg = fill_vec6(bg * dist2g)
        lH = fill_mat6(barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PP_3D_H(a0, b0))
        M = M_E(a0, a1, b0, b1, eps_x)
        Mg = M_g(a0, a1, b0, b1, eps_x)
        g = lg * M + b * Mg
        H = lH * M + 2 * lg.outer_product(Mg) + b * M_H(a0, a1, b0, b1, eps_x)
        load_hessian_and_gradient4(H, g, ti.Vector([PPM[r, 0], PPM[r, 1], PPM[r, 2], PPM[r, 3]]), cnt[None] + r * 144)
    cnt[None] += n_PPM[None] * 144
    '''
    for r in range(n_PEM[None]):
        a0, a1, b0, b1 = x[PEM[r, 0]], x[PEM[r, 1]], x[PEM[r, 2]], x[PEM[r, 3]]
        _a0, _a1, _b0, _b1 = x0[PEM[r, 0]], x0[PEM[r, 1]], x0[PEM[r, 2]], x0[PEM[r, 3]]
        eps_x = M_threshold(_a0, _a1, _b0, _b1)
        dist2 = PE_3D_E(a0, b0, b1)
        dist2g = PE_3D_g(a0, b0, b1)
        b = barrier_E(dist2, dHat2, kappa)
        bg = barrier_g(dist2, dHat2, kappa)
        lg = fill_vec9(bg * dist2g)
        PE3D = PE_3D_H(a0, b0, b1)
        lH = fill_mat9(barrier_H(dist2, dHat2, kappa) * dist2g.outer_product(dist2g) + bg * PE3D)
        M = M_E(a0, a1, b0, b1, eps_x)
        Mg = M_g(a0, a1, b0, b1, eps_x)
        g = lg * M + b * Mg
        # print(g) # 2428
        H = lH * M + 2 * lg.outer_product(Mg) + b * M_H(a0, a1, b0, b1, eps_x)
        # print(g, H) # 4938
        '''
        load_hessian_and_gradient4(H, g, ti.Vector([PEM[r, 0], PEM[r, 1], PEM[r, 2], PEM[r, 3]]), cnt[None] + r * 144)
        '''
    cnt[None] += n_PEM[None] * 144


if __name__ == "__main__":
    t = time.time()
    compute_hessian_and_gradient()
    print('Compilation time', time.time() - t)
    # ti.print_profile_info()
