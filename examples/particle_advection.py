import taichi as ti
import random
ti.init(arch=ti.gpu)

dim = 2
n_particles = 8192
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 2.0e-4
p_vol = (dx * 0.5)**2
p_rho = 1
p_mass = p_vol * p_rho
E = 400

x = ti.Vector(dim, dt=ti.f32, shape=n_particles)
v = ti.Vector(dim, dt=ti.f32, shape=n_particles)
C = ti.Matrix(dim, dim, dt=ti.f32, shape=n_particles)
grid_v = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.var(dt=ti.f32, shape=(n_grid, n_grid))

@ti.func
def sample_v(pos):
    base = (pos * inv_dx - 0.5).cast(int)
    fx = pos * inv_dx - base.cast(float)
    w = [
        0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2
    ]
    new_v = ti.Vector.zero(ti.f32, 2)
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            
    return new_v

@ti.kernel
def substep():
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        affine = p_mass * C[p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v[base + offset].atomic_add(
                    weight * (p_mass * v[p] + affine @ dpos))
                grid_m[base + offset].atomic_add(weight * p_mass)
    
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            inv_m = 1 / grid_m[i, j]
            grid_v[i, j] = inv_m * grid_v[i, j]
    
    for p in x:
        # Move the particle (RK2)
        v_mid = sample_v(x[p] + v[p] * dt * 0.5)
        x[p] = x[p] + v_mid * dt
        
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [
            0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2
        ]
        new_v = ti.Vector.zero(ti.f32, 2)
        new_C = ti.Matrix.zero(ti.f32, 2, 2)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.Vector([i, j]).cast(float) - fx
                g_v = grid_v[base + ti.Vector([i, j])]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        
        v[p] = new_v
        C[p] = new_C

for i in range(n_particles):
    x[i] = [random.random() * 0.6 + 0.2, random.random() * 0.6 + 0.2]
    v[i] = [x[i][1] - 0.5, 0.5 - x[i][0]]

gui = ti.GUI("MPM88", (512, 512))
for frame in range(20000):
    for s in range(50):
        grid_v.fill([0, 0])
        grid_m.fill(0)
        substep()
    gui.clear(0x112F41)
    gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    gui.show()
