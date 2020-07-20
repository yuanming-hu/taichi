import taichi as ti
import random
ti.init(arch=ti.gpu)

dim = 2
n_particles = 8192
n_grid = 32
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 2.0e-3
use_apic = False


x = ti.Vector(dim, dt=ti.f32, shape=n_particles)
v = ti.Vector(dim, dt=ti.f32, shape=n_particles)
C = ti.Matrix(dim, dim, dt=ti.f32, shape=n_particles)
grid_v = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.var(dt=ti.f32, shape=(n_grid, n_grid))

@ti.func
def clamp_pos(pos):
    return ti.Vector([max(min(0.95, pos[0]), 0.05), max(min(0.95, pos[1]), 0.05)])

@ti.kernel
def substep_PIC():
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic B-spline
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                weight = w[i][0] * w[j][1]
                grid_v[base + offset] += weight * v[p]
                grid_m[base + offset] += weight
    
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            inv_m = 1 / grid_m[i, j]
            grid_v[i, j] = inv_m * grid_v[i, j]
    
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic B-spline
        w = [
            0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2
        ]
        new_v = ti.Vector.zero(ti.f32, 2)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                weight = w[i][0] * w[j][1]
                new_v += weight * grid_v[base + ti.Vector([i, j])]

        x[p] = clamp_pos(x[p] + v[p] * dt)
        v[p] = new_v

@ti.kernel
def substep_APIC():
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic B-spline
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        affine = C[p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v[base + offset] += weight * (v[p] + affine @ dpos)
                grid_m[base + offset] += weight
    
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            inv_m = 1 / grid_m[i, j]
            grid_v[i, j] = inv_m * grid_v[i, j]
    
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic B-spline
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

        x[p] = clamp_pos(x[p] + new_v * dt)
        v[p] = new_v
        C[p] = new_C

@ti.kernel
def reset(mode: ti.i32):
    for i in range(n_particles):
        x[i] = [ti.random() * 0.6 + 0.2, ti.random() * 0.6 + 0.2]
        if mode == 0:
            v[i] = [1, 0]
        elif mode == 1:
            v[i] = [x[i][1] - 0.5, 0.5 - x[i][0]]
        elif mode == 2:
            v[i] = [0, x[i][0] - 0.5]
        else:
            v[i] = [0, x[i][1] - 0.5]
        
reset(1)

gui = ti.GUI("PIC v.s. APIC", (512, 512))
for frame in range(2000000):
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == 't': reset(0)
        elif gui.event.key == 'r': reset(1)
        elif gui.event.key == 's': reset(2)
        elif gui.event.key == 'd': reset(3)
        elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]: break
        elif gui.event.key == 'a': use_apic = not use_apic
    for s in range(10):
        grid_v.fill([0, 0])
        grid_m.fill(0)
        if use_apic:
            substep_APIC()
        else:
            substep_PIC()
    scheme = 'APIC' if use_apic else 'PIC'
    gui.clear(0x112F41)
    gui.circles(x.to_numpy(), radius=2, color=0x068587)
    gui.text('(D) Reset as dilation', pos=(0.05, 0.25))
    gui.text('(T) Reset as translation', pos=(0.05, 0.2))
    gui.text('(R) Reset as rotation', pos=(0.05, 0.15))
    gui.text('(S) Reset as shearing', pos=(0.05, 0.1))
    gui.text(f'(A) Scheme={scheme}', pos=(0.05, 0.05))
    gui.show()
