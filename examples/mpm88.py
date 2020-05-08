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
J = ti.var(dt=ti.f32, shape=n_particles)
grid_v = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.var(dt=ti.f32, shape=(n_grid, n_grid))

@ti.kernel
def substep():
  for p in x:
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1), 0.5 * ti.sqr(fx - 0.5)]
    J[p] *= 1 + dt * C[p].trace()
    stress = -dt * p_vol * (J[p] - 1) * 4 * inv_dx * inv_dx * E
    affine = ti.Matrix.identity(ti.f32, 2) * stress + p_mass * C[p]
    for i, j in ti.static(ti.ndrange(3, 3)):
        offset = ti.Vector([i, j])
        dpos = (offset.cast(float) - fx) * dx
        weight = w[i][0] * w[j][1]
        grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
        grid_m[base + offset] += weight * p_mass

  for i, j in grid_m:
    if grid_m[i, j] > 0:
      bound = 3
      grid_v[i, j] = grid_v[i, j] / grid_m[i, j]
      grid_v[i, j][1] -= dt * 9.8
      if i < bound and grid_v[i, j][0] < 0:
        grid_v[i, j][0] = 0
      if i > n_grid - bound and grid_v[i, j][0] > 0:
        grid_v[i, j][0] = 0
      if j < bound and grid_v[i, j][1] < 0:
        grid_v[i, j][1] = 0
      if j > n_grid - bound and grid_v[i, j][1] > 0:
        grid_v[i, j][1] = 0

  for p in x:
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    w = [
        0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1.0), 0.5 * ti.sqr(fx - 0.5)
    ]
    v[p] = ti.Vector.zero(ti.f32, 2)
    C[p] = ti.Matrix.zero(ti.f32, 2, 2)
    for i, j in ti.static(ti.ndrange(3, 3)):
      dpos = ti.Vector([i, j]).cast(float) - fx
      g_v = grid_v[base + ti.Vector([i, j])]
      weight = w[i][0] * w[j][1]
      v[p] += weight * g_v
      C[p] += 4 * weight * ti.outer_product(g_v, dpos) * inv_dx
    x[p] += dt * v[p]

gui = ti.GUI("MPM88", (512, 512))

for i in range(n_particles):
  x[i] = [random.random() * 0.4 + 0.2, random.random() * 0.4 + 0.2]
  v[i] = [0, -1]
  J[i] = 1

for frame in range(20000):
  for s in range(50):
    grid_v.fill([0, 0])
    grid_m.fill(0)
    substep()

  gui.clear(0x112F41)
  pos = x.to_numpy(as_vector=True)
  gui.circles(pos, radius=1.5, color=0x068587)
  gui.show()
