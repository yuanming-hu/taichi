import taichi as ti
import math

ti.init(arch=ti.cpu)

real = ti.f32
dim = 2
n_nodes_x = 50
n_nodes_y = 6
node_mass = 1
n_nodes = n_nodes_x * n_nodes_y
n_elements = (n_nodes_x - 1) * (n_nodes_y - 1) * 2
dt = 1e-3
dx = 1 / 32
p_mass = 1
p_vol = 1
E, nu = 1000, 0.3
la = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
element_V = 0.01

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(dim, dt=real)
mat = lambda: ti.Matrix(dim, dim, dt=real)

x = ti.Vector(dim, dt=real, shape=n_nodes, needs_grad=True)
v = ti.Vector(dim, dt=real, shape=n_nodes)
B = ti.Matrix(dim, dim, dt=real, shape=n_elements)
total_energy = ti.var(dt=real, shape=(), needs_grad=True)
vertices = ti.var(dt=ti.i32, shape=(n_elements, 3))


@ti.func
def compute_D(i):
    a = vertices[i, 0]
    b = vertices[i, 1]
    c = vertices[i, 2]
    return ti.Matrix.cols([x[b] - x[a], x[c] - x[a]])


@ti.kernel
def compute_B():
    for i in range(n_elements):
        B[i] = compute_D(i).inverse()


@ti.kernel
def compute_total_energy():
    for i in range(n_elements):
        D = compute_D(i)
        F = D @ B[i]
        # NeoHookean
        I1 = (F @ F.transpose()).trace()
        J = max(0.01, F.determinant())
        element_energy_density = 0.5 * mu * (
            I1 - 2) - mu * ti.log(J) + 0.5 * la * ti.log(J)**2
        total_energy[None] += element_energy_density * element_V

sphere = ti.Vector([0.5, 0.2])
sphere_radius = 0.1

@ti.kernel
def integrate():
    for p in x:
        # Collide with sphere
        offset = x[p] - sphere
        if offset.norm() < sphere_radius:
            n = offset.normalized()
            x[p] = sphere + sphere_radius * n
            v[p] = v[p] - v[p].dot(n) * n
        # Collide with ground
        if x[p][1] < 0.2:
            x[p][1] = 0.2
            v[p][1] = 0
        v[p] = (v[p] + (
            (-x.grad[p] / node_mass) + ti.Vector([0, -10])) * dt) * math.exp(
                dt * -3)
        x[p] += dt * v[p]


gui = ti.GUI("Linear tetrahedral FEM", (640, 640), background_color=0x112F41)

mesh = lambda i, j: i * n_nodes_y + j

for i in range(n_nodes_x):
    for j in range(n_nodes_y):
        t = mesh(i, j)
        x[t] = [0.1 + i * dx * 0.5, 0.7 + j * dx * 0.5 + i * dx * 0.1]
        v[t] = [0, -1]

# build mesh
for i in range(n_nodes_x - 1):
    for j in range(n_nodes_y - 1):
        # element id
        eid = (i * (n_nodes_y - 1) + j) * 2
        vertices[eid, 0] = mesh(i, j)
        vertices[eid, 1] = mesh(i + 1, j)
        vertices[eid, 2] = mesh(i, j + 1)

        eid = (i * (n_nodes_y - 1) + j) * 2 + 1
        vertices[eid, 0] = mesh(i, j + 1)
        vertices[eid, 1] = mesh(i + 1, j + 1)
        vertices[eid, 2] = mesh(i + 1, j)

compute_B()

vertices_ = vertices.to_numpy()

while True:
    for s in range(10):
        # Note that we are now differentiating the total energy w.r.t. the particle position.
        # Recall that F = - \partial (total_energy) / \partial x
        with ti.Tape(total_energy):
            compute_total_energy()
        integrate()

    gui.circle((0.5, 0.5), radius=45, color=0x068587)
    
    node_x = x.to_numpy()
    for i in range(n_elements):
        for j in range(3):
            a, b = vertices_[i, j], vertices_[i, (j + 1) % 3]
            gui.line((node_x[a][0], node_x[a][1]),
                     (node_x[b][0], node_x[b][1]),
                     radius=1,
                     color=0x4FB99F)
    gui.circles(node_x, radius=1.5, color=0x3241f4)
    gui.line((0.00, 0.2), (1.0, 0.2), color=0xFFFFFF, radius=3)
    gui.show()
