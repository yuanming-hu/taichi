import taichi as ti

ti.init(arch=ti.cpu)
x, y = ti.field(ti.f32), ti.field(ti.f32)

n = 16

ti.root.dense(ti.ij, n).place(x, y)


@ti.kernel
def laplace():
    for i, j in x:
        if 0 < i < n - 1 and 0 < j < n - 1:
            if (i + j) % 3 == 0:
                y[i,
                  j] = 4.0 * x[i, j] - x[i - 1,
                                         j] - x[i + 1,
                                                j] - x[i, j - 1] - x[i, j + 1]
            else:
                y[i, j] = 0.0


for i in range(10):
    x[i, i + 1] = 1.0

laplace()
ti.get_runtime().print_snode_tree()

for i in range(10):
    print(y[i, i + 1])
