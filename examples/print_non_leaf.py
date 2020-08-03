import taichi as ti
ti.init()

a = ti.field(dtype=ti.i32)

blk = ti.root.dense(ti.ij, 2)
blk.dense(ti.ij, (2, 4)).place(a)

@ti.kernel
def task():
    for i, j in blk:
        print(i, j)

task()
