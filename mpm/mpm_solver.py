import taichi as ti

ti.init(arch=ti.cpu, async_mode=False, debug=True)
grid_size = 1024

indices = ti.ij

m = ti.var(dt=ti.f32)

grid_block_size = 128
grid1 = ti.root.pointer(indices, grid_size // grid_block_size)

use2 = True
if use2:
    grid2 = ti.root.dense(indices, grid_size // grid_block_size)

leaf_block_size = 16

block = grid1.pointer(indices,
                          grid_block_size // leaf_block_size).place(m)

if use2:
    m2 = ti.var(dt=ti.f32)
    block2 = grid2.dense(indices,
                              grid_block_size // leaf_block_size).place(m2)


grid1.deactivate_all()
ti.sync()
print('successfully finishes')