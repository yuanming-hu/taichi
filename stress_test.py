import taichi as ti

n = 4
ti.init(arch=ti.cuda, print_kernel_llvm_ir=True, print_kernel_llvm_ir_optimized=True, print_ir=True)
x = ti.field(ti.i32)
l = ti.field(ti.i32, shape=n)

ti.root.dense(ti.i, n).dynamic(ti.j, n, 8).place(x)

@ti.kernel
def func():
  for i in range(n):  # 1
    for j in range(4):
      v = j * 100 + i
      # print('value', v)
      ti.append(x.parent(), j, v)

  for i in range(n):  # 2
    l[i] = ti.length(x.parent(), i)

func()

for i in range(n):
    print(i, end=': ')
    for j in range(l[i]):
        print(x[i, j], end=' ')
    print()

