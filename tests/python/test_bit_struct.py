import taichi as ti
import numpy as np


def test_a():
    ti.init(arch=ti.cpu, debug=True, print_ir=True, cfg_optimization=False)
    ci13 = ti.type_factory_.get_custom_int_type(14, True)
    return


def test_b():
    ti.init(arch=ti.cpu, debug=True, print_ir=True, cfg_optimization=False)
    ci13 = ti.type_factory_.get_custom_int_type(14, True)
    return
