import taichi as ti
import numpy as np


def test_simple_array():
    ti.init(arch=ti.cpu, debug=True, print_ir=True, cfg_optimization=False)
    ci13 = ti.type_factory_.get_custom_int_type(13, True)
    cu19 = ti.type_factory_.get_custom_int_type(19, False)

    return

def test_custom_int_load_and_store():
    ti.init(arch=ti.cpu, debug=True, print_ir=True, cfg_optimization=False)
    ci13 = ti.type_factory_.get_custom_int_type(13, True)
    cu14 = ti.type_factory_.get_custom_int_type(14, False)
    ci5 = ti.type_factory_.get_custom_int_type(5, True)

    return
