import sys
import os
import numpy as np
import pytest

sys.path.append('..')
import autodiff.optimize as optimize


def test_find_minimum():
    def test_func(args):
        return 2 * (args[0] ** 2) + 3 * (args[1] ** 2)

    assert(np.isclose(optimize.gradient_descent(test_func, [2, 100])[0], [0, 0]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func, [2, 100], method = 'BFGS')[0], [0, 0]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func, [2, 100], method = 'DFP')[0], [0, 0]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func, [2, 100], method = 'Broyden')[0], [0, 0]).all())

    assert(np.isclose(optimize.gradient_descent(test_func, [100, 2])[0], [0, 0]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func, [100, 2], method = 'BFGS')[0], [0, 0]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func, [100, 2], method = 'DFP')[0], [0, 0]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func, [100, 2], method = 'Broyden')[0], [0, 0]).all())

    assert(np.isclose(optimize.gradient_descent(test_func, [-100, 100])[0], [0, 0]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func, [-100, 100], method = 'BFGS')[0], [0, 0]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func, [-100, 100], method = 'DFP')[0], [0, 0]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func, [-100, 100], method = 'Broyden')[0], [0, 0]).all())


    assert(np.isclose(optimize.gradient_descent(test_func, [0.4242, -0.54234])[0], [0, 0]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func, [0.4242, -0.54234], method = 'BFGS')[0], [0, 0]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func, [0.4242, -0.54234], method = 'DFP')[0], [0, 0]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func, [0.4242, -0.54234], method = 'Broyden')[0], [0, 0]).all())


    def test_func2(args):
        return (args[0] - 5) ** 2

    assert(np.isclose(optimize.gradient_descent(test_func2, [1])[0], [5]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func2, [1], method = 'BFGS')[0], [5]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func2, [1], method = 'DFP')[0], [5]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func2, [1], method = 'Broyden')[0], [5]).all())

    assert(np.isclose(optimize.gradient_descent(test_func2, [-0.534])[0], [5]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func2, [-0.534], method = 'BFGS')[0], [5]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func2, [-0.534], method = 'DFP')[0], [5]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func2, [-0.534], method = 'Broyden')[0], [5]).all())

    assert(np.isclose(optimize.gradient_descent(test_func2, [20])[0], [5]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func2, [20], method = 'BFGS')[0], [5]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func2, [20], method = 'DFP')[0], [5]).all())
    assert(np.isclose(optimize.quasi_newtons_method(test_func2, [20], method = 'Broyden')[0], [5]).all())


    def rosenbrock(args, a = 2, b = 3):
        return (a - args[0]) ** 2 + b * (args[1] - args[0] ** 2) ** 2

    assert(np.isclose(optimize.gradient_descent(rosenbrock, [1, 1])[0], [2, 4]).all())
    assert(np.isclose(optimize.quasi_newtons_method(rosenbrock, [1, 1], method = 'BFGS')[0], [2, 4]).all())
    assert(np.isclose(optimize.quasi_newtons_method(rosenbrock, [1, 1], method = 'DFP')[0], [2, 4]).all())
    assert(np.isclose(optimize.quasi_newtons_method(rosenbrock, [1, 1], method = 'Broyden')[0], [2, 4]).all())


    assert(np.isclose(optimize.gradient_descent(rosenbrock, [2, 3])[0], [2, 4]).all())
    assert(np.isclose(optimize.quasi_newtons_method(rosenbrock, [2, 3], method = 'BFGS')[0], [2, 4]).all())
    assert(np.isclose(optimize.quasi_newtons_method(rosenbrock, [2, 3], method = 'DFP')[0], [2, 4]).all())
    assert(np.isclose(optimize.quasi_newtons_method(rosenbrock, [2, 3], method = 'Broyden')[0], [2, 4]).all())


    def test_func3(x):
        x = x[0]
        return np.array([x**3 -1])

    assert (np.isclose(optimize.newtons_method(test_func3,[5], method="inverse", max_iter= 1000)[0][0], 1.0))
    assert (np.isclose(optimize.newtons_method(test_func3,[5], method="exact", max_iter= 1000)[0][0], 1.0))
    assert (np.isclose(optimize.newtons_method(test_func3,[5], method="gmres", max_iter= 1000)[0][0], 1.0))
    assert (np.isclose(optimize.newtons_method(test_func3,[5], method="gmres_action", max_iter= 1000)[0][0], 1.0))

    def test_func4(x):
        x1 = x[0]
        x2 = x[1]
        return np.array([ 4 + 8*(x1**2) - (x1**4), 4 + 8*(x2**2) - (x2**4)])

    assert (np.isclose(optimize.newtons_method(test_func4, [-3,3], method="inverse", max_iter= 1000)[0] , [-2.91069338,  2.91069338]))[0]
    assert (np.isclose(optimize.newtons_method(test_func4, [-3,3], method="exact", max_iter= 1000)[0] , [-2.91069338,  2.91069338]))[0]
    assert (np.isclose(optimize.newtons_method(test_func4, [-3,3], method="gmres", max_iter= 1000)[0] , [-2.91069338,  2.91069338]))[0]
    assert (np.isclose(optimize.newtons_method(test_func4, [-3,3], method="gmres_action", max_iter= 1000)[0] , [-2.91069338,  2.91069338]))[0]

    def test_func5(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return np.array([x1 + x1 + 1, x2 ** 3 - 3 * x2 - 3, -3 * x3 + 1])

    assert (np.isclose(optimize.newtons_method(test_func5, [5, 4, 5], method="inverse", max_iter=1000)[0],
                       [-0.5, 2.1038034, 0.33333333]))[0]
    assert (np.isclose(optimize.newtons_method(test_func5, [5, 4, 5], method="exact", max_iter=1000)[0],
                       [-0.5, 2.1038034, 0.33333333]))[0]
    assert (np.isclose(optimize.newtons_method(test_func5, [5, 4, 5], method="gmres", max_iter=1000)[0],
                       [-0.5, 2.1038034, 0.33333333]))[0]
    assert (np.isclose(optimize.newtons_method(test_func5, [5, 4, 5], method="gmres_action", max_iter=1000)[0],
                       [-0.5, 2.1038034, 0.33333333]))[0]

    def test_func6(x):
        x1 = x[0]
        return np.array([(x1 ** 2) + 3])

    with pytest.raises(RuntimeError):
        optimize.newtons_method(test_func6, [5], method="gmres", max_iter= 1000)

    with pytest.raises(RuntimeError):
        optimize.newtons_method(test_func6, [5], method="gmres_action", max_iter=1000)

    with pytest.raises(Exception):
        optimize.newtons_method(test_func6, [5,5,5], method="gmres", max_iter= 1000)

    with pytest.raises(Exception):
        optimize.newtons_method(test_func6, [5,5,5], method="gkkk", max_iter= 1000)

    with pytest.raises(Exception):
        optimize.quasi_newtons_method(rosenbrock, [2, 3], method='gfdgh')