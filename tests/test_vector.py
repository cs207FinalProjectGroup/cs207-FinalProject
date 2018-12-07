import sys
import os
import numpy as np
import pytest

sys.path.append('..')
import autodiff as ad


def test_create_vector():
    v = ad.create_vector('v', [1, 2])
    assert(v[0].getValue() == 1)
    assert(v[1].getValue() == 2)
    derivs = ad.get_deriv(v) 
    assert(np.array_equal(np.array([deriv.get('v1', 0) for deriv in derivs]), np.array([1, 0])))
    assert(np.array_equal(np.array([deriv.get('v2', 0) for deriv in derivs]), np.array([0, 1])))
    jacobian = ad.get_jacobian(v, ['v1', 'v2'])
    assert(np.array_equal(jacobian, np.array([[1, 0], [0, 1]])))
    jacobian = ad.get_jacobian(v, ['v1', 'v2', 'hello'])
    assert(np.array_equal(jacobian, np.array([[1, 0, 0], [0, 1, 0]])))

    v = ad.create_vector('v', [1, 2], [3, 4])
    assert(v[0].getValue() == 1)
    assert(v[1].getValue() == 2)
    derivs = ad.get_deriv(v) 
    assert(np.array_equal(np.array([deriv.get('v1', 0) for deriv in derivs]), np.array([3, 0])))
    assert(np.array_equal(np.array([deriv.get('v2', 0) for deriv in derivs]), np.array([0, 4])))
    jacobian = ad.get_jacobian(v, ['v1', 'v2'])
    assert(np.array_equal(jacobian, np.array([[3, 0], [0, 4]])))
    jacobian = ad.get_jacobian(v, ['v1', 'v2', 'hello'])
    assert(np.array_equal(jacobian, np.array([[3, 0, 0], [0, 4, 0]])))

    with pytest.raises(Exception):    
        v = ad.create_vector('v', [1, 2], [3, 4, 5])
    
    x = ad.Scalar('x', 1)
    y = ad.Scalar('y', 2)
    v = np.array([x, y])
    assert(np.array_equal(ad.get_value(v), np.array([1, 2])))
    jacobian = ad.get_jacobian(v, ['v1', 'v2'])
    assert(np.array_equal(jacobian, np.array([[0, 0], [0, 0]])))
    jacobian = ad.get_jacobian(v, ['x', 'y'])
    assert(np.array_equal(jacobian, np.array([[1, 0], [0, 1]])))

    x = ad.Scalar('x', 1)
    y = ad.Scalar('y', 2)
    v = np.array([x, 2 * y])
    assert(np.array_equal(ad.get_value(v), np.array([1, 4])))
    jacobian = ad.get_jacobian(v, ['x', 'y'])
    assert(np.array_equal(jacobian, np.array([[1, 0], [0, 2]])))
    jacobian = ad.get_jacobian(v, ['y', 'x'])
    assert(np.array_equal(jacobian, np.array([[0, 1], [2, 0]])))
    
    x = ad.Scalar('x', 1)
    y = ad.Scalar('y', 2)
    v = np.array([x + y, 2 * y])
    assert(np.array_equal(ad.get_value(v), np.array([3, 4])))
    jacobian = ad.get_jacobian(v, ['x', 'y'])
    assert(np.array_equal(jacobian, np.array([[1, 1], [0, 2]])))
    jacobian = ad.get_jacobian(v, ['y', 'x'])
    assert(np.array_equal(jacobian, np.array([[1, 1], [2, 0]])))

def test_add():
    v1 = ad.create_vector('v', [1, 2])
    v2 = ad.create_vector('v', [1, 5])
    v3 = v1 + v2 
    assert(v3[0].getValue() == 2)
    assert(v3[1].getValue() == 7)
    jacobian = ad.get_jacobian(v3, ['v1', 'v2'])
    assert(np.array_equal(jacobian, np.array([[2, 0], [0, 2]])))

    v1 = ad.create_vector('v', [1, 2])
    v2 = v1 + 10
    assert(v2[0].getValue() == 11)
    assert(v2[1].getValue() == 12)
    jacobian = ad.get_jacobian(v2, ['v1', 'v2'])
    assert(np.array_equal(jacobian, np.array([[1, 0], [0, 1]])))

    v1 = ad.create_vector('v', [1, 2])
    v2 = ad.Scalar('v2', 4)
    v3 = ad.Scalar('v1', 7)
    v4 = v1 + np.array([v2, v3])
    assert(v4[0].getValue() == 5)
    assert(v4[1].getValue() == 9)
    jacobian = ad.get_jacobian(v4, ['v1', 'v2'])
    assert(np.array_equal(jacobian, np.array([[1, 1], [1, 1]])))    

    x = ad.Scalar('x', 1)
    y = ad.Scalar('y', 2)
    v1 = np.array([x, y])
    v2 = ad.create_vector('v', [1, 5])
    v3 = v1 + v2
    assert(v3[0].getValue() == 2)
    assert(v3[1].getValue() == 7)   
    jacobian = ad.get_jacobian(v3, ['x', 'y'])
    assert(np.array_equal(jacobian, np.array([[1, 0], [0, 1]])))    

    x = ad.Scalar('x', 1)
    y = ad.Scalar('y', 2)
    v1 = np.array([x, y])
    v2 = np.array([x + y, x])
    v3 = v1 + v2
    assert(v3[0].getValue() == 4)
    assert(v3[1].getValue() == 3)   
    jacobian = ad.get_jacobian(v3, ['x', 'y'])
    assert(np.array_equal(jacobian, np.array([[2, 1], [1, 1]])))    

    x = ad.Scalar('x', 1)
    y = ad.Scalar('y', 2)
    v1 = np.array([x, y])
    v2 = np.array([y, 10])
    v3 = v1 + v2
    assert(v3[0].getValue() == 3)
    assert(v3[1].getValue() == 12)  
    jacobian = ad.get_jacobian(v3, ['x', 'y'])
    assert(np.array_equal(jacobian, np.array([[1, 1], [0, 1]])))    

def test_mul():
    v1 = ad.create_vector('v', [1, 2])
    v2 = ad.create_vector('w', [3, 5])
    v3 = v1 * v2 
    assert(v3[0].getValue() == 3)
    assert(v3[1].getValue() == 10)
    jacobian = ad.get_jacobian(v3, ['v1', 'v2', 'w1', 'w2'])
    assert(np.array_equal(jacobian, np.array([[3, 0, 1, 0], [0, 5, 0, 2]])))    

    x = ad.Scalar('x', 1)
    y = ad.Scalar('y', 2)
    v = ad.Scalar('v', 3)
    v1 = np.array([x, y])
    v2 = np.array([v, 3 * v])
    v3 = v1 * v2
    assert(v3[0].getValue() == 3)
    assert(v3[1].getValue() == 18)  
    jacobian = ad.get_jacobian(v3, ['x', 'y', 'v'])
    assert(np.array_equal(jacobian, np.array([[3, 0, 1], [0, 9, 6]])))    


    v1 = ad.create_vector('v', [2, 3])
    v3 = v1 * v1
    assert(v3[0].getValue() == 4)
    assert(v3[1].getValue() == 9)
    jacobian = ad.get_jacobian(v3, ['v1', 'v2'])
    assert(np.array_equal(jacobian, np.array([[4, 0], [0, 6]])))    


    v1 = ad.create_vector('v', [1, 2])
    v2 = v1 * 10
    assert(v2[0].getValue() == 10)
    assert(v2[1].getValue() == 20)
    jacobian = ad.get_jacobian(v2, ['v1', 'v2'])
    assert(np.array_equal(jacobian, np.array([[10, 0], [0, 10]])))    


    x = ad.Scalar('x', 5)
    y = ad.Scalar('y', 2)
    v1 = np.array([x, y])
    v2 = np.array([x * y, (x + y)])
    v3 = v1 * v2
    assert(v3[0].getValue() == 50)
    assert(v3[1].getValue() == 14)   
    jacobian = ad.get_jacobian(v3, ['x', 'y'])
    assert(np.array_equal(jacobian, np.array([[20, 25], [2, 9]])))    

    x = ad.Scalar('x', 1)
    y = ad.Scalar('y', 2)
    v1 = np.array([x, y])
    v2 = np.array([y, 10])
    v3 = v1 * v2
    assert(v3[0].getValue() == 2)
    assert(v3[1].getValue() == 20)  
    jacobian = ad.get_jacobian(v3, ['x', 'y'])
    assert(np.array_equal(jacobian, np.array([[2, 1], [0, 10]])))    

def test_neg():
    v1 = ad.create_vector('v', [1, 2])
    v2 = -v1
    assert(v2[0].getValue() == -1)
    assert(v2[1].getValue() == -2)
    jacobian = ad.get_jacobian(v2, ['v1', 'v2'])
    assert(np.array_equal(jacobian, np.array([[-1, 0], [0, -1]])))
    v3 = -v2
    assert(v3[0].getValue() == 1)
    assert(v3[1].getValue() == 2)
    jacobian = ad.get_jacobian(v3, ['v1', 'v2'])
    assert(np.array_equal(jacobian, np.array([[1, 0], [0, 1]])))

    v1 = ad.create_vector('v', [1, 2])
    v2 = -1 * -v1
    assert(v2[0].getValue() == 1)
    assert(v2[1].getValue() == 2)
    jacobian = ad.get_jacobian(v2, ['v1', 'v2'])
    assert(np.array_equal(jacobian, np.array([[1, 0], [0, 1]])))


def test_sub():
    x = ad.Scalar('x', 1)
    y = ad.Scalar('y', 2)
    v1 = np.array([x, y])
    v2 = np.array([y, x])
    v3 = v1 - v2 
    assert(v3[0].getValue() == -1)
    assert(v3[1].getValue() == 1)
    jacobian = ad.get_jacobian(v3, ['x', 'y'])
    assert(np.array_equal(jacobian, np.array([[1, -1], [-1, 1]])))

    v1 = ad.create_vector('v', [1, 2])
    v2 = v1 - 10
    assert(v2[0].getValue() == -9)
    assert(v2[1].getValue() == -8)
    jacobian = ad.get_jacobian(v2, ['v1', 'v2'])
    assert(np.array_equal(jacobian, np.array([[1, 0], [0, 1]])))

    x = ad.Scalar('x', 1)
    y = ad.Scalar('y', 2)
    v1 = np.array([x, y])
    v2 = ad.create_vector('v', [1, 5])
    v3 = v1 - v2
    assert(v3[0].getValue() == 0)
    assert(v3[1].getValue() == -3)  
    jacobian = ad.get_jacobian(v3, ['x', 'y', 'v1', 'v2'])
    assert(np.array_equal(jacobian, np.array([[1, 0, -1, 0], [0, 1, 0, -1]])))

    x = ad.Scalar('x', 1)
    y = ad.Scalar('y', 2)
    v1 = np.array([x, y])
    v2 = np.array([y, 10])
    v3 = v1 - v2
    assert(v3[0].getValue() == -1)
    assert(v3[1].getValue() == -8)  
    jacobian = ad.get_jacobian(v3, ['x', 'y'])
    assert(np.array_equal(jacobian, np.array([[1, -1], [0, 1]])))

def test_pow():
    v1 = ad.create_vector('v', [2, 5])
    v2 = v1 ** 2
    assert(v2[0].getValue() == 4)
    assert(v2[1].getValue() == 25)
    jacobian = ad.get_jacobian(v2, ['v1', 'v2'])
    assert(np.array_equal(jacobian, np.array([[4, 0], [0, 10]])))

    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 5)
    v1 = np.array([x, y])
    v2 = v1 ** 2
    assert(v2[0].getValue() == 4)
    assert(v2[1].getValue() == 25)
    jacobian = ad.get_jacobian(v2, ['x', 'y'])
    assert(np.array_equal(jacobian, np.array([[4, 0], [0, 10]])))

    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 3)
    v1 = np.array([x, y])
    v2 = (v1 ** 2) ** 3
    assert(v2[0].getValue() == 64)
    assert(v2[1].getValue() == 729)
    jacobian = ad.get_jacobian(v2, ['x', 'y'])
    assert(np.array_equal(jacobian, np.array([[6 * (2 ** 5), 0], [0, 6 * (3 ** 5)]])))

    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 3)
    v1 = np.array([x, y])
    v2 = np.array([y, 2])
    v3 = v1 ** v2
    assert(v3[0].getValue() == 8)
    assert(v3[1].getValue() == 9)
    jacobian = ad.get_jacobian(v3, ['x', 'y'])
    assert(np.array_equal(jacobian, np.array([[12, np.log(2) * 8], [0, 6]])))

def test_rpow():
    v1 = ad.create_vector('v', [2, 5])
    v2 = 2 ** v1
    assert(v2[0].getValue() == 4)
    assert(v2[1].getValue() == 32)
    jacobian = ad.get_jacobian(v2, ['v1', 'v2'])
    assert(np.array_equal(jacobian, np.array([[np.log(2) * 4, 0], [0, np.log(2) * 32]])))

    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 5)
    v1 = np.array([x, y])
    v2 = 2 ** v1
    assert(v2[0].getValue() == 4)
    assert(v2[1].getValue() == 32)
    jacobian = ad.get_jacobian(v2, ['x', 'y'])
    assert(np.array_equal(jacobian, np.array([[np.log(2) * 4, 0], [0, np.log(2) * 32]])))

    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 3)
    v1 = np.array([x, y])
    v2 = 2 ** (2 * v1)
    assert(v2[0].getValue() == 16)
    assert(v2[1].getValue() == 64)
    jacobian = ad.get_jacobian(v2, ['x', 'y'])
    assert(np.array_equal(jacobian, np.array([[np.log(2) * 32, 0], [0, np.log(2) * 128]])))

    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 3)
    v1 = np.array([x, y])
    v2 = (2 ** 2) ** v1
    assert(v2[0].getValue() == 16)
    assert(v2[1].getValue() == 64)
    jacobian = ad.get_jacobian(v2, ['x', 'y'])
    assert(np.array_equal(jacobian, np.array([[np.log(2) * (2 ** 4) * 2, 0], [0, np.log(2) * (2 ** 6) * 2]])))

    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 3)
    v1 = np.array([x + y, x])
    v2 = (2 ** 2) ** v1
    assert(v2[0].getValue() == 2 ** 10)
    assert(v2[1].getValue() == 16)
    jacobian = ad.get_jacobian(v2, ['x', 'y'])
    assert(np.array_equal(jacobian, np.array([[np.log(2) * (2 ** 10) * 2, np.log(2) * (2 ** 10) * 2], [np.log(2) * (2 ** 4) * 2, 0]])))

    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 3)
    v1 = np.array([x + y, x])
    v2 = 2 ** (2 * v1)
    assert(v2[0].getValue() == 2 ** 10)
    assert(v2[1].getValue() == 16)
    jacobian = ad.get_jacobian(v2, ['x', 'y'])
    assert(np.array_equal(jacobian, np.array([[np.log(2) * (2 ** 10) * 2, np.log(2) * (2 ** 10) * 2], [np.log(2) * (2 ** 4) * 2, 0]])))


def test_exp():
    v1 = ad.create_vector('v', [2, 5])
    v2 = ad.exp(v1)
    assert(np.isclose(v2[0].getValue(), np.exp(2)))
    assert(np.isclose(v2[1].getValue(), np.exp(5)))
    jacobian = ad.get_jacobian(v2, ['v1', 'v2'])
    assert(np.array_equal(jacobian, np.array([[np.exp(2), 0], [0, np.exp(5)]])))

    v1 = ad.create_vector('v', [2, 5])
    v2 = ad.exp(2 * v1)
    assert(np.isclose(v2[0].getValue(), np.exp(4)))
    assert(np.isclose(v2[1].getValue(), np.exp(10)))
    jacobian = ad.get_jacobian(v2, ['v1', 'v2'])
    assert(np.array_equal(jacobian, 2 * np.array([[np.exp(4), 0], [0, np.exp(10)]])))

    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 3)
    v1 = ad.exp(np.array([x + y, x * y]))
    assert(np.isclose(v1[0].getValue(), np.exp(5)))
    assert(np.isclose(v1[1].getValue(), np.exp(6)))
    jacobian = ad.get_jacobian(v1, ['x', 'y'])
    assert(np.array_equal(jacobian, np.array([[np.exp(5), np.exp(5)], [3 * np.exp(6), 2 * np.exp(6)]])))


def test_sin():
    v1 = ad.create_vector('v', [0, 100])
    v2 = ad.sin(v1)
    assert(v2[0].getValue() == 0)
    assert(np.isclose(v2[1].getValue(), np.sin(100)))
    jacobian = ad.get_jacobian(v2, ['v1', 'v2'])
    assert(np.array_equal(jacobian, np.array([[1, 0], [0, np.cos(100)]])))

    v1 = ad.Scalar('x', 4)
    v2 = ad.Scalar('y', 10)
    v3 = ad.sin(np.array([v1, v2])) / ad.sin(np.array([v1, v2]))
    assert(np.isclose(v3[0].getValue(), 1))
    assert(np.isclose(v3[1].getValue(), 1))
    jacobian = ad.get_jacobian(v3, ['x', 'y'])
    assert(np.isclose(jacobian, np.array([[0, 0], [0, 0]])).all())

    v1 = ad.Scalar('x', 4)
    v2 = ad.Scalar('y', 10)
    v3 = ad.sin(np.array([v1, v2])) ** 2
    assert(np.isclose(v3[0].getValue(), np.sin(4) ** 2))
    assert(np.isclose(v3[1].getValue(), np.sin(10) ** 2))
    jacobian = ad.get_jacobian(v3, ['x', 'y'])
    assert(np.isclose(jacobian, np.array([[2 * np.sin(4) * np.cos(4), 0], [0, 2 * np.sin(10) * np.cos(10)]])).all())

    v1 = ad.Scalar('x', 4)
    v2 = ad.Scalar('y', 10)
    v3 = ad.sin(np.array([v1 * v2, v1 + v2])) ** 2
    assert(np.isclose(v3[0].getValue(), np.sin(40) ** 2))
    assert(np.isclose(v3[1].getValue(), np.sin(14) ** 2))
    jacobian = ad.get_jacobian(v3, ['x', 'y'])
    assert(np.isclose(jacobian, np.array([[2 * np.sin(40) * np.cos(40)  * 10, 2 * np.sin(40) * np.cos(40)  * 4], 
            [2 * np.sin(14) * np.cos(14), 2 * np.sin(14) * np.cos(14)]])).all())

def test_cos():
    #Similar to sin. 
    v1 = ad.create_vector('v', [0, 100])
    v2 = ad.cos(v1)
    assert(v2[0].getValue() == 1)
    assert(np.isclose(v2[1].getValue(), np.cos(100)))
    jacobian = ad.get_jacobian(v2, ['v1', 'v2'])
    assert(np.isclose(jacobian, np.array([[0, 0], [0, -np.sin(100)]])).all())


def test_tan():
    v1 = ad.create_vector('v', [0, 100])
    v2 = ad.tan(v1)
    assert(v2[0].getValue() == 0)
    assert(np.isclose(v2[1].getValue(), np.tan(100)))
    jacobian = ad.get_jacobian(v2, ['v1', 'v2'])
    assert(np.isclose(jacobian, np.array([[1, 0], [0, 1 / (np.cos(100) ** 2)]])).all())






