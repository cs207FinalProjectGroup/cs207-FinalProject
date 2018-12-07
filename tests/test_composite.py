import sys
import os
import numpy as np
import pytest

sys.path.append('..')
import autodiff as ad

def test_composite():
    #Test some more complicated functions / identities, including some multivariate ones.


    x = ad.Scalar('x', 2)
    z = (5 * (x + 20)  / 10) ** 2
    d = z.getGradient(['x'])
    assert(z.getValue() == 121)
    assert(np.array_equal(d, [11]))

    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 3)
    z = (x + 20) * y
    d = z.getGradient(['x', 'y'])
    assert(z.getValue() == 66)
    assert(np.array_equal(d, [3, 22]))

    x = ad.Scalar('x', 1)
    y = ad.Scalar('y', 3)
    z = (x * y + x) * y
    d = z.getGradient(['x', 'y'])
    assert(z.getValue() == 12)
    assert(np.array_equal(d, [12, 7]))

    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 3)
    z = (x + y) / y
    d = z.getGradient(['x', 'y'])
    assert(np.isclose(z.getValue(), 5/3))
    assert(np.allclose(d, [1.0/3, -2.0/9]))

    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 3)
    z = x + ((y ** 2) / y)
    d = z.getGradient(['x', 'y'])
    assert(z.getValue() == 5)
    assert(np.array_equal(d, [1, 1]))

    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 3)
    z = (x ** y) ** 2
    d = z.getGradient(['x', 'y'])
    assert(z.getValue() == 64)
    assert(np.array_equal(d, [6 * 32, 2 * np.log(2) * 64]))        

    x = ad.Scalar('x', 16)
    y = ad.sin(ad.sqrt(x))
    assert(np.isclose(y.getValue(), np.sin(4)))
    assert(np.isclose(y.getDeriv()['x'], 1/8 * np.cos(4)))

    #trig identity
    x = ad.Scalar('x', 5)
    y = ad.sin(x) ** 2 + ad.cos(x) ** 2
    assert(np.isclose(y.getValue(), 1))
    assert(np.isclose(y.getDeriv()['x'], 0))

    #trig identity multivariate
    x = ad.Scalar('x', 5)
    y = ad.Scalar('y', 243423)
    z = ad.sin(x * 1231 * y) ** 2 + ad.cos(x * 1231 * y) ** 2
    assert(np.isclose(z.getValue(), 1))
    assert(np.isclose(z.getDeriv()['x'], 0))
    assert(np.isclose(z.getDeriv()['y'], 0))

    x = ad.Scalar('x', 16)
    y = ad.sqrt(ad.power(x, 2))
    assert(y.getValue() == 16)
    assert(y.getDeriv()['x'] == 1)

    x = ad.Scalar('x', 10)
    y = ad.tan(x) * ad.cos(x) / ad.sin(x)
    assert(np.isclose(y.getValue(), 1))
    assert(np.isclose(y.getDeriv()['x'], 0))

    #https://math.berkeley.edu/~kruckman/fall2010/9-29-problems.pdf
    x = ad.Scalar('x', 10)
    y = (x ** 2 * ad.sin(x) / (x ** 2 + 1))
    assert(np.isclose(y.getValue(), 100 * np.sin(10) / (101)))
    assert(np.isclose(y.getDeriv()['x'], (20 * np.sin(10) + (10 ** 4 + 100) * np.cos(10)) / (101 ** 2)))

    x = ad.Scalar('x', 4)
    y = (x ** 3 * ad.exp(x))
    assert(np.isclose(y.getValue(), 4 ** 3 * (np.exp(4))))
    assert(np.isclose(y.getDeriv()['x'], 3 * 16 * np.exp(4) + 4 ** 3 * np.exp(4)))

    x = ad.Scalar('x', 4)
    y = ad.sin(x) * ad.cos(x) * ad.tan(x)
    assert(np.isclose(y.getValue(), np.sin(4) * np.cos(4) * np.tan(4)))
    assert(np.isclose(y.getDeriv()['x'], np.sin(8)))

    x = ad.Scalar('x', 4)
    y = ad.sqrt(x) / ad.tan(x)
    assert(np.isclose(y.getValue(), 2 / np.tan(4)))
    assert(np.isclose(y.getDeriv()['x'], 1 / (np.tan(4) * 4) - 2 / (np.sin(4) ** 2)))

    x = ad.Scalar('x', 4)
    y = ad.exp(ad.sqrt(x + 1)) 
    assert(np.isclose(y.getValue(), np.exp(5 ** 0.5)))
    assert(np.isclose(y.getDeriv()['x'], np.exp(5 ** 0.5) / (2 * (5 ** 0.5))))    

    x = ad.Scalar('x', 4)
    y = ad.exp(ad.sin(ad.exp(x)))
    assert(np.isclose(y.getValue(), np.exp(np.sin(np.exp(4)))))
    assert(np.isclose(y.getDeriv()['x'], np.cos(np.exp(4)) * np.exp(np.sin(np.exp(4)) + 4)))       

    x = ad.Scalar('x', 4)
    y = (ad.sin(x ** (1/3))) ** (1/3)
    assert(np.isclose(y.getValue(), np.sin(4 ** (1/3)) ** (1/3)))
    assert(np.isclose(y.getDeriv()['x'], np.cos(4 ** (1/3)) / (9 * ((4 * np.sin(4 ** (1/3))) ** (2/3)))))       

    x = ad.Scalar('x', 16)
    y = ad.Scalar('y', 9)
    z = ad.sqrt(x * y)
    assert(z.getValue() == 12)
    assert(np.isclose(z.getDeriv()['x'], 9/2 * (16 * 9)**(-0.5)))
    assert(np.isclose(z.getDeriv()['y'], 16/2 * (16 * 9)**(-0.5)))

    x = ad.Scalar('x', 16)
    y = ad.Scalar('y', 9)
    z = ad.sqrt(x * (y ** 4))
    assert(z.getValue() == 4 * 81)
    assert(np.isclose(z.getDeriv()['x'], (9 ** 4)/2 * (16 * (9 ** 4))**(-0.5)))
    assert(np.isclose(z.getDeriv()['y'], (4 * 16 * (9 ** 3))/2 * (16 * (9 ** 4))**(-0.5)))

    x = ad.Scalar('x', 3)
    y = ad.Scalar('y', 2)
    z = ad.cos(ad.sin(x * y))
    assert(np.isclose(z.getValue(), np.cos(np.sin(6))))
    assert(np.isclose(z.getDeriv()['x'], -2 * np.cos(6) * np.sin(np.sin(6))))
    assert(np.isclose(z.getDeriv()['y'], -3 * np.cos(6) * np.sin(np.sin(6))))

    x = ad.Scalar('x', 3)
    y = ad.Scalar('y', 2)
    x = ad.log(ad.Scalar('x', 10), 10)
    x2 = ad.log(ad.Scalar('y', 100), 10)
    x3 = x - x2
    x4 = ad.log(ad.Scalar('x', 10) / ad.Scalar('y', 100), 10)
    assert(np.isclose(x3.getValue(), -1))
    assert(np.isclose(x3.getDeriv()['x'], 1 / (np.log(10) * 10)))
    assert(np.isclose(x3.getDeriv()['y'], -1 / (np.log(10) * 100)))
    assert(np.isclose(x3.getValue(), x4.getValue()))
    assert(np.isclose(x3.getDeriv()['x'], x4.getDeriv()['x']))
    assert(np.isclose(x3.getDeriv()['y'], x4.getDeriv()['y']))

    x = ad.Scalar('x', 3)
    y = ad.Scalar('y', 2)
    z = ad.sin(2 * x + ad.exp(y ** 2) + 4 * x * y)
    assert(np.isclose(z.getValue(), np.sin(6 + np.exp(4) + 24)))    
    assert(np.isclose(z.getDeriv()['x'], np.cos(6 + np.exp(4) + 24) * (10)))
    assert(np.isclose(z.getDeriv()['y'],np.cos(6 + np.exp(4) + 24) * (4 * np.exp(4) + 12)))

    #http://math.gmu.edu/~memelian/teaching/Fall08/partDerivExamples.pdf
    x = ad.Scalar('x', 3)
    y = ad.Scalar('y', 2)
    z = x * ad.exp(2 * x + 3 * y)
    assert(np.isclose(z.getValue(), 3 * np.exp(12)))    
    assert(np.isclose(z.getDeriv()['x'], 6 * np.exp(12) + np.exp(12)))
    assert(np.isclose(z.getDeriv()['y'], 9 * np.exp(12)))

    x = ad.Scalar('x', 3)
    y = ad.Scalar('y', 2)
    z = (x - y) / (x + y)
    assert(np.isclose(z.getValue(), 1 / 5))    
    assert(np.isclose(z.getDeriv()['x'], 4 / 25))
    assert(np.isclose(z.getDeriv()['y'], -6 / 25))

    x = ad.Scalar('x', 3)
    y = ad.Scalar('y', 2)
    z = ad.Scalar('z', 5)
    f = x * ad.cos(z) + ((x ** 2) * (y ** 3) * ad.exp(z))
    assert(np.isclose(f.getValue(), 3 * np.cos(5) + (9 * 8 * np.exp(5))))    
    assert(np.isclose(f.getDeriv()['x'], np.cos(5) + 2 * 3 * 8 * np.exp(5)))
    assert(np.isclose(f.getDeriv()['y'], 3 * 9 * 4 * np.exp(5)))
    assert(np.isclose(f.getDeriv()['z'], -3 * np.sin(5) + 9 * 8 * np.exp(5)))

    x = ad.Scalar('x', 3)
    y = ad.Scalar('y', 2)
    f = (y / x) * ad.ln(x)
    assert(np.isclose(f.getValue(), 2/3 * np.log(3)))    
    assert(np.isclose(f.getDeriv()['x'], 2/9 * (1- np.log(3))))
    assert(np.isclose(f.getDeriv()['y'], 1/3 * np.log(3)))

    x = ad.Scalar('x', 3)
    y = ad.Scalar('y', 2)
    f = 1 / (x ** 2 + y ** 2)
    assert(np.isclose(f.getValue(), 1 /13))    
    assert(np.isclose(f.getDeriv()['x'], -6 / (13 ** 2)))
    assert(np.isclose(f.getDeriv()['y'], -4 / (13 ** 2)))










