import sys
import os
import numpy as np
import pytest

sys.path.append('..')
import autodiff as ad

def test_composite():

    x = ad.Scalar('x', 16)
    y = ad.sin(ad.sqrt(x))
    assert(y.getValue() == np.sin(4))
    assert(y.getDeriv()['x'] == 1/8 * np.cos(4))

    #trig identity
    x = ad.Scalar('x', 5)
    y = ad.sin(x) ** 2 + ad.cos(x) ** 2
    assert(y.getValue() == 1)
    assert(y.getDeriv()['x'] == 0)


    x = ad.Scalar('x', 16)
    y = ad.sqrt(ad.power(x, 2))
    assert(y.getValue() == 16)
    assert(y.getDeriv()['x'] == 1)

    x = ad.Scalar('x', 10)
    y = ad.tan(x) * ad.cos(x) / ad.sin(x)
    assert(y.getValue() == 1)
    assert(np.isclose(y.getDeriv()['x'], 0))


    x = ad.Scalar('x', 16)
    y = ad.Scalar('y', 9)
    z = ad.sqrt(x * y)
    assert(z.getValue() == 12)
    assert(z.getDeriv()['x'] == 9/2 * (16 * 9)**(-0.5))
    assert(z.getDeriv()['y'] == 16/2 * (16 * 9)**(-0.5))


    x = ad.Scalar('x', 3)
    y = ad.Scalar('y', 2)
    z = ad.cos(ad.sin(x * y))
    assert(z.getValue() == np.cos(np.sin(6)))
    assert(z.getDeriv()['x'] == -2 * np.cos(6) * np.sin(np.sin(6)))
    assert(z.getDeriv()['y'] == -3 * np.cos(6) * np.sin(np.sin(6)))