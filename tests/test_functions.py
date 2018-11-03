import sys
import os
import numpy as np
import pytest

sys.path.append('..')
import autodiff as ad

def test_sin():
    
    x = ad.Scalar('x', 2)
    val = ad.sin(x)

    assert (np.isclose(val.getValue(),0.90929742682)==True)
    assert (np.isclose(val.getDeriv()['x'],-0.41614683654)==True)

    x = ad.Scalar('x', -2)
    val = ad.sin(x)

    assert (np.isclose(val.getValue(),-0.9092974268256817)==True)
    assert (np.isclose(val.getDeriv()['x'],-0.4161468365471424)==True)

    x = ad.Scalar('x', 0)
    val = ad.sin(x)

    assert (np.isclose(val.getValue(),0.0)==True)
    assert (np.isclose(val.getDeriv()['x'],1.0)==True)
    
def test_cos():
    
    x = ad.Scalar('x', 2)
    val = ad.cos(x)

    assert (np.isclose(val.getValue(),-0.41614683654)==True)
    assert (np.isclose(val.getDeriv()['x'],-0.9092974268256817)==True)

    x = ad.Scalar('x', -2)
    val = ad.cos(x)

    assert (np.isclose(val.getValue(),-0.4161468365471424)==True)
    assert (np.isclose(val.getDeriv()['x'],0.9092974268256817)==True)

    x = ad.Scalar('x', 0)
    val = ad.cos(x)

    assert (np.isclose(val.getValue(),1.0)==True)
    assert (np.isclose(val.getDeriv()['x'],0.0)==True)

def test_exp():
    x = ad.Scalar('x', 8)
    y = ad.exp(x)
    assert(np.isclose(y.getValue(), np.exp(8)))
    assert(np.isclose(y.getDeriv()['x'], np.exp(8)))

    x = ad.Scalar('x', -3)
    x._deriv['x'] = -2.3
    y = ad.exp(x)
    assert(np.isclose(y.getValue(), np.exp(-3)))
    assert(np.isclose(y.getDeriv()['x'], -2.3 * np.exp(-3)))
