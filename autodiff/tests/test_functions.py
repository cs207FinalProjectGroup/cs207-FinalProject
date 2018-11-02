import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import autodiff as ad

def test_sin():
    
    x = ad.Scalar('x', 2)
    val = sin(x)

    assert (np.isclose(val.getValue(),0.90929742682)==True)
    assert (np.isclose(val.getDeriv()['x'],-0.41614683654)==True)

    x = ad.Scalar('x', -2)
    val = sin(x)

    assert (np.isclose(val.getValue(),-0.9092974268256817)==True)
    assert (np.isclose(val.getDeriv()['x'],-0.4161468365471424)==True)

    x = ad.Scalar('x', 0)
    val = sin(x)

    assert (np.isclose(val.getValue(),0.0)==True)
    assert (np.isclose(val.getDeriv()['x'],1.0)==True)
    
def test_cos():
    
    x = ad.Scalar('x', 2)
    val = cos(x)

    assert (np.isclose(val.getValue(),-0.41614683654)==True)
    assert (np.isclose(val.getDeriv()['x'],-0.9092974268256817)==True)

    x = ad.Scalar('x', -2)
    val = cos(x)

    assert (np.isclose(val.getValue(),-0.4161468365471424)==True)
    assert (np.isclose(val.getDeriv()['x'],0.9092974268256817)==True)

    x = ad.Scalar('x', 0)
    val = cos(x)

    assert (np.isclose(val.getValue(),1.0)==True)
    assert (np.isclose(val.getDeriv()['x'],0.0)==True)