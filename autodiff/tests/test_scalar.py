import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import autodiff as ad


def test_add():
    
    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 5)
    val = x+y
    assert(val.getValue() == 7.0)
    assert(val.getDeriv()['x'] == 1.0)
    assert(val.getDeriv()['y'] == 1.0)
    
    x = ad.Scalar('x', 0)
    y = ad.Scalar('y', 0)
    val = x+y
    assert(val.getValue() == 0.0)
    assert(val.getDeriv()['x'] == 1.0)
    assert(val.getDeriv()['y'] == 1.0)
    
    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 5)
    y._deriv['y']=-3
    val = x+y
    assert (val.getValue()==7.0)
    assert (val.getDeriv()['x']==1.0)
    assert (val.getDeriv()['y']==-3.0)
    
    x = ad.Scalar('x', 8)
    val = x+8
    assert (val.getValue()==16)
    assert (val.getDeriv()['x']==1)
    
def test_mul():
    
    x,y = ad.Scalar('x', 5), ad.Scalar('y', 6)

    val = x*y
    assert (val.getValue()==30.0)
    assert(val.getDeriv()['x']==6.0)
    assert(val.getDeriv()['y']==5.0)

    x,y = ad.Scalar('x', 0), ad.Scalar('y', 0)
    val = x*y
    assert (val.getValue()==0)
    assert(val.getDeriv()['x']==0)
    assert(val.getDeriv()['y']==0)

    x,y = ad.Scalar('x', 1), ad.Scalar('y', 0)
    val = x*y
    assert (val.getValue()==0)
    assert(val.getDeriv()['x']==0)
    assert(val.getDeriv()['y']==1.0)
    
    x = ad.Scalar('x', 2)
    val = x * 2
    assert(val.getValue()==4)
    assert(val.getDeriv()['x']==2)
    
def test_neg():
    x,y = ad.Scalar('x', 5), ad.Scalar('y', 1)
    val = x-y
    assert (val.getValue()==4.0)
    assert(val.getDeriv()['x']==1.0)
    assert(val.getDeriv()['y']==-1.0)

    x,y = ad.Scalar('x', 3), ad.Scalar('y', 5)
    val = x-y
    assert (val.getValue()==-2.0)
    assert(val.getDeriv()['x']==1.0)
    assert(val.getDeriv()['y']==-1.0)

    x= ad.Scalar('x', 1)
    val = x-5
    assert (val.getValue()==-4.0)
    assert(val.getDeriv()['x']==1.0)



def test_iadd():
    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 5)

    x += y
    assert(x.getValue() == 7)
    assert(x.getDeriv()['x'] == 1)
    assert(x.getDeriv()['y'] == 1)

    y = ad.Scalar('y', 0)
    y._deriv['y'] = -3
    x += y
    assert(x.getDeriv()['x'] == 1)
    assert(x.getDeriv()['y'] == -2)

    x += 8
    assert(x.getValue() == 15)
    assert(x.getDeriv()['x'] == 1)


def test_isub():
    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 5)

    x -= y
    assert(x.getValue() == -3)
    assert(x.getDeriv()['x'] == 1)
    assert(x.getDeriv()['y'] == -1)

    x -= 8
    assert(x.getValue() == -11)
    assert(x.getDeriv()['x'] == 1)
    assert(x.getDeriv()['y'] == -1)


def test_imul():
    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 5)

    x *= y
    assert(x.getValue() == 10)
    assert(x.getDeriv()['x'] == 5)
    assert(x.getDeriv()['y'] == 2)

    x *= -3
    assert(x.getValue() == -30)
    assert(x.getDeriv()['x'] == -15)
    assert(x.getDeriv()['y'] == -6)


def test_rsub():
    x = 2
    y = ad.Scalar('y', 5)

    z = x - y
    assert(z.getValue() == -3)
    assert(z.getDeriv()['y'] == -1)

    y = ad.Scalar('y', 1)
    y._deriv['y'] = -3
    z = -5.2 - y
    assert(z.getValue() == -6.2)
    assert(z.getDeriv()['y'] == 3)

    with pytest.raises(TypeError):
        "3" - y


def test_radd():
    x = 2
    y = ad.Scalar('y', 5)

    z = x + y
    assert(z.getValue() == 7)
    assert(z.getDeriv()['y'] == 1)

    y = ad.Scalar('y', 1)
    y._deriv['y'] = -3
    z = -5.2 + y
    assert(z.getValue() == -4.2)
    assert(z.getDeriv()['y'] == -3)

    with pytest.raises(TypeError):
        "3" + y


def test_rmul():
    x = 2
    y = ad.Scalar('y', 5)

    z = x * y
    assert(z.getValue() == 10)
    assert(z.getDeriv()['y'] == 2)

    z = -3 * z
    assert(z.getDeriv()['y'] == -6)

    with pytest.raises(TypeError):
        "8" * z


def test_itruediv():
    x = ad.Scalar('x', 3)
    y = ad.Scalar('y', 2)
    x /= y
    assert(x.getValue() == 1.5)
    assert(x.getDeriv()['x'] == 0.5)
    assert(x.getDeriv()['y'] == -0.75)

    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 5)
    x._deriv['x'] = -3
    x /= y
    assert(abs(x.getDeriv()['x'] - (-0.6)) < 1e-6)

    with pytest.raises(TypeError):
        x /= "5"

    with pytest.raises(ZeroDivisionError):
        x /= 0

    with pytest.raises(ZeroDivisionError):
        x /= ad.Scalar('z', 0)

    x = ad.Scalar('x', 3)
    y = ad.Scalar('y', 2)
    y._deriv['y'] = -2
    x /= y
    assert(x.getValue() == 1.5)
    assert(x.getDeriv()['y'] == 3/8)

def test_ipow():
    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 5)

    x **= y
    assert(x.getValue() == 32)
    assert(x.getDeriv()['x'] == 80)
    assert(abs(x.getDeriv()['y'] - np.log(2) * 32) < 1e-6)

    x = ad.Scalar('x', 2)
    x **= 3
    assert(x.getValue() == 8)
    assert(x.getDeriv()['x'] == 12)

    x = ad.Scalar('x', 0)
    x **= 0
    assert(x.getValue() == 1)
    assert(x.getDeriv()['x'] == 0)

    x = ad.Scalar('x', -3)
    with pytest.raises(ValueError):
        x **= 2.5

    x = ad.Scalar('x', -3)
    with pytest.raises(ValueError):
        x **= ad.Scalar('y', 1.4)

    x = ad.Scalar('x', 0)
    with pytest.raises(ZeroDivisionError):
        x **= -0.1

    x = ad.Scalar('x', 0)
    with pytest.raises(ZeroDivisionError):
        x **= -5

    x = ad.Scalar('x', 0)
    x **= 1.2
    assert(x.getValue() == 0)
    assert(x.getDeriv() == 0)

    x = ad.Scalar('x', 0)
    with pytest.raises(ZeroDivisionError):
        x **= 0.5





