import sys
import os
import numpy as np
import pytest

sys.path.append('..')
import autodiff as ad


def test_add():
    
    x,y = ad.Scalar('x', 2),ad.Scalar('y', 5)
    val = x+y
    assert(val.getValue() == 7.0)
    assert(val.getDeriv()['x'] == 1.0)
    assert(val.getDeriv()['y'] == 1.0)
    
    x,y = ad.Scalar('x', 2.0),ad.Scalar('y', 5.0)
    val = x+y
    assert(val.getValue() == 7.0)
    assert(val.getDeriv()['x'] == 1.0)
    assert(val.getDeriv()['y'] == 1.0)

    x,y = ad.Scalar('x', 0),ad.Scalar('y', 0)
    val = x+y
    assert(val.getValue() == 0.0)
    assert(val.getDeriv()['x'] == 1.0)
    assert(val.getDeriv()['y'] == 1.0)
    
    x,y = ad.Scalar('x', 2),ad.Scalar('y', 5)
    y._deriv['y']=-3
    val = x+y
    assert (val.getValue()==7.0)
    assert (val.getDeriv()['x']==1.0)
    assert (val.getDeriv()['y']==-3.0)
    
def test_mul():

    x = ad.Scalar('x', 5)
    val = 5 * x
    assert (val.getValue()==25.0)
    assert(val.getDeriv()['x']==5.0)

    x,y = ad.Scalar('x', 5), ad.Scalar('y', 6)
    val = x*y
    assert (val.getValue()==30.0)
    assert(val.getDeriv()['x']==6.0)
    assert(val.getDeriv()['y']==5.0)

    x,y = ad.Scalar('x', 5.0), ad.Scalar('y', 6.0)
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
    
    
def test_neg():
    x,y = ad.Scalar('x', 5), ad.Scalar('y', 1)
    val = x-y
    assert (val.getValue()==4.0)
    assert(val.getDeriv()['x']==1.0)
    assert(val.getDeriv()['y']==-1.0)

    x,y = ad.Scalar('x', 5.0), ad.Scalar('y', 1.0)
    val = x-y
    assert (val.getValue()==4.0)
    assert(val.getDeriv()['x']==1.0)
    assert(val.getDeriv()['y']==-1.0)

    x,y = ad.Scalar('x', 3), ad.Scalar('y', 5)
    val = x-y
    assert (val.getValue()==-2.0)
    assert(val.getDeriv()['x']==1.0)
    assert(val.getDeriv()['y']==-1.0)

def test_sub():
    
    x= ad.Scalar('x', 1)
    val = x-5
    assert (val.getValue()==-4.0)
    assert(val.getDeriv()['x']==1.0)
    
    x= ad.Scalar('x', 1.0)
    val = x-5
    assert (val.getValue()==-4.0)
    assert(val.getDeriv()['x']==1.0)

    x= ad.Scalar('x', 0)
    val = x-3.3
    assert (val.getValue()==-3.3)
    assert(val.getDeriv()['x']==1.0)
    
    with pytest.raises(TypeError):
        x - "5"

        
def test_pow():
    
    x=ad.Scalar('x', 2)
    val = x**2
    assert(val.getValue()==4.0)
    assert(val.getDeriv()['x']==4.0)

    x=ad.Scalar('x', 2.0)
    val = x**2
    assert(val.getValue()==4.0)
    assert(val.getDeriv()['x']==4.0)

    x=ad.Scalar('x', 1)
    val = x**2
    assert(val.getValue()==1.0)
    assert(val.getDeriv()['x']==2.0)
    
    x=ad.Scalar('x', -2)
    val = x**2
    assert(np.isclose(val.getValue(), 4.0) );
    assert(np.isclose(val.getDeriv()['x'], -4.0) );
    
    x=ad.Scalar('x', -2)
    val = x**3
    assert(np.isclose(val.getValue(), -8.0) );
    assert(np.isclose(val.getDeriv()['x'], 12.0) );
    
    
    
    #make sure Scalar**Scalar is right
    x = ad.Scalar('x', 4);
    z = ad.Scalar('z', 2.3);
    val = x**z;
    assert(np.isclose(val.getValue(), 4**2.3) );
    assert(np.isclose(val.getDeriv()['x'], 2.3*(4**1.3)) );
    assert(np.isclose(val.getDeriv()['z'], np.log(4)*4**2.3));
    

    x=ad.Scalar('x', 0)
    val = x**2
    assert(val.getValue()==0.0)
    assert(val.getDeriv()['x']==0.0)

    x = ad.Scalar('x', 0)
    with pytest.raises(ZeroDivisionError):
        x ** 0.8
    with pytest.raises(ZeroDivisionError):
        x ** -0.1
    
    y = x ** 3.5
    assert(y.getValue() == 0)
    assert(y.getDeriv()['x'] == 0)
    

def test_rpow():
    
    x=ad.Scalar('x', 2)
    val = 2.0**x
    assert(val.getValue())
    assert(val.getDeriv()['x'])
    assert(np.isclose(val._deriv['x'], 4.0 * np.log(2.0))==True)

    x=ad.Scalar('x', 2.0)
    val = 2.0**x
    assert(val.getValue())
    assert(val.getDeriv()['x'])
    assert(np.isclose(val._deriv['x'], 4.0 * np.log(2.0))==True)

    x=ad.Scalar('x', 2)
    val = 3.0**x
    assert(val.getValue()==9)
    assert(np.isclose(val._deriv['x'], 9.0 * np.log(3.0))==True)


    x=ad.Scalar('x', 2)
    val = 1.0**x
    assert(val.getValue()==1)
    assert(np.isclose(val._deriv['x'], 0.0)==True)
    
    #test that 0**2 returns value=0 and derive=0
    x=ad.Scalar('x', 2)
    val = 0**x
    assert(val.getValue()==0)
    assert(np.isclose(val._deriv['x'], 0.0)==True)
    
    #test that 0**-1 is a ZeroDivisionError
    with pytest.raises(ZeroDivisionError):    
        z = ad.Scalar('z', -1);
        0**z
    
    #test that 0**0 is a ZeroDivisionError
    with pytest.raises(ZeroDivisionError):    
        x = ad.Scalar('x', 0);
        0**x


def test_truediv():
    x,y=ad.Scalar('x', 3),ad.Scalar('y', 2)
    val = x/y
    assert(val.getValue() == 1.5)
    assert(val.getDeriv()['x']== 0.5)
    assert(val.getDeriv()['y']== -0.75)

    x,y=ad.Scalar('x', 3.0),ad.Scalar('y', 2.0)
    val = x/y
    assert(val.getValue() == 1.5)
    assert(val.getDeriv()['x']== 0.5)
    assert(val.getDeriv()['y']== -0.75)

    x,y=ad.Scalar('x', 0),ad.Scalar('y', 8)
    val = x/y
    assert(val.getValue() == 0.0)
    assert(val.getDeriv()['x'] == 0.125)
    assert(val.getDeriv()['y'] == -0.0)

    x=ad.Scalar('x', 0)
    val = x/8
    assert(val.getValue() == 0.0)
    assert(val.getDeriv()['x'] == 0.125)

    x = ad.Scalar('x', 3)
    with pytest.raises(ZeroDivisionError):
        x / 0.0
        
    x = ad.Scalar('x', 3)
    with pytest.raises(ZeroDivisionError):
        x / ad.Scalar('z', 0.0)
        
    with pytest.raises(TypeError):
        x / "0"
        
def test_rtruediv():
    
    x,y=ad.Scalar('x', 3),ad.Scalar('y', 2)
    val = x/y
    assert(val.getValue() ==1.5)
    assert(val.getDeriv()['x'] ==0.5)
    assert(val.getDeriv()['y'] ==-0.75)

    x,y=ad.Scalar('x', 3.0),ad.Scalar('y', 2.0)
    val = x/y
    assert(val.getValue() ==1.5)
    assert(val.getDeriv()['x'] ==0.5)
    assert(val.getDeriv()['y'] ==-0.75)

    y=ad.Scalar('y', 1)
    val = 10/y
    assert(val.getValue() ==10.0)
    assert(val.getDeriv()['y'] ==-10.0)

    y=ad.Scalar('y', 2)
    val = 8/y
    assert(val.getValue()==4.0 )
    assert(val.getDeriv()['y']==-2.0)

        
    y = ad.Scalar('x', 0.0)
    with pytest.raises(ZeroDivisionError):
        3 / y
    
    z = ad.Scalar('x', 34)
    with pytest.raises(TypeError):
        "3"/z


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
    assert(x.getDeriv()['y'] == -2)

    x += y + 8
    assert(x.getValue() == 23)
    assert(x.getDeriv()['x'] == 1)
    assert(x.getDeriv()['y'] == -5)


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

    x,y = ad.Scalar('x', 2.0), ad.Scalar('y', 5.0)
    val = 2 - (x - y)
    assert(val.getValue() == 5.0)
    assert(val.getDeriv()['x'] == -1.0)
    assert(val.getDeriv()['y'] == 1.0)

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

    x,y = ad.Scalar('x', 2.0),ad.Scalar('y', 5.0)
    val = 2 + (x+y)
    assert(val.getValue() == 9.0)
    assert(val.getDeriv()['x'] == 1.0)
    assert(val.getDeriv()['y'] == 1.0)


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
    assert(x.getDeriv()['y'] == 1.5)

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
    assert(x.getDeriv()['x'] == 0)

    x = ad.Scalar('x', 0)
    with pytest.raises(ZeroDivisionError):
        x **= 0.5

    x = ad.Scalar('x', 0)
    y = ad.Scalar('y', 0)
    x **= y
    assert(x.getValue() == 1)
    assert(x.getDeriv()['x'] == 0)
    assert(x.getDeriv()['y'] == 0)

    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 1)

    x **= x + y

    assert(x.getValue() == 8)
    assert(abs(x.getDeriv()['x'] - 8*(1.5 + np.log(2))) < 1e-7)

def test_str():
    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 3)
    z = x + y
    assert(str(z) == "Value: 5.0, Derivatives: {'x': 1.0, 'y': 1.0}" or
           str(z) == "Value: 5.0, Derivatives: {'y': 1.0, 'x': 1.0}")

def test_repr():
    assert(repr(ad.Scalar('x', 2)) == "Scalar(2.0)")

def test_get_gradients():
    x = ad.Scalar('x', 4)
    y = -2 * x
    d = y.getGradient(['x', 'z'])
    assert(np.array_equal(d, [-2, 0]))
    
    
def test_composite():
    #Test assortment of composite operations.
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



    
    
