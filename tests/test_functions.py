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


def test_power():
    assert(ad.power(5, 3) == 125.0)
    
    
    
def test_sqrt():
    #basic test for sqrt function
    b = ad.Scalar('b', 4);
    b_sqrt = ad.sqrt(b);
    assert(b_sqrt.getValue() == 2.0); #4**(0.5) = 2
    assert( np.isclose(b_sqrt.getDeriv()['b'], 0.5*1/np.sqrt(4)) );
    
    #harder test for sqrt function
    c = ad.Scalar('c', 11.27);
    c_sqrt = ad.sqrt(c);
    assert(c_sqrt.getValue() == np.sqrt(11.27)); #4**(0.5) = 2
    assert( np.isclose(c_sqrt.getDeriv()['c'], 0.5*1/np.sqrt(11.27) ) );
    
    #test that sqrt works with ints/floats
    sqrt_six = ad.sqrt(6.5);
    assert(sqrt_six == np.sqrt(6.5) );
    
    #test that sqrt throws error for negative value
    with pytest.raises(TypeError):
        ad.sqrt(-1);
        
    #test that proper error is thrown for sqrt of Scalar=0
    with pytest.raises(ZeroDivisionError):    
        z = ad.Scalar('z', 0);
        z_sqrt = ad.sqrt(z);
   
    #test that sqrt function works with integer 0
    assert(ad.sqrt(0) == np.sqrt(0) ); #4**(0.5) = 2
        