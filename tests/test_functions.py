import sys
import os
import numpy as np
import pytest

sys.path.append('..')
import autodiff as ad



def test_sin():    
    x = ad.Scalar('x', 2)
    val = ad.sin(x)

    assert (np.isclose(val.getValue(),0.90929742682))
    assert (np.isclose(val.getDeriv()['x'],-0.41614683654))

    x = ad.Scalar('x', 2)
    val = ad.sin(2 * x)
    assert (np.isclose(val.getValue(),np.sin(4)))
    assert (np.isclose(val.getDeriv()['x'],2 * np.cos(4)))

    x = ad.Scalar('x', 2)
    val = 3 * ad.sin(x) + 10 * x + 5
    assert (np.isclose(val.getValue(), 3 * np.sin(2) + 25))
    assert (np.isclose(val.getDeriv()['x'], 3 * np.cos(2) + 10))


    x = ad.Scalar('x', 2)
    y = ad.Scalar('y', 5)
    val = 3 * ad.sin(x) + 10 * ad.sin(y)
    assert (np.isclose(val.getValue(), 3 * np.sin(2) + 10 * np.sin(5)))
    assert (np.isclose(val.getDeriv()['x'], 3 * np.cos(2)))
    assert (np.isclose(val.getDeriv()['y'], 10 * np.cos(5)))


    x = ad.Scalar('x', -2)
    val = ad.sin(x)
    assert (np.isclose(val.getValue(),-0.9092974268256817))
    assert (np.isclose(val.getDeriv()['x'],-0.4161468365471424))

    x = ad.Scalar('x', 0)
    val = ad.sin(x)

    assert (np.isclose(val.getValue(),0.0))
    assert (np.isclose(val.getDeriv()['x'],1.0))

    assert (ad.sin(0) == 0)
    assert (ad.sin(13443) == np.sin(13443))
    
def test_cos():
    
    x = ad.Scalar('x', 2)
    val = ad.cos(x)

    assert (np.isclose(val.getValue(),-0.41614683654))
    assert (np.isclose(val.getDeriv()['x'],-0.9092974268256817))


    x = ad.Scalar('x', 2)
    val = ad.cos(2 * x)
    assert (np.isclose(val.getValue(),np.cos(4)))
    assert (np.isclose(val.getDeriv()['x'],-2 * np.sin(4)))

    x = ad.Scalar('x', -2)
    val = ad.cos(x)

    assert (np.isclose(val.getValue(),-0.4161468365471424))
    assert (np.isclose(val.getDeriv()['x'],0.9092974268256817))

    x = ad.Scalar('x', 0)
    val = ad.cos(x)

    assert (np.isclose(val.getValue(),1.0))
    assert (np.isclose(val.getDeriv()['x'],0.0))

    x = ad.cos(ad.Scalar('x', 100))
    x2 = ad.sin(ad.Scalar('x', 100))
    val = x ** 2 + x2 ** 2
    assert (np.isclose(val.getValue(),1) )
    assert (np.isclose(val.getDeriv()['x'],0.0))     

    assert (ad.cos(0) == 1)
    assert (ad.cos(13443) == np.cos(13443))


def test_exp():
    x = ad.Scalar('x', 8)
    y = ad.exp(x)
    assert(np.isclose(y.getValue(), np.exp(8)))
    assert(np.isclose(y.getDeriv()['x'], np.exp(8)))

    x = ad.Scalar('x', 8)
    y = ad.exp(2 * x)
    assert(np.isclose(y.getValue(), np.exp(16)))
    assert(np.isclose(y.getDeriv()['x'], 2 *np.exp(16)))

    x = ad.Scalar('x', -3)
    x._deriv['x'] = -2.3
    y = ad.exp(x)
    assert(np.isclose(y.getValue(), np.exp(-3)))
    assert(np.isclose(y.getDeriv()['x'], -2.3 * np.exp(-3)))

    assert (ad.exp(0) == 1)
    assert np.isclose(ad.exp(4), np.exp(4))

    x = ad.Scalar('x', -3)
    y = ad.Scalar('y', 5)
    z = ad.exp(x) * ad.exp(y)
    assert(np.isclose(z.getValue(), np.exp(2)))
    assert(np.isclose(z.getDeriv()['x'],  np.exp(2)))
    assert(np.isclose(z.getDeriv()['y'],  np.exp(2)))

    x = ad.Scalar('x', -3)
    y = ad.Scalar('y', 5)
    z = ad.exp(x + y)
    assert(np.isclose(z.getValue(), np.exp(2)))
    assert(np.isclose(z.getDeriv()['x'],  np.exp(2)))
    assert(np.isclose(z.getDeriv()['y'],  np.exp(2)))

    x = ad.Scalar('x', -3)
    y = ad.Scalar('y', 5)
    z = ad.exp(x * y)
    assert(np.isclose(z.getValue(), np.exp(-15)))
    assert(np.isclose(z.getDeriv()['x'],  5 * np.exp(-15)))
    assert(np.isclose(z.getDeriv()['y'],  -3 * np.exp(-15)))

def test_logistic():
    x = ad.Scalar('x', 8)
    y = ad.logistic(x)
    assert(np.isclose(y.getValue(), 1 / (1 + np.exp(-8))))
    assert(np.isclose(y.getDeriv()['x'], (1 / (1 + np.exp(-8))) * (1 - 1 / (1 + np.exp(-8)))))

    x = ad.Scalar('x', 0)
    y = ad.logistic(x)
    assert(np.isclose(y.getValue(), 0.5))
    assert(np.isclose(y.getDeriv()['x'], (0.25)))   

    x = ad.Scalar('x', 5)
    y = ad.Scalar('y', 2)
    z = ad.logistic(x * y)
    assert(np.isclose(z.getValue(), 1 / (1 + np.exp(-10))))
    assert(np.isclose(z.getDeriv()['x'], 2  * (1 / (1 + np.exp(-10))) * (1 - 1 / (1 + np.exp(-10)))))   
    assert(np.isclose(z.getDeriv()['y'], 5  * (1 / (1 + np.exp(-10))) * (1 - 1 / (1 + np.exp(-10)))))   

    x = ad.Scalar('x', 8)
    y = ad.logistic(2 * x)
    assert(np.isclose(y.getValue(), 1 / (1 + np.exp(-16))))
    assert(np.isclose(y.getDeriv()['x'], 2  * (1 / (1 + np.exp(-16))) * (1 - 1 / (1 + np.exp(-16)))))

    assert(np.isclose(ad.logistic(0), 0.5))



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


def test_sinh():
    x = ad.sinh(ad.Scalar('x', 0))
    assert(np.isclose(x.getValue(), 0))
    assert(np.isclose(x.getDeriv()['x'],1.0))

    x = ad.sinh(ad.Scalar('x', 10))
    assert(np.isclose(x.getValue(), np.sinh(10)))
    assert(np.isclose(x.getDeriv()['x'],np.cosh(10)))

    x = ad.sinh(2 * ad.Scalar('x', 10))
    assert(np.isclose(x.getValue(), np.sinh(20)) )
    assert(np.isclose(x.getDeriv()['x'],2 * np.cosh(20)))

    assert(np.isclose(ad.sinh(123), np.sinh(123)))


def test_cosh():
    x = ad.cosh(ad.Scalar('x', 0))
    assert(np.isclose(x.getValue(), 1))
    assert(np.isclose(x.getDeriv()['x'],0))

    x = ad.cosh(ad.Scalar('x', 10))
    assert(np.isclose(x.getValue(), np.cosh(10)))
    assert(np.isclose(x.getDeriv()['x'],np.sinh(10)))

    x = ad.cosh(2 * ad.Scalar('x', 10))
    assert(np.isclose(x.getValue(), np.cosh(20)))
    assert(np.isclose(x.getDeriv()['x'],2 * np.sinh(20)))


    assert(np.isclose(ad.cosh(123), np.cosh(123)))

    x = ad.cosh(ad.Scalar('x', 3))
    x2 = ad.sinh(ad.Scalar('x', 3))
    x3 = x ** 2 - x2 ** 2
    assert(np.isclose(x3.getValue(), 1))
    assert(np.isclose(x3.getDeriv()['x'], 0))


def test_tanh():
    x = ad.tanh(ad.Scalar('x', 0))
    assert(np.isclose(x.getValue(), 0))
    assert(np.isclose(x.getDeriv()['x'],1))

    x = ad.tanh(ad.Scalar('x', 10))
    assert(np.isclose(x.getValue(), np.tanh(10)))
    assert(np.isclose(x.getDeriv()['x'], 1 - np.tanh(10) ** 2))

    x = ad.tanh(2 * ad.Scalar('x', 10))
    assert(np.isclose(x.getValue(), np.tanh(20)))
    assert(np.isclose(x.getDeriv()['x'], 2 * (1 - np.tanh(20) ** 2)))

    assert(np.isclose(ad.tanh(123), np.tanh(123)))


def test_log():
    x = ad.log(ad.Scalar('x', 10), 10)
    assert(np.isclose(x.getValue(), 1))
    assert(np.isclose(x.getDeriv()['x'], 1 / (np.log(10) * 10)))

    x = ad.log(2 * ad.Scalar('x', 10), 10)
    assert(np.isclose(x.getValue(), np.log(20) / np.log(10)))
    assert(np.isclose(x.getDeriv()['x'], 1 / (np.log(10) * 10)))

    x = ad.log(ad.Scalar('x', 10), 10)
    x2 = ad.log(ad.Scalar('x', 100), 10)
    x3 = x - x2
    x4 = ad.log(ad.Scalar('x', 10) / ad.Scalar('x', 100), 10)
    assert(np.isclose(x3.getValue(), -1) )
    assert(np.isclose(x3.getDeriv()['x'], 1 / (np.log(10) * 10) - 1 / (np.log(10) * 100)))
    assert(np.isclose(x3.getValue(), x4.getValue()))
    assert(np.isclose(x3.getDeriv()['x'], x4.getDeriv()['x']))

    x = ad.ln(ad.Scalar('x', np.e))
    assert(np.isclose(x.getValue(), 1))
    assert(np.isclose(x.getDeriv()['x'], 1 / np.e))

    x = ad.exp(ad.ln(ad.Scalar('x', 10)))
    assert(np.isclose(x.getValue(), 10))
    assert(np.isclose(x.getDeriv()['x'], 1)) 

    assert(np.isclose(ad.log(100, 10), 2))


def test_arcsin():
    x = ad.Scalar('x', 0.5);
    y = ad.arcsin(x);
    assert(np.isclose(y.getValue(), np.arcsin(0.5)));
    assert(np.isclose(y.getDeriv()['x'], 1 / np.sqrt(1-0.5**2)));
    
    y = ad.arcsin(-0.1);
    assert(np.isclose(y, np.arcsin(-0.1)) );

    #test function with two variables
    x = ad.Scalar('x', 0.5);
    y = ad.Scalar('y', -0.2);
    z = ad.arcsin(x * y);
    assert( np.isclose(z.getValue(), np.arcsin(0.5 * -0.2)) );
    assert( np.isclose(z.getDeriv()['x'], -0.2 * 1 / np.sqrt(1 - (-0.1)**2)) );
    assert( np.isclose(z.getDeriv()['y'], 0.5 * 1 / np.sqrt(1 - (-0.1)**2)) );
    

def test_arccos():
    x = ad.Scalar('x', 0.5);
    y = ad.arccos(x);
    assert(np.isclose(y.getValue(), np.arccos(0.5)));
    assert(np.isclose(y.getDeriv()['x'], -1 / np.sqrt(1-0.5**2)));
    
    y = ad.arccos(-0.1);
    assert(np.isclose(y, np.arccos(-0.1)));
    
    #test function with two variables
    x = ad.Scalar('x', 0.5);
    y = ad.Scalar('y', -0.2);
    z = ad.arccos(x * y);
    assert( np.isclose(z.getValue(), np.arccos(0.5 * -0.2)) );
    assert( np.isclose(z.getDeriv()['x'], -0.2 * -1 / np.sqrt(1 - (-0.1)**2)) );
    assert( np.isclose(z.getDeriv()['y'], 0.5 * -1 / np.sqrt(1 - (-0.1)**2)) );
    

def test_arctan():
    x = ad.Scalar('x', 0.5);
    y = ad.arctan(x);
    assert(np.isclose(y.getValue(), np.arctan(0.5)));
    assert(np.isclose(y.getDeriv()['x'], 1 / (1+0.5**2)));
    
    y = ad.arctan(-0.1);
    assert(np.isclose(y, np.arctan(-0.1)));
    
    #test function with two variables
    x = ad.Scalar('x', 0.5);
    y = ad.Scalar('y', -0.2);
    z = ad.arctan(x * y);
    assert( np.isclose(z.getValue(), np.arctan(0.5 * -0.2)) );
    assert( np.isclose(z.getDeriv()['x'], -0.2 * 1 / (1 + (-0.1)**2)) );
    assert( np.isclose(z.getDeriv()['y'], 0.5 * 1 / (1 + (-0.1)**2)) );