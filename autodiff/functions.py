#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from autodiff.scalar import Scalar

def sin(sclr: Scalar):
    '''
    This function takes in a Scalar object, applies the sine function to its value, and returns a new Scalar object with the updated value and derivative.
    
    INPUTS
    ======= 
    scalar: A Scalar object on which the sine function will applied.
    
    RETURNS
    ========
    Scalar
    The new Scalar resulting from applying the sine function to  'sclr'.
    
    NOTES
    =====
    POST:
        - 'sclr' is not changed by the function
        - returns a Scalar object, resulting from applying the sine function to  'sclr'.

    EXAMPLES
    =========
    >>> x = Scalar('x', 2)
    >>> z = sin(x)
    >>> z._val
    0.9092974268256817
    >>> np.isclose(z._deriv['x'], -0.41614683654714241)
    True
    '''
    
    result = Scalar(None, np.sin(sclr._val) ); #create new Scalar object with updated value
    result._deriv = sclr._deriv.copy(); #result's derivative map is a copy of the passed in Scalar
    #update derivatives for all of the variables in result by applying cos(deriv)
    for key in result._deriv.keys():
        d = result._deriv[key];
        result._deriv[key] = np.cos(sclr._val) * d;
    return result;

def cos(sclr: Scalar):
    '''
    This function takes in a Scalar object, applies the cosine function to its value, and returns a new Scalar object with the updated value and derivative.
    
    INPUTS
    ======= 
    scalar: A Scalar object on which the cosine function will applied.
    
    RETURNS
    ========
    Scalar
    The new Scalar resulting from applying the cosine function to  'sclr'.
    
    NOTES
    =====
    POST:
        - 'sclr' is not changed by the function
        - returns a Scalar object, resulting from applying the cosine function to  'sclr'.

    EXAMPLES
    =========
    >>> x = Scalar('x', 2)
    >>> z = cos(x)
    >>> z._val
    -0.4161468365471424
    >>> np.isclose(z._deriv['x'], -0.90929742682568171)
    True
    '''
    
    result = Scalar(None, np.cos(sclr._val) ); #create new Scalar object with updated value
    result._deriv = sclr._deriv.copy(); #result's derivative map is a copy of the passed in Scalar
    #update derivatives for all of the variables in result by applying -sin(deriv)
    for key in result._deriv.keys():
        d = result._deriv[key];
        result._deriv[key] = -1 * np.sin(sclr._val) * d;
    return result;

def tan(sclr: Scalar):
    '''
    This function takes in a Scalar object, applies the tangent function to its value, and returns a new Scalar object with the updated value and derivative.
    
    INPUTS
    ======= 
    scalar: A Scalar object on which the tanget function will applied.
    
    RETURNS
    ========
    Scalar
    The new Scalar resulting from applying the tangent function to  'sclr'.
    
    NOTES
    =====
    POST:
        - 'sclr' is not changed by the function
        - returns a Scalar object, resulting from applying the tangent function to  'sclr'.

    EXAMPLES
    =========
    >>> x = Scalar('x', 2)
    >>> z = tan(x)
    >>> z._val
    -2.185039863261519
    >>> np.isclose(z._deriv['x'], 5.7743992040419174)
    True
    '''
    #return sin(Scalar) / cos(Scalar) since these functions are already implemented
    return sin(sclr) / cos(sclr);

def power(x, y):
    """Returns a Scalar object representing the operation x ** y, where 'x' and 'y' can be any combination of ints, floats, or Scalar objects. Calculation of new Scalar's derivatives follow rules for exponents and power rule of differentiation respectively. 
    
    INPUTS
    =======   
    x: int or float or Scalar
    The constant/Scalar base that we raise 'y' to the power of
    y: int or float or Scalar
    The exponent that 'x' is raised to

    RETURNS
    ========
    Scalar
    The new Scalar resulting from raising the base 'x' to the power of 'y'.

    NOTES
    =====
    POST:
        - 'x' and 'y' are not changed by the function
        - returns a Scalar object, resulting from raising 'x' to the power of 'y'

    EXAMPLES
    =========
    >>> a = Scalar('a', 3)
    >>> b = power(2.0, a)
    >>> b._val
    8.0
    >>> np.isclose(b._deriv['a'], 5.5451774444795623)
    True
    >>> x = Scalar('x', 2)
    >>> x = Scalar('x', 2)
    >>> y = Scalar('y', 3)
    >>> z = power(x, y)
    >>> z._val
    8.0
    >>> z._deriv['x']
    12.0
    >>> np.isclose(z._deriv['y'], 5.545177444479562)
    True
    """
    if isinstance(x, Scalar) or isinstance(y, Scalar):
        return x**y;
    else:
        return float(x**y);
    
def exp(sclr: Scalar):
    """Returns a Scalar object representing the operation e^(sclar), where 'sclr' is the current Scalar object. Calculations of new Scalar's derivatives follow the power rule of differentiation.

    INPUTS
    =======   
    sclr: Scalar object
    The Scalar we raise 'e' to the power of.

    RETURNS
    ========
    Scalar
    The new Scalar resulting from raising 'e' to the power of 'sclr'.

    NOTES
    =====
    POST:
        - 'sclr' is not changed by the function
        - returns a Scalar object, resulting from raising 'e' to the power of 'sclr'

    EXAMPLES
    =========
    >>> x = Scalar('x', 2)
    >>> y = exp(x)
    >>> y._val
    7.38905609893065
    >>> np.isclose(y._deriv['x'], 7.38905609893065)
    True
    """
    result = Scalar(None, np.exp(sclr._val) ); #create new Scalar object with value = e^val
    result._deriv = sclr._deriv.copy(); #result's derivative map is a copy of the passed in Scalar
    #update derivatives for all of the variables in result by applying e^(deriv)
    for key in result._deriv.keys():
        d = result._deriv[key];
        result._deriv[key] = result._val * d;
    return result;
        
    