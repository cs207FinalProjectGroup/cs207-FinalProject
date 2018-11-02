#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 21:04:11 2018

@author: bhavenpatel
"""
import numpy as np
from autodiff.scalar import Scalar

def sin(sclr: Scalar):
    '''
    This function takes in a Scalar object, applies the sine function to its value, and returns a new Scalar object with the updated value and derivative.
    
    Parameters:
    scalar -- A Scalar object on which the sine function will applied.
    
    Returns:
    A Scalar object.
    '''
    
    result = Scalar(None, np.sin(sclr._val) ); #create new Scalar object with updated value
    result._deriv = sclr._deriv;
    #update derivatives for all of the variables in result by applying cos(deriv)
    for key in result._deriv.keys():
        d = result._deriv[key];
        result._deriv[key] = np.cos(d);
    return result;

def cos(sclr: Scalar):
    '''
    This function takes in a Scalar object, applies the cosine function to its value, and returns a new
    Scalar object with the updated value and derivative.
    
    Parameters:
    scalar -- A Scalar object on which the cosine function will applied.
    
    Returns:
    A Scalar object.
    '''
    
    result = Scalar(None, np.sin(sclr._val) ); #create new Scalar object with updated value
    result._deriv = sclr._deriv;
    #update derivatives for all of the variables in result by applying -sin(deriv)
    for key in result._deriv.keys():
        d = result._deriv[key];
        result._deriv[key] = -1 * np.sin(d);
    return result;

def tan(sclr: Scalar):
    '''
    This function takes in a Scalar object, applies the tangent function to its value, and returns a new Scalar object with the updated value and derivative.
    
    Parameters:
    scalar -- A Scalar object on which the cosine function will applied.
    
    Returns:
    A Scalar object.
    '''
    #return sin(Scalar) / cos(Scalar) since these functions are already implemented
    return sin(sclr) / cos(sclr);

def power(x, y):
    """Returns a Scalar object representing the operation x ** y, where 'x' and 'y' can be any combination of ints, floats, or Scalar objects. Calculation of new Scalar's value and derivations follow rules for exponents and power rule of differentiation respectively. 
    
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
    PRE: 
        - x is an int or float or Scalar
        - y is an int or
    POST:
        - 'x' and 'y' are not changed by the function
        - returns a scalar object, resulting from raising 'x' to the power of 'y'

    EXAMPLES
    =========
    >>> x = Scalar('x', 2)
    >>> y = Scalar('y', 3)
    >>> z = power(x, y)
    >>> z._val
    8.0
    >>> z._deriv
    {'x': 12.0, 'y': 5.545177444479561}

    """
    return x**y;
    
    