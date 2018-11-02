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
    This function takes in a Scalar object, applies the sine function to its value, and returns a new
    Scalar object with the updated value and derivative.
    
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
    
    
    