#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 14:35:05 2018

@author: bhavenpatel
"""

class Vector():
    
    def __init__(self, scalars):
        self._scalars = scalars
        self._derivs = [scalar.getDeriv() for scalar in scalars]
        variables = set()
        for deriv in self._derivs:
            variables = variables.union(deriv.keys())
        self._variables = list(variables)

    
    def __add__(self, b):
        try:
            #Add component wise. Ensures two vectors are same size
            added = Vector([scalar + b._scalars[i] for i, scalar in enumerate(self._scalars)])                     
        except AttributeError:
            added = Vector([scalar + b] for scalar in self._scalars)
        return added
    
    def __mul__(self, b):
        try:
            #Multiply component wise. Ensures two vectors are same size
            multiplied = Vector([scalar * b._scalars[i] for i, scalar in enumerate(self._scalars)])                     
        except AttributeError:
            multiplied = Vector([scalar * b] for scalar in self._scalars)
        return multiplied
    
    def __neg__(self):
        return
    
    def __sub__(self, b):
        return

    def __pow__(self, b):
        return
                
    def __rpow__(self, b):
        return 
    
    def __truediv__(self, b):
        return 
    
    def __rtruediv__(self, b):
        return 

    def __iadd__(self, b):
        return 
    
    def __isub__(self, b):
        return 
    
    def __imul__(self, b):
        return 
        
    def __ipow__(self, b):
        return 
        
    def __itruediv__(self, b):
        return
    
    def getValues(self):
        return 
    
    def getDerivs(self):
        return 
    
    #returns order of variables. 
    def getOrder(self):
        return self._variables

    def setOrder(self, new_order):
        assert (set(new_order) == set(self._variables))
        self._variables = new_order
    
    #Can choose to not get all variables
    def getGradient(self, variables = None):     
        return
    
    #This is equivalent to user not passing in any arguments to getGradient
    def getHessian(self):
        return self.getGradient()
    
    __radd__ = __add__
    __rmul__ = __mul__  
    