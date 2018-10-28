import numpy as np

class Scalar():
    
    def __init__(self, variable, val):
        self._val = float(val) #ensures _val is numeric
        self._deriv = {variable: 1.0}
        
    def __add__(self, b):
        try:
            added = Scalar(None, self._val + b._val)
            added._deriv.pop(None, None)
            for variable in (set(self._deriv.keys()) | set(b._deriv.keys())):
                if variable not in self._deriv.keys():
                    added._deriv[variable] = b._deriv[variable]
                elif variable not in b._deriv.keys():
                    added._deriv[variable] = self._deriv[variable]
                else:
                    added._deriv[variable] = self._deriv[variable] + b._deriv[variable] 
                    
        except AttributeError:
            added = Scalar(None, self._val + b)
            added._deriv = self._deriv
        return added
    
    #might need to account for 0 cases?
    def __mul__(self, b):
        try:
            multiplied = Scalar(None, self._val * b._val)
            multiplied._deriv.pop(None, None)
            for variable in (set(self._deriv.keys()) | set(b._deriv.keys())):
                if variable not in self._deriv.keys():
                    multiplied._deriv[variable] = self._val * b._deriv[variable]
                elif variable not in b._deriv.keys():
                    multiplied._deriv[variable] = b._val * self._deriv[variable] 

                else:
                    multiplied._deriv[variable] = self._val * b._deriv[variable] + b._val * self._deriv[variable] 
            
        except AttributeError:
            multiplied = Scalar(None, self._val * b)
            multiplied._deriv.pop(None, None)
            for variable in self._deriv.keys():
                multiplied._deriv[variable] = b * self._deriv[variable]
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
    
    def getValue(self):
        return self._val
    
    def getDeriv(self):
        return self._deriv
    
    def getGradient(self, variables = None):
        return 
        
    __radd__ = __add__
    __rmul__ = __mul__



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
    