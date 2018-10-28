import numpy as np

class Scalar():
    
    def __init__(self, variable, val):
        self._val = float(val) #ensures _val is numeric
        self._deriv = {variable: 1.0}
        
    def __add__(self, b):
        try:
            added = Scalar(None, self._val + b._val); #create new Scalar object with None in the dictionary
            added._deriv.pop(None, None); # remove None from the derivatives dictionary
            for variable in (set(self._deriv.keys()) | set(b._deriv.keys())):
                if variable not in self._deriv.keys():
                    added._deriv[variable] = b._deriv[variable]
                elif variable not in b._deriv.keys():
                    added._deriv[variable] = self._deriv[variable]
                else:
                    added._deriv[variable] = self._deriv[variable] + b._deriv[variable] 
                    
        except AttributeError: #catches Exception if "b" is a int or float
            added = Scalar(None, self._val + b);
            added._deriv = self._deriv; #set derivative of new objec to old one
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
        negated = Scalar(None, -self._val)
        negated._deriv.pop(None, None)
        for variable in self._deriv.keys():
            negated._deriv[variable] = -self._deriv[variable]
        return negated
    
    def __sub__(self, b):
        return self + -b
    
    def __pow__(self, b):
        try:
            powered = Scalar(None, self._val ** b._val)
            powered._deriv.pop(None, None)
            for variable in (set(self._deriv.keys()) | set(b._deriv.keys())):
                # _derivative of x^y with respect to y (exponential rule)
                if variable not in self._deriv.keys():
                    powered._deriv[variable] = (self._val ** b._val) * np.log(self._val) * b._deriv[variable]
                # _derivative of x^y with respect to x (power rule)
                elif variable not in b._deriv.keys():
                    powered._deriv[variable] = b._val * (self._val ** (b._val - 1)) * self._deriv[variable] 
                # y = x ^ x 
                # Credits to http://mathcentral.uregina.ca/QQ/database/QQ.09.03/cher1.html for formula
                else:
                    powered._deriv[variable] = self._val * self._val * (np.log(self._val) + 1) * self._deriv[variable] 
            
        except AttributeError:
            powered = Scalar(None, self._val ** b)
            powered._deriv.pop(None, None)
            for variable in self._deriv.keys():
                powered._deriv[variable] = b * (self._val ** (b - 1)) * self._deriv[variable]
        return powered
        

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
        return self._val;
    
    def getDeriv(self):
        return self._deriv;
    
    def getGradient(self, variables = None):
        return 
        
    __radd__ = __add__
    __rmul__ = __mul__



