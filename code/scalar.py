import numpy as np

class Scalar():
    
    def __init__(self, variable, val):
        self._val = float(val) #ensures _val is numeric
        self._deriv = {variable: 1.0}
        
    def __str__(self):
        return "Value: {0}, Derivatives: {1}".format(self._val, self._deriv);
    
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
    
    def __rsub__(self, b):
        return b + -self
    
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
        powered = Scalar(None, self._val ** b)
        powered._deriv.pop(None, None)
        for variable in self._deriv.keys():
            powered._deriv[variable] = (b ** self._val) * np.log(b) * self._deriv[variable]
        return powered
    
    def __truediv__(self, b):
        return self * (b ** -1)
    
    def __rtruediv__(self, b):
        return b * (self ** -1)
    
    def __iadd__(self, b):
        try:
            self._val += b._val
            for variable in (set(self._deriv.keys()) | set(b._deriv.keys())):
                if variable not in self._deriv.keys():
                    self._deriv[variable] = b._deriv[variable]
                #_derivative is from self, so do not need to do anything
                elif variable not in b._deriv.keys():
                    continue
                else:
                    self._deriv[variable] += b._deriv[variable] 
                    
        except AttributeError:
            self._val += b
        return self

    def __isub__(self, b):
        self += -b
        return self
    
    def __imul__(self, b):
        try:
            #need original self._val for _derivative computations
            original_self_val = self._val
            self._val *= b._val 
            for variable in (set(self._deriv.keys()) | set(b._deriv.keys())):
                if variable not in self._deriv.keys():
                    self._deriv[variable] = original_self_val * b._deriv[variable]
                elif variable not in b._deriv.keys():
                    self._deriv[variable] = b._val * self._deriv[variable] 
                else:
                    self._deriv[variable] = original_self_val * b._deriv[variable] + b._val * self._deriv[variable] 

        except AttributeError:
            #technically can multiply string by number. Prevent this edge case from happening.
            self._val *= float(b)
            for variable in self._deriv.keys():
                self._deriv[variable] = b * self._deriv[variable]
        return self
    
    def __ipow__(self, b):
        original_self_val = self._val
        try:
            self._val **= b._val
            for variable in (set(self._deriv.keys()) | set(b._deriv.keys())):
                # _derivative of x^y with respect to y (exponential rule)
                if variable not in self._deriv.keys():
                    self._deriv[variable] = (original_self_val ** b._val) * np.log(self._val) * b._deriv[variable]
                # _derivative of x^y with respect to x (power rule)
                elif variable not in b._deriv.keys():
                    self._deriv[variable] = b._val * (original_self_val ** (b._val - 1)) * self._deriv[variable] 
                # y = x ^ x 
                # Credits to http://mathcentral.uregina.ca/QQ/database/QQ.09.03/cher1.html for formula
                else:
                    self._deriv[variable] = original_self_val * self._val * (np.log(self._val) + 1) * self._deriv[variable] 
        
        except AttributeError:
            self._val **= b
            for variable in self._deriv.keys():
                self._deriv[variable] = b * (original_self_val ** (b - 1)) * self._deriv[variable]
        return self
    
    def __itruediv__(self, b):
        self *= (b ** -1)
        return self
    
    def getValue(self):
        return self._val
    
    def getDeriv(self):
        return self._deriv
    
    def getGradient(self, variables = None):
        if variables is None:
            variables = self._deriv.keys()
        derivs = []
        for variable in variables:
            derivs.append(self._deriv[variable])
        derivs = np.array(derivs)
        return derivs
        
    __radd__ = __add__
    __rmul__ = __mul__