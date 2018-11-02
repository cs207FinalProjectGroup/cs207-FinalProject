import numpy as np

class Scalar():
    
    """
    Object that represents a scalar variable. 
    """

    def __init__(self, variable, val):  
        """The Scalar object contains two attributes, _val and _deriv, where the former is the value of the Scalar and the latter the derivatives.
        When a new Scalar is initialized. A user sets the name and value of the scalar during initialization. 
    
        INPUTS
        =======   
        variable: String
        The name of the Scalar object. Used as a key in ._deriv

        val: int or float 
        The value of the Scalar
        """
        self._val = float(val) #ensures _val is numeric
        self._deriv = {variable: 1.0}
        
    def __str__(self):
        """String representation of the Scalar object. Tells both the value and the derivatives."""
        return "Value: {0}, Derivatives: {1}".format(self._val, self._deriv);
    
    def __add__(self, b):
        """Returns a Scalar object representing the operation x + b, where x is the current Scalar object and b is either another Scalar object or a numeric value.
        Calculations of new Scalar's value and derivatives follow rules of addition and sum rule in differentiation respectively. Due to commutativity, x + b
        is equivalent to b + x, so __radd__ is equal to __add__.
        
        INPUTS
        =======   
        b: int or float or Scalar
        The constant or Scalar object we are adding to the current Scalar object with

        RETURNS
        ========
        Scalar
        The new Scalar resulting from adding the current Scalar with a constant or other Scalar

        NOTES
        =====
        PRE: 
             - b is an int or float or Scalar
        POST:
             - self is not changed by the function
             - b is not changed by the function
             - returns a scalar object, resulting from adding self and b

        EXAMPLES
        =========
        >>> x = Scalar('x', 2)
        >>> y = Scalar('y', 1)
        >>> z = x + y
        >>> z._val
        3.0
        >>> z._deriv['x']
        1.0
        >>> z._deriv['y']
        1.0
        >>> z = x + 1
        >>> z._val
        3.0
        >>> z._deriv
        {'x': 1.0}

        """
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
        """Returns a Scalar object representing the operation x * b, where x is the current Scalar object and b is either another Scalar object or a numeric value.
        Calculations of new Scalar's value and derivations follow rules of multiplication and product rule of differentiation respectively. 
        Due to commutativity, x * b is equivalent to b * x, so __rmul__ is equal to __mul__.

        INPUTS
        =======   
        b: int or float or Scalar
        The constant or Scalar object we are multiplying the current Scalar object with

        RETURNS
        ========
        Scalar
        The new Scalar resulting from multiplying the current Scalar with a constant or other Scalar

        NOTES
        =====
        PRE: 
             - b is an int or float or Scalar
        POST:
             - self is not changed by the function
             - b is not changed by the function
             - returns a scalar object, resulting from multiplying self and b

        EXAMPLES
        =========
        >>> x = Scalar('x', 2)
        >>> y = Scalar('y', 1)
        >>> z = x * y
        >>> z._val
        2.0
        >>> z._deriv['x']
        1.0
        >>> z._deriv['y']
        2.0
        >>> z = x * 2
        >>> z._val
        4.0
        >>> z._deriv
        {'x': 2.0}

        """
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
        """Negates both the value and the derivatives."""

        negated = Scalar(None, -self._val)
        negated._deriv.pop(None, None)
        for variable in self._deriv.keys():
            negated._deriv[variable] = -self._deriv[variable]
        return negated
    
    def __sub__(self, b):
        """Returns a Scalar object representing the operation x - b, where x is the current Scalar object and b is either another Scalar object or a numeric value.
        This is just adding the current Scalar with the negation of the other term.
        """
        return self + -b
    
    def __rsub__(self, b):
        """Returns a Scalar object representing the operation b - x, where x is the current Scalar object and b is either another Scalar object or a numeric value.
        This is just adding the negation of the current Scalar with the other term.
        """
        return b + -self

    def __pow__(self, b):
        """Returns a Scalar object representing the operation x ** b, where x is the current Scalar object and b is either another Scalar object or a numeric value.
        Calculations of new Scalar's value and derivations follow rules for exponents and power rule of differentiation respectively. 

        INPUTS
        =======   
        b: int or float or Scalar
        The constant/Scalar we raise the current Scalar to the power of 

        RETURNS
        ========
        Scalar
        The new Scalar resulting from raising the base (self) to the power of b 

        NOTES
        =====
        PRE: g
             - b is an int or float or Scalar
        POST:
             - self is not changed by the function
             - b is not changed by the function
             - returns a scalar object, resulting from raising self to the power of b

        EXAMPLES
        =========
        >>> x = Scalar('x', 2)
        >>> y = x ** 2
        >>> y._val
        4.0
        >>> y._deriv
        {'x': 4.0}

        """
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
        """Returns a Scalar object representing the operation c ** x, where x is the current Scalar object and c is either another Scalar object or a numeric value.
        Calculations of new Scalar's value and derivations follow rules for exponents and power rule of differentiation respectively. 

        INPUTS
        =======   
        b: int or float or Scalar
        The base that is raised to the power by the current Scalar

        RETURNS
        ========
        Scalar
        The new Scalar resulting from raising b to the power of self

        NOTES
        =====
        PRE: 
             - b is an int or float or Scalar
        POST:
             - self is not changed by the function
             - b is not changed by the function
             - returns a scalar object, resulting from raising self to the power of b

        EXAMPLES
        =========
        >>> x = Scalar('x', 2)
        >>> y = 2 ** x
        >>> y._val
        4.0
        >>> np.isclose(y._deriv['x'], 4 * np.log(2))
        True
        """
        powered = Scalar(None, b ** self._val);
        powered._deriv.pop(None, None); #get rid of None in the dictionary
        for variable in self._deriv.keys():
            powered._deriv[variable] = (b ** self._val) * np.log(b) * self._deriv[variable]
        return powered
    
    def __truediv__(self, b):
        """Returns a Scalar object representing the operation x / b, where x is the current Scalar object and b is either another Scalar object or a numeric value.
        This is just x multiplied by (b ** -1).
        """
        return self * (b ** -1)
    
    def __rtruediv__(self, b):
        """Returns a Scalar object representing the operation b / x, where x is the current Scalar object and b is either another Scalar object or a numeric value.
        This is just b multiplied by (x ** -1).
        """
        return b * (self ** -1)
    
    def __iadd__(self, b):
        """In place addition. Changes the values and derivatives of self directly."""
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
        """In place subtraction. Changes the values and derivatives of self directly."""
        self += -b
        return self
    
    def __imul__(self, b):
        """In place multiplication. Changes the values and derivatives of self directly."""
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
        """In place exponent. Changes the values and derivatives of self directly."""
#        original_self_val = self._val
#        try:
#            self._val **= b._val
#            for variable in (set(self._deriv.keys()) | set(b._deriv.keys())):
#                # _derivative of x^y with respect to y (exponential rule)
#                if variable not in self._deriv.keys():
#                    self._deriv[variable] = (original_self_val ** b._val) * np.log(original_self_val) * b._deriv[variable]
#                # _derivative of x^y with respect to x (power rule)
#                elif variable not in b._deriv.keys():
#                    self._deriv[variable] = b._val * (original_self_val ** (b._val - 1)) * self._deriv[variable] 
#                # y = x ^ x 
#                # Credits to http://mathcentral.uregina.ca/QQ/database/QQ.09.03/cher1.html for formula
#                else:
#                    self._deriv[variable] = original_self_val * original_self_val * (np.log(original_self_val) + 1) * self._deriv[variable] 
#
#        except AttributeError:
#            self._val **= b
#            for variable in self._deriv.keys():
#                self._deriv[variable] = b * (original_self_val ** (b - 1)) * self._deriv[variable]
        
        #just use __pow__ to compute- Bhaven
        result = self ** b;
        self._val = result._val;
        self._deriv = result._deriv;
        return self;
    
    def __itruediv__(self, b):
        """In place division. Changes the values and derivatives of self directly."""
        
        result = self / b; #use truediv to calculate the value and derivative
        self._val = result._val; #reassign the value of self
        self._deriv = result._deriv; #reassign the value of deriv
        return self;
    
    def getValue(self):
        """Returns the value of the scalar so that users does not access the value directly and potentially change it."""
        return self._val
    
    def getDeriv(self):
        """Returns the derivatives dictionary. Users can still potentially change it. Will resolve later. Maybe just return a copy."""
        return self._deriv
    
    def getGradient(self, variables):
        """Returns the derivatives as a numpy array, with the option to choose which specific partial derivatives to return.
         INPUTS
        =======   
        variables: list
        A list of strings corresponding to the variable names

        RETURNS
        ========
        derivs: numpy array
        The numpy array of partial derivatives 

        """
        derivs = []
        for variable in variables:
            derivs.append(self._deriv[variable])
        derivs = np.array(derivs)
        return derivs
        
    __radd__ = __add__
    __rmul__ = __mul__

