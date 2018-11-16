import numpy as np
from autodiff.scalar import Scalar


def create_vector(vector_name, values):
    """
    Returns an array of Scalar containing the values
    with names derived from vector_name.

    INPUTS
    =======
    vector_name: string
    The constant or Scalar object we are multiplying the current Scalar object with

    values: list
    The values of the Scalar that will be in the output.

    RETURNS
    ========
    np.ndarray
    Returns the vector as a numpy array object.

    NOTES
    =====
    PRE:
         - vector_name is a string
         - values is a list of ints or floats
    POST:
         - vector_name is not changed by the function
         - values is not changed by the function

    EXAMPLES
    =========
    >>> w = create_vector('w', [2, 1, 3])
    >>> w[0]._val
    2.0
    >>> w[1].getDeriv()['w2']
    1.0
    """

    return np.array([Scalar("%s%i" % (vector_name, i), value)
                     for i, value in enumerate(values, 1)])
