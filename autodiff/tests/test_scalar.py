import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import autodiff as ad

"""
iadd
isub
imul
ipow
itruediv
radd
rmul
rsub
"""

print(dir(ad))