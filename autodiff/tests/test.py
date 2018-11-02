import os, sys
sys.path.append('../../..')

print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import autodiff as ad