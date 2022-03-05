"""
" All the biometric routines used in
" the various applications
"""

import logging

from .biocore import extract_features, shift_to_CoM
from .background import cannybration, fingerfocus, backelcpp, show_bool, show_uint16
from .scores import *
from .utils import shift

#Silencing those pesky, bulky messages from matplotlib
logging.getLogger("matplotlib").setLevel(logging.WARNING)
