"""
" All the biometric routines used in
" the various applications
"""

import logging

from .extraction_pipeline import run_pipeline
from .scores import *
from .distances import *
from .utils import shift, img_hist

#Silencing those pesky, bulky messages from matplotlib
logging.getLogger("matplotlib").setLevel(logging.WARNING)
