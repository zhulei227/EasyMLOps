__version__ = "0.1.7"

from .table.core import TablePipeLine
from .nlp.core import NLPPipeline
from .ts.core import TSPipeLine

import warnings

warnings.filterwarnings("ignore")
