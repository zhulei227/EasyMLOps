__version__ = "0.2.1"

from .table.core import TablePipeLine
from .nlp.core import NLPPipeline
from .ts.core import TSPipeLine

from .yolo import YOLOPipeline
from .ocr import OCRPipeLine

import warnings

warnings.filterwarnings("ignore")
