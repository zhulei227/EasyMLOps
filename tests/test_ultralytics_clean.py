import sys

paths_to_remove = [
    r"C:\Users\Alei\AppData\Roaming\Python\Python310\site-packages",
    r"C:\Users\Alei\AppData\Roaming\Python\Python310"
]

for p in paths_to_remove:
    while p in sys.path:
        sys.path.remove(p)

import torch
print(f"torch file: {torch.__file__}")
print(f"torch version: {torch.__version__}")

import ultralytics
print(f'ultralytics version: {ultralytics.__version__}')
print('ultralytics OK')
