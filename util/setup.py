from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy as np


setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension('fast_functions', sources=['fast_functions.pyx', 'predict_contour.c'], include_dirs=[np.get_include()], extra_compile_args=['-std=c99'])
    ]
)
