from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="combinations_filter",
        sources=["combinations_filter.pyx"],
        language="c++", 
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="combinations_filter",
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"} 
    ),
    zip_safe=False,
)
