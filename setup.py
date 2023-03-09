from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
	
ext_modules=[Extension('int.aello',["int/aello.pyx"],libraries=["m"],extra_compile_args=["-ffast-math"])]
setup(ext_modules = cythonize(ext_modules ,language_level=3))


