from setuptools import setup, Extension
import numpy
import os, platform, sys, re

try:
    from Cython.Build import cythonize
except ImportError:
	print('Cython is required')
	quit(1)

# path to libs and headers
include_dirs = [os.path.join('src', 'svm_rank'), numpy.get_include()]
library_dirs = []
libraries = []
packagedir = os.path.join('src', 'cython')

# version number
with open(os.path.join(packagedir, '__init__.py'), 'r') as initfile:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        initfile.read(), re.MULTILINE).group(1)

# set runtime libraries
runtime_library_dirs = []
extra_compile_args = []
extra_link_args = []
if platform.system() in ['Linux', 'Darwin']:
    extra_link_args.append(','.join(['-Wl', '-rpath'] + library_dirs))

extensions = [
	Extension('svmrank.svm_rank',
	[
	  os.path.join(packagedir, 'svm_rank.pyx'),
	],
	include_dirs=include_dirs,
	library_dirs=library_dirs,
	libraries=libraries,
	runtime_library_dirs=runtime_library_dirs,
	extra_compile_args=extra_compile_args,
	extra_link_args=extra_link_args
	)]

extensions = cythonize(extensions)

with open('README.md') as f:
    long_description = f.read()

setup(
    name = 'PySVMRank',
    version = version,
    description = 'Python interface and modeling environment for SVMrank',
    long_description = long_description,
    url = 'https://github.com/ds4dm/PySVMRank',
    author = 'Maxime Gasse',
    author_email = 'maxime.gasse@polymtl.ca',
    license = 'MIT',
    ext_modules = extensions,
    packages = ['svmrank'],
    package_dir = {'svmrank': packagedir},
    package_data = {'svmrank': ['svm_rank.pyx', 'svm_rank.pxd', '*.pxi']}
)
