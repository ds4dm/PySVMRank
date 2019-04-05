from setuptools import setup, Extension
import numpy
import os, platform, sys, re

try:
    from Cython.Build import cythonize
except ImportError:
	print('Cython is required')
	quit(1)

# path to libs and headers
package_dir = os.path.join('src', 'svmrank')
svm_rank_sourcedir = os.path.join('src', 'c')
include_dirs = [package_dir, svm_rank_sourcedir, numpy.get_include()]
library_dirs = []
libraries = []

# version number
with open(os.path.join(package_dir, '__init__.py'), 'r') as initfile:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        initfile.read(), re.MULTILINE).group(1)

# set runtime libraries
runtime_library_dirs = []
extra_compile_args = ['-O3', '-fomit-frame-pointer', '-ffast-math', '-Wall']
extra_link_args = ['-lm', '-Wall']
if platform.system() in ['Linux', 'Darwin']:
    for libdir in library_dirs:
        extra_link_args.append(f'-Wl,-rpath,{libdir}')

extensions = [
	Extension('svmrank.svm_rank',
	[
	  os.path.join(package_dir, 'svm_rank.pyx'),
      os.path.join(package_dir, 'utilities.c'),
      os.path.join(svm_rank_sourcedir, 'svm_struct_api.c'),
      os.path.join(svm_rank_sourcedir, 'svm_struct_learn_custom.c'),
      os.path.join(svm_rank_sourcedir, 'svm_struct', 'svm_struct_common.c'),
      os.path.join(svm_rank_sourcedir, 'svm_struct', 'svm_struct_learn.c'),
      # os.path.join(svm_rank_sourcedir, 'svm_struct', 'svm_struct_main.c'),  # main() here
      # os.path.join(svm_rank_sourcedir, 'svm_struct', 'svm_struct_classify.c'),  # main() here
      os.path.join(svm_rank_sourcedir, 'svm_light', 'svm_common.c'),
      os.path.join(svm_rank_sourcedir, 'svm_light', 'svm_learn.c'),
      # os.path.join(svm_rank_sourcedir, 'svm_light', 'svm_learn_main.c'),  # main() here
      # os.path.join(svm_rank_sourcedir, 'svm_light', 'svm_classify.c'),  # main() here
      os.path.join(svm_rank_sourcedir, 'svm_light', 'svm_hideo.c'),
	],
	include_dirs=include_dirs,
	library_dirs=library_dirs,
	libraries=libraries,
	runtime_library_dirs=runtime_library_dirs,
	extra_compile_args=extra_compile_args,
	extra_link_args=extra_link_args
	)]

extensions = cythonize(extensions, compiler_directives={'language_level': 3})

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
    package_dir = {'svmrank': package_dir},
    package_data = {'svmrank': ['svm_rank.pyx', 'svm_rank.pxd', '*.pxi']}
)
