from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(name='FLmodel_f1_cpp',
	ext_modules=[
		CppExtension('FLmodel_f1_cpp', ['FLmodel_f1.cpp'],
			extra_compile_args=['-fopenmp','-O3','-msse4']
		),
	],
	cmdclass={'build_ext': BuildExtension})
