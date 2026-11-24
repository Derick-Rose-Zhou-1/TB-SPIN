import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


eigen_include = os.environ.get("SCPAS_EIGEN_INCLUDE", "/usr/include/eigen3")
if not os.path.exists(os.path.join(eigen_include, "Eigen", "Core")):
    raise RuntimeError("SCPAS_EIGEN_INCLUDE must point to Eigen headers")


setup(
    ext_modules=[
        Pybind11Extension(
            "scpas_pycompat.apml0.kernels._native",
            ["scpas_pycompat/apml0/kernels/_native.cpp"],
            include_dirs=[eigen_include],
            cxx_std=14,
            extra_compile_args=["-O2", "-DEIGEN_DONT_PARALLELIZE"],
        )
    ],
    cmdclass={"build_ext": build_ext},
)
