from os import environ, name as operating_system

if operating_system == "nt" and "VSCMD_VER" in environ:
    # Reuse the active Visual Studio developer environment. Without this flag,
    # PyTorch's build helper refuses to risk activating a second MSVC toolchain.
    environ.setdefault("DISTUTILS_USE_SDK", "1")

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


if operating_system == "nt":
    compile_args = ["/O2", "/std:c++20", "/Zc:preprocessor"]
else:
    compile_args = ["-O3", "-std=c++20"]


setup(
    name="kernelscope",
    version="0.1.0",
    description="Correctness-first CUDA transformer-kernel optimization study",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name="kernelscope._C",
            sources=["cpp/bindings.cpp", "cpp/cpu_reference.cpp"],
            extra_compile_args=compile_args,
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=True,
            no_python_abi_suffix=True,
        )
    },
    python_requires=">=3.10",
)
