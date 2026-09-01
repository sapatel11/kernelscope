from os import environ, name as operating_system

if operating_system == "nt" and "VSCMD_VER" in environ:
    # Reuse the active Visual Studio developer environment. Without this flag,
    # PyTorch's build helper refuses to risk activating a second MSVC toolchain.
    environ.setdefault("DISTUTILS_USE_SDK", "1")

# Build specifically for the RTX 4060 / Ada target used by this project unless
# the caller explicitly overrides the architecture list.
environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


if operating_system == "nt":
    cxx_compile_args = ["/O2", "/std:c++20", "/Zc:preprocessor"]
    nvcc_compile_args = [
        "-O2",
        "-std=c++20",
        "-Xcompiler=/Zc:preprocessor",
    ]
else:
    cxx_compile_args = ["-O3", "-std=c++20"]
    nvcc_compile_args = ["-O3", "-std=c++20"]


setup(
    name="kernelscope",
    version="0.1.0",
    description="Correctness-first CUDA transformer-kernel optimization study",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="kernelscope._C",
            sources=[
                "cpp/bindings.cpp",
                "cpp/cpu_reference.cpp",
                "cuda/rmsnorm_naive.cu",
                "cuda/rmsnorm_fused.cu",
                "cuda/rmsnorm_vectorized.cu",
                "cuda/swiglu_fused.cu",
            ],
            extra_compile_args={
                "cxx": cxx_compile_args,
                "nvcc": nvcc_compile_args,
            },
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
