import ctypes
import numpy as np
import os

# Load the compiled shared library (.so file built by CMake)
# This is the bridge between Python and your CUDA kernels
_lib_path = os.path.join(os.path.dirname(__file__), "../build/libgpu_ops.so")
_lib = ctypes.CDLL(_lib_path)

# Tell Python the exact argument types for each function
# This must match the signatures in ops.h exactly
# ctypes.POINTER(ctypes.c_float) = float*
# ctypes.c_int                   = int

_lib.gpu_matmul.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # A
    ctypes.POINTER(ctypes.c_float),  # B
    ctypes.POINTER(ctypes.c_float),  # C (output)
    ctypes.c_int,                    # M
    ctypes.c_int,                    # K
    ctypes.c_int,                    # N
]
_lib.gpu_matmul.restype = None

_lib.gpu_relu.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # size
]
_lib.gpu_relu.restype = None

_lib.gpu_softmax.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # size
]
_lib.gpu_softmax.restype = None


def _ptr(arr):
    """Get a ctypes float pointer from a numpy array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def matmul(A, B):
    """
    GPU matrix multiplication.
    A: numpy array of shape (M, K)
    B: numpy array of shape (K, N)
    Returns C: numpy array of shape (M, N)
    """
    assert A.ndim == 2 and B.ndim == 2, "Inputs must be 2D arrays"
    assert A.shape[1] == B.shape[0], "A columns must match B rows"

    M, K = A.shape
    K2, N = B.shape

    # Ensure contiguous float32 arrays — ctypes requires this
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    _lib.gpu_matmul(_ptr(A), _ptr(B), _ptr(C), M, K, N)
    return C


def relu(x):
    """
    GPU ReLU activation.
    x: numpy array of any shape
    Returns output array of same shape
    """
    x_flat = np.ascontiguousarray(x.flatten(), dtype=np.float32)
    out = np.zeros_like(x_flat)
    _lib.gpu_relu(_ptr(x_flat), _ptr(out), len(x_flat))
    return out.reshape(x.shape)


def softmax(x):
    """
    GPU softmax.
    x: 1D numpy array
    Returns normalized probability array of same shape
    """
    assert x.ndim == 1, "Softmax input must be 1D"
    x_c = np.ascontiguousarray(x, dtype=np.float32)
    out = np.zeros_like(x_c)
    _lib.gpu_softmax(_ptr(x_c), _ptr(out), len(x_c))
    return out