import ctypes
import numpy as np
import os

_lib_path = os.path.join(os.path.dirname(__file__), "../build/libgpu_ops.so")
_lib = ctypes.CDLL(_lib_path)

# ── Forward pass bindings ─────────────────────────────────────────────────────

_lib.gpu_matmul.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
_lib.gpu_matmul.restype = None

_lib.gpu_relu.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]
_lib.gpu_relu.restype = None

_lib.gpu_softmax.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]
_lib.gpu_softmax.restype = None

# ── Backward pass bindings ────────────────────────────────────────────────────

_lib.gpu_matmul_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # A
    ctypes.POINTER(ctypes.c_float),  # B
    ctypes.POINTER(ctypes.c_float),  # dC
    ctypes.POINTER(ctypes.c_float),  # dA (output)
    ctypes.POINTER(ctypes.c_float),  # dB (output)
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
_lib.gpu_matmul_backward.restype = None

_lib.gpu_relu_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # dout
    ctypes.POINTER(ctypes.c_float),  # din (output)
    ctypes.c_int,
]
_lib.gpu_relu_backward.restype = None

_lib.gpu_softmax_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # softmax_out
    ctypes.POINTER(ctypes.c_float),  # dout
    ctypes.POINTER(ctypes.c_float),  # din (output)
    ctypes.c_int,
]
_lib.gpu_softmax_backward.restype = None


# ── Helper ────────────────────────────────────────────────────────────────────

def _ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# ── Forward pass API ──────────────────────────────────────────────────────────

def matmul(A, B):
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    K2, N = B.shape
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    _lib.gpu_matmul(_ptr(A), _ptr(B), _ptr(C), M, K, N)
    return C

def relu(x):
    x_flat = np.ascontiguousarray(x.flatten(), dtype=np.float32)
    out = np.zeros_like(x_flat)
    _lib.gpu_relu(_ptr(x_flat), _ptr(out), len(x_flat))
    return out.reshape(x.shape)

def softmax(x):
    assert x.ndim == 1
    x_c = np.ascontiguousarray(x, dtype=np.float32)
    out = np.zeros_like(x_c)
    _lib.gpu_softmax(_ptr(x_c), _ptr(out), len(x_c))
    return out


# ── Backward pass API ─────────────────────────────────────────────────────────

def matmul_backward(A, B, dC):
    """
    Given dC (gradient of loss w.r.t output C),
    returns (dA, dB) — gradients w.r.t A and B.
    """
    assert A.ndim == 2 and B.ndim == 2 and dC.ndim == 2
    M, K = A.shape
    K2, N = B.shape
    A   = np.ascontiguousarray(A,  dtype=np.float32)
    B   = np.ascontiguousarray(B,  dtype=np.float32)
    dC  = np.ascontiguousarray(dC, dtype=np.float32)
    dA  = np.zeros((M, K), dtype=np.float32)
    dB  = np.zeros((K, N), dtype=np.float32)
    _lib.gpu_matmul_backward(_ptr(A), _ptr(B), _ptr(dC),
                             _ptr(dA), _ptr(dB), M, K, N)
    return dA, dB

def relu_backward(input, dout):
    """
    Given original input and dout (gradient from above),
    returns din — gradient to pass to the layer below.
    """
    input_f = np.ascontiguousarray(input.flatten(), dtype=np.float32)
    dout_f  = np.ascontiguousarray(dout.flatten(),  dtype=np.float32)
    din     = np.zeros_like(input_f)
    _lib.gpu_relu_backward(_ptr(input_f), _ptr(dout_f), _ptr(din), len(input_f))
    return din.reshape(input.shape)

def softmax_backward(softmax_out, dout):
    """
    Given softmax output and dout (gradient from above),
    returns din — gradient to pass to the layer below.
    """
    assert softmax_out.ndim == 1 and dout.ndim == 1
    s    = np.ascontiguousarray(softmax_out, dtype=np.float32)
    d    = np.ascontiguousarray(dout,        dtype=np.float32)
    din  = np.zeros_like(s)
    _lib.gpu_softmax_backward(_ptr(s), _ptr(d), _ptr(din), len(s))
    return din