import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import gpu_ops

print("=" * 50)
print("Backward Pass — correctness tests")
print("=" * 50)


# ── Test 1: ReLU backward ─────────────────────────────────────────────────────
print("\n[1] ReLU backward")

input = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
dout  = np.array([ 1.0,  2.0, 3.0, 4.0, 5.0], dtype=np.float32)

gpu_result = gpu_ops.relu_backward(input, dout)
# Expected: gradient passes where input > 0, blocked elsewhere
expected   = np.array([0.0, 0.0, 0.0, 4.0, 5.0], dtype=np.float32)

print(f"  Input:    {input}")
print(f"  dout:     {dout}")
print(f"  GPU din:  {gpu_result}")
print(f"  Expected: {expected}")

if np.allclose(gpu_result, expected):
    print("  PASS ✓")
else:
    print("  FAIL ✗")
    sys.exit(1)


# ── Test 2: Softmax backward ──────────────────────────────────────────────────
print("\n[2] Softmax backward")

# Use numpy softmax as ground truth for the forward pass output
x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
e = np.exp(x - np.max(x))
softmax_out = (e / e.sum()).astype(np.float32)
dout = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

gpu_result = gpu_ops.softmax_backward(softmax_out, dout)

# CPU reference
dot = float(np.dot(dout, softmax_out))
expected = (softmax_out * (dout - dot)).astype(np.float32)

print(f"  softmax_out: {np.round(softmax_out, 4)}")
print(f"  dout:        {dout}")
print(f"  GPU din:     {np.round(gpu_result, 6)}")
print(f"  Expected:    {np.round(expected, 6)}")

if np.allclose(gpu_result, expected, atol=1e-5):
    print("  PASS ✓")
else:
    print("  FAIL ✗")
    sys.exit(1)


# ── Test 3: Matmul backward small ─────────────────────────────────────────────
print("\n[3] Matmul backward (small 2x2)")

A  = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
B  = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
dC = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

dA_gpu, dB_gpu = gpu_ops.matmul_backward(A, B, dC)

# numpy reference: dA = dC @ B.T,  dB = A.T @ dC
dA_expected = dC @ B.T
dB_expected = A.T @ dC

print(f"  dA GPU:\n{dA_gpu}")
print(f"  dA expected:\n{dA_expected}")
print(f"  dB GPU:\n{dB_gpu}")
print(f"  dB expected:\n{dB_expected}")

if np.allclose(dA_gpu, dA_expected, atol=1e-4) and \
   np.allclose(dB_gpu, dB_expected, atol=1e-4):
    print("  PASS ✓")
else:
    print("  FAIL ✗")
    sys.exit(1)


# ── Test 4: Matmul backward large ─────────────────────────────────────────────
print("\n[4] Matmul backward (256x256)")

np.random.seed(0)
A  = np.random.randn(256, 256).astype(np.float32)
B  = np.random.randn(256, 256).astype(np.float32)
dC = np.random.randn(256, 256).astype(np.float32)

dA_gpu, dB_gpu = gpu_ops.matmul_backward(A, B, dC)
dA_expected = dC @ B.T
dB_expected = A.T @ dC

max_diff_dA = np.max(np.abs(dA_gpu - dA_expected))
max_diff_dB = np.max(np.abs(dB_gpu - dB_expected))

print(f"  dA max diff vs numpy: {max_diff_dA:.6f}")
print(f"  dB max diff vs numpy: {max_diff_dB:.6f}")

if max_diff_dA < 0.05 and max_diff_dB < 0.05:
    print("  PASS ✓")
else:
    print("  FAIL ✗")
    sys.exit(1)


# ── Test 5: ReLU backward large ───────────────────────────────────────────────
print("\n[5] ReLU backward (1M elements)")

np.random.seed(1)
input = np.random.randn(1_000_000).astype(np.float32)
dout  = np.random.randn(1_000_000).astype(np.float32)

gpu_result  = gpu_ops.relu_backward(input, dout)
cpu_result  = np.where(input > 0, dout, 0.0)
max_diff    = np.max(np.abs(gpu_result - cpu_result))

print(f"  Max diff vs numpy: {max_diff:.6f}")

if max_diff < 1e-6:
    print("  PASS ✓")
else:
    print("  FAIL ✗")
    sys.exit(1)


print("\n" + "=" * 50)
print("All backward pass tests passed!")
print("=" * 50)