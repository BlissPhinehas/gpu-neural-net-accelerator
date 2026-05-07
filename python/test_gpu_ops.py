import numpy as np
import sys
import os

# Add the python folder to path so we can import gpu_ops
sys.path.insert(0, os.path.dirname(__file__))
import gpu_ops

print("=" * 50)
print("GPU Operations — ctypes wrapper test")
print("=" * 50)


# ── Test 1: ReLU ──────────────────────────────────────
print("\n[1] ReLU")
x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
result = gpu_ops.relu(x)
expected = np.array([0.0, 0.0, 0.0, 1.0, 3.0], dtype=np.float32)

print(f"  Input:    {x}")
print(f"  Output:   {result}")
print(f"  Expected: {expected}")

if np.allclose(result, expected):
    print("  PASS ✓")
else:
    print("  FAIL ✗")
    sys.exit(1)


# ── Test 2: Softmax ───────────────────────────────────
print("\n[2] Softmax")
x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
result = gpu_ops.softmax(x)

print(f"  Input:        {x}")
print(f"  Output:       {np.round(result, 4)}")
print(f"  Sum of output: {result.sum():.6f} (should be 1.0)")

if np.isclose(result.sum(), 1.0, atol=1e-5) and np.all(result >= 0):
    print("  PASS ✓")
else:
    print("  FAIL ✗")
    sys.exit(1)


# ── Test 3: Matrix multiply ───────────────────────────
print("\n[3] Matrix multiplication")
A = np.array([[1.0, 2.0],
              [3.0, 4.0]], dtype=np.float32)
B = np.array([[5.0, 6.0],
              [7.0, 8.0]], dtype=np.float32)

result   = gpu_ops.matmul(A, B)
expected = np.matmul(A, B)  # numpy as ground truth

print(f"  A:\n{A}")
print(f"  B:\n{B}")
print(f"  GPU result:\n{result}")
print(f"  Expected:\n{expected}")

if np.allclose(result, expected, atol=1e-4):
    print("  PASS ✓")
else:
    print("  FAIL ✗")
    sys.exit(1)


# ── Test 4: Large matrix correctness ─────────────────
print("\n[4] Large matrix correctness (512x512)")
np.random.seed(42)
A = np.random.randn(512, 512).astype(np.float32)
B = np.random.randn(512, 512).astype(np.float32)

gpu_result = gpu_ops.matmul(A, B)
cpu_result = np.matmul(A, B)

max_diff = np.max(np.abs(gpu_result - cpu_result))
print(f"  Max difference vs numpy: {max_diff:.6f}")

if max_diff < 0.01:
    print("  PASS ✓")
else:
    print("  FAIL ✗")
    sys.exit(1)


# ── Test 5: ReLU on large random array ───────────────
print("\n[5] ReLU large array (1M elements)")
x = np.random.randn(1_000_000).astype(np.float32)
gpu_result = gpu_ops.relu(x)
cpu_result = np.maximum(x, 0)

max_diff = np.max(np.abs(gpu_result - cpu_result))
print(f"  Max difference vs numpy: {max_diff:.6f}")

if max_diff < 1e-6:
    print("  PASS ✓")
else:
    print("  FAIL ✗")
    sys.exit(1)


print("\n" + "=" * 50)
print("All tests passed!")
print("=" * 50)