"""
EECS E4750 Fall 2026 - Assignment 2 - PyCUDA template

##############################################################################
#                        READ THIS BEFORE YOU START                          #
##############################################################################
# 1. ONLY write code inside the blocks marked:                               #
#        # ============ STUDENT CODE STARTS HERE ============                #
#        # ============= STUDENT CODE ENDS HERE =============                #
#    and inside the kernel bodies in CUDA_CODE.                              #
#    DO NOT touch anything else: the kernel signatures, the function         #
#    signatures/return values, cpu(), and the entire TEST HARNESS            #
#    section are used for grading and must remain exactly as provided.       #
# 2. Make all times returned in SECONDS and NOT milliseconds.                #
#    (Careful: cuda events' time_till() returns MILLISECONDS.)               #
# 3. Run modes:                                                             #
#        python3 pycuda_assignment2.py -m test                               #
#        python3 pycuda_assignment2.py -m bench                              #
#        python3 pycuda_assignment2.py -m ratio                              #
#        python3 pycuda_assignment2.py -m profile-nsys                       #
#        python3 pycuda_assignment2.py -m profile-ncu                        #
#    All five must run error-free, and every correctness test must print     #
#    PASS. The bench and ratio modes save the plots for your report.         #
##############################################################################

You must complete:
    - transpose_kernel body in CUDA_CODE
    - affine_kernel    body in CUDA_CODE
    - fused_kernel     body in CUDA_CODE
    - cudaModule.unfused_hostroundtrip()
    - cudaModule.unfused()
    - cudaModule.fused()

------------------------------------------------------------------------------
THE PROBLEM
------------------------------------------------------------------------------
Inference pipelines convert tensors between channel-major and batch-major
layouts between stages. At inference time a BatchNorm layer reduces to a
per-channel affine transform, because the running mean and variance are frozen
after training; the trained parameters are folded on the host into two
per-channel constants, gamma[c] and beta[c].

The combined operation is:

    Input   X      : shape (C, N), float32, CHANNEL-major (C channels of N)
    Params  gamma  : shape (C,),   float32
            beta   : shape (C,),   float32
    Output  Y      : shape (N, C), float32, BATCH-major

    Y[n, c] = gamma[c] * X[c, n] + beta[c]

Worked example (C = 2, N = 3):

    X     = [[1, 2, 3],
             [4, 5, 6]]
    gamma = [2, 0.5]
    beta  = [0, 1]

    Y     = [[2,   3  ],
             [4,   3.5],
             [6,   4  ]]

This can be written as two kernels -- transpose X into an intermediate array
Xt, then apply the affine to Xt -- or as one. The two-kernel version costs an
extra write and an extra read of an (N, C) array that is never used again. The
one-kernel version produces the same Y without allocating the intermediate.

You implement three versions, and profile them with NSight Systems and NSight
Compute:

    1. unfused_hostroundtrip()  - two kernels, with Xt copied back to the host
                                  and then back to the device in between.
    2. unfused()                - two kernels, with Xt kept resident on the
                                  device. The host round trip is removed; the
                                  extra global memory round trip is not.
    3. fused()                  - one kernel, no intermediate.

------------------------------------------------------------------------------
INDEX MAPPING - THIS IS FIXED, AND YOU MUST FOLLOW IT
------------------------------------------------------------------------------
Every array is row-major (C order, C-contiguous), as it is in C and in a
default numpy array. The kernels receive flat pointers.

The block size is fixed at 32 x 32 for every kernel. The mapping of threads
onto (c, n) is not the same for all three kernels; theory question B2 asks you
to explain the difference:

    transpose_kernel and fused_kernel
        x dimension indexes n,  y dimension indexes c

    affine_kernel
        x dimension indexes c,  y dimension indexes n

Both C and N may be ANY positive integer, including 1 and including values that
are not multiples of 32. Every kernel needs a boundary check.

------------------------------------------------------------------------------
A NOTE ON THE CORRECTNESS TOLERANCE
------------------------------------------------------------------------------
The tests use np.allclose(..., rtol=1e-5, atol=1e-6), not exact equality.
numpy evaluates gamma * X.T and then adds beta as two separately-rounded
float32 operations. nvcc may contract `gamma[c] * x + beta[c]` into a single
fused multiply-add, which rounds once instead of twice, so the two results can
differ in the last bit or two. This is expected. Do not disable FMA
contraction; a kernel that fails at this tolerance has an error in an index, a
stride, or a boundary check.
"""

import time
import argparse
import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

"""
Date <10/25/2026>
Written by <Fanghao Hu> <fh2551@columbia.edu>
The code has not benn generated by AI tools, or copied from an external resources.
"""
# ===================== CUDA KERNELS ====================
# Do NOT change the kernel names or their parameters - only fill in the bodies.
CUDA_CODE = """
// ===================== transpose_kernel =====================
// Xt[n, c] = X[c, n]
//
// This kernel performs no arithmetic; it only moves data.
//
// Required mapping (see the header): x indexes n, y indexes c.
// One thread handles exactly one element. Threads outside the (C, N)
// domain must do nothing.
__global__ void transpose_kernel(
    float* Xt, const float* X,
    const unsigned int C, const unsigned int N
) {
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x<N && y<C){
        // Adjacent lanes read consecutive n values, while output rows are C elements apart.
        Xt[x*C+y] = X[y*N+x];
    }
}


// ===================== affine_kernel =====================
// Y[n, c] = gamma[c] * Xt[n, c] + beta[c]
//
// This kernel reads the ALREADY-TRANSPOSED matrix. Both its input and its
// output are (N, C), so this is a plain element-wise op with a broadcast
// along the row: every element in column c uses the same gamma[c], beta[c].
//
// Required mapping (see the header): x indexes c, y indexes n.
// Note that x indexes c here, NOT n.
// One thread handles exactly one element. Threads outside the (C, N)
// domain must do nothing.
__global__ void affine_kernel(
    float* Y, const float* Xt,
    const float* gamma, const float* beta,
    const unsigned int C, const unsigned int N
) {
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x<C && y<N){
        // Mapping x to c keeps both Xt reads and Y writes contiguous across a warp.
        Y[y*C+x] = Xt[y*C+x] * gamma[x] + beta[x];
    }
}


// ===================== fused_kernel =====================
// Y[n, c] = gamma[c] * X[c, n] + beta[c]
//
// The whole operation in one kernel. Xt is never allocated, written or read.
// Note that the read index and the write index use different strides.
//
// Required mapping (same as transpose_kernel): x indexes n, y indexes c.
// One thread handles exactly one element. Threads outside the (C, N)
// domain must do nothing.
__global__ void fused_kernel(
    float* Y, const float* X,
    const float* gamma, const float* beta,
    const unsigned int C, const unsigned int N
) {
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x<N && y<C){
        // The two strides preserve contiguous X reads while writing Y without an Xt buffer.
        Y[x*C+y] = X[y*N+x] * gamma[y] + beta[y];
    }
}
"""


class cudaModule:
    def __init__(self):
        """
        **Do not modify this code**
        The kernel code is compiled once here so it does not need to be
        recompiled for every call. The block size is fixed at 32 x 32 for
        every kernel; you compute the grid from C and N inside each function,
        using the mapping documented in the file header.
        """
        self.mod = SourceModule(CUDA_CODE)
        self.block = (32, 32, 1)

    # ------------------------------------------------------------------
    # Version 1: two kernels, with a host round trip for the intermediate
    # ------------------------------------------------------------------
    def unfused_hostroundtrip(self, X, gamma, beta):
        """
        Compute Y[n, c] = gamma[c] * X[c, n] + beta[c] with TWO kernels, and
        deliberately bring the intermediate Xt back to the host in between.

        The sequence you must implement is:

            H2D:  X, gamma, beta
            run:  transpose_kernel  ->  Xt on device
            D2H:  Xt  ->  a host numpy array
            H2D:  that host array  ->  a new device buffer
            run:  affine_kernel     ->  Y on device
            D2H:  Y   ->  host

        The two Xt copies cancel out and have no effect on the result. This
        version exists so that the cost of a host round trip can be measured
        separately from the cost of the intermediate array itself.

        Requirements (these fix the choices; follow them exactly):
          - Device memory: cuda.mem_alloc() only, no gpuarray. Transfers:
            cuda.memcpy_htod() / cuda.memcpy_dtoh() only.
          - Xt comes back into a host numpy array of shape (N, C). Free the
            original device Xt buffer, then copy the host array into a NEW
            device buffer; affine_kernel reads that new buffer.
          - Launch transpose_kernel and affine_kernel once each, and no other
            kernel.

        Time the operation twice, both with CUDA events:
          - kernel_time: the SUM of the two kernel durations ONLY. Nothing
            else -- no copies, no allocations.
          - total_time : the whole sequence above, end to end, including every
            allocation and every transfer.

        Arguments:
            X           :   2D numpy array of np.float32, shape (C, N),
                            C-contiguous
            gamma       :   1D numpy array of np.float32, shape (C,)
            beta        :   1D numpy array of np.float32, shape (C,)
        Returns:
            Y           :   2D numpy array of np.float32, shape (N, C)
            kernel_time :   summed kernel-only execution time, in seconds
            total_time  :   total end-to-end execution time, in seconds
        """
        C, N = X.shape
        C_u, N_u = np.uint32(C), np.uint32(N)

        # ============ STUDENT CODE STARTS HERE ============
        # Use separate event pairs so the host round trip contributes to total time but not kernel time.
        [total_start, total_end, kernel1_start, kernel1_end, kernel2_start, kernel2_end] = [cuda.Event() for _ in range(6)]

        transpose = self.mod.get_function("transpose_kernel")
        affine = self.mod.get_function("affine_kernel")

        grid_size = ((N-1) // self.block[0]+1, (C-1) // self.block[1]+1, 1)

        total_start.record()

        Xt = np.empty((N, C), dtype=np.float32)
        X_gpu = cuda.mem_alloc(X.nbytes)
        Xt_gpu = cuda.mem_alloc(Xt.nbytes)
        gamma_gpu = cuda.mem_alloc(gamma.nbytes)
        beta_gpu = cuda.mem_alloc(beta.nbytes)
        cuda.memcpy_htod(X_gpu, X)
        cuda.memcpy_htod(gamma_gpu, gamma)
        cuda.memcpy_htod(beta_gpu, beta)

        kernel1_start.record()
        transpose(Xt_gpu, X_gpu, C_u, N_u, block=self.block, grid=grid_size)
        kernel1_end.record()

        cuda.memcpy_dtoh(Xt, Xt_gpu)
        X_gpu.free(), Xt_gpu.free()

        grid_size = ((C-1) // self.block[1] + 1, (N-1) // self.block[0] + 1, 1)

        Y = np.empty((N, C), dtype=np.float32)
        Y_gpu = cuda.mem_alloc(Y.nbytes)
        # Reallocate Xt so affine reads a new buffer filled by the host round trip.
        Xt_gpu = cuda.mem_alloc(Xt.nbytes)
        cuda.memcpy_htod(Xt_gpu, Xt)


        kernel2_start.record()
        affine(Y_gpu, Xt_gpu, gamma_gpu, beta_gpu, C_u, N_u, block=self.block, grid=grid_size)
        kernel2_end.record()
        Xt_gpu.free()

        cuda.memcpy_dtoh(Y, Y_gpu)

        total_end.record()
        total_end.synchronize()

        total_time = total_end.time_since(total_start) * 1e-3
        # Sum the separately bracketed kernels so intermediate copies stay out of kernel_time.
        kernel_time = (kernel1_end.time_since(kernel1_start) + kernel2_end.time_since(kernel2_start)) * 1e-3
        # ============= STUDENT CODE ENDS HERE =============

        return Y, kernel_time, total_time

    # ------------------------------------------------------------------
    # Version 2: two kernels, intermediate stays on the device
    # ------------------------------------------------------------------
    def unfused(self, X, gamma, beta):
        """
        Compute Y[n, c] = gamma[c] * X[c, n] + beta[c] with TWO kernels, with
        the intermediate Xt allocated on the device and kept there.

            H2D:  X, gamma, beta
            alloc: Xt on the device
            run:  transpose_kernel  ->  Xt
            run:  affine_kernel     ->  Y
            free: Xt
            D2H:  Y  ->  host

        Note that Xt does not touch the host in this version. The host
        transfers here and in fused() are byte-for-byte identical, so the
        difference between the two is device-side global memory traffic only:
        one extra (N x C) write, one extra (N x C) read, one extra device
        allocation and one extra kernel launch. unfused_hostroundtrip() is
        what demonstrates the host-transfer cost separately.

        Requirements (these fix the choices; follow them exactly):
          - Device memory: cuda.mem_alloc() only, no gpuarray. Transfers:
            cuda.memcpy_htod() / cuda.memcpy_dtoh() only.
          - Xt is never the source or destination of a memcpy. Free it after
            affine_kernel is launched and before Y is copied back.
          - Launch transpose_kernel and affine_kernel once each, and no other
            kernel.

        Time the operation twice, both with CUDA events:
          - kernel_time: the SUM of the two kernel durations ONLY.
          - total_time : end to end, including the H2D copies, the device-side
            allocation and free of Xt, both kernels, and the D2H copy.

        Arguments:
            X           :   2D numpy array of np.float32, shape (C, N),
                            C-contiguous
            gamma       :   1D numpy array of np.float32, shape (C,)
            beta        :   1D numpy array of np.float32, shape (C,)
        Returns:
            Y           :   2D numpy array of np.float32, shape (N, C)
            kernel_time :   summed kernel-only execution time, in seconds
            total_time  :   total end-to-end execution time, in seconds
        """
        C, N = X.shape
        C_u, N_u = np.uint32(C), np.uint32(N)

        # ============ STUDENT CODE STARTS HERE ============
        [total_start, total_end, kernel1_start, kernel1_end, kernel2_start, kernel2_end] = [cuda.Event() for _ in range(6)]

        transpose = self.mod.get_function("transpose_kernel")
        affine = self.mod.get_function("affine_kernel")

        grid_size = ((N-1) // self.block[0]+1, (C-1) // self.block[1]+1, 1)

        total_start.record()

        Xt = np.empty((N, C), dtype=np.float32)
        X_gpu = cuda.mem_alloc(X.nbytes)
        # Keep Xt device-resident to isolate intermediate global-memory traffic from host transfers.
        Xt_gpu = cuda.mem_alloc(Xt.nbytes)
        gamma_gpu = cuda.mem_alloc(gamma.nbytes)
        beta_gpu = cuda.mem_alloc(beta.nbytes)
        cuda.memcpy_htod(X_gpu, X)
        cuda.memcpy_htod(gamma_gpu, gamma)
        cuda.memcpy_htod(beta_gpu, beta)

        Y = np.empty((N, C), dtype=np.float32)
        Y_gpu = cuda.mem_alloc(Y.nbytes)

        kernel1_start.record()
        transpose(Xt_gpu, X_gpu, C_u, N_u, block=self.block, grid=grid_size)
        kernel1_end.record()

        # The affine pass maps x to c, making neighboring lanes access adjacent Xt and Y values.
        grid_size = ((C-1) // self.block[1] + 1, (N-1) // self.block[0] + 1, 1)

        kernel2_start.record()
        affine(Y_gpu, Xt_gpu, gamma_gpu, beta_gpu, C_u, N_u, block=self.block, grid=grid_size)
        kernel2_end.record()
        X_gpu.free(),Xt_gpu.free()

        cuda.memcpy_dtoh(Y, Y_gpu)

        total_end.record()
        total_end.synchronize()

        total_time = total_end.time_since(total_start) * 1e-3
        kernel_time = (kernel1_end.time_since(kernel1_start) + kernel2_end.time_since(kernel2_start)) * 1e-3
        # ============= STUDENT CODE ENDS HERE =============

        return Y, kernel_time, total_time

    # ------------------------------------------------------------------
    # Version 3: one fused kernel
    # ------------------------------------------------------------------
    def fused(self, X, gamma, beta):
        """
        Compute Y[n, c] = gamma[c] * X[c, n] + beta[c] with ONE kernel.

            H2D:  X, gamma, beta
            run:  fused_kernel  ->  Y
            D2H:  Y  ->  host

        No intermediate array is allocated, written or read at any point.
        The host transfers are byte-for-byte identical to unfused(), so
        comparing the two isolates the cost of materializing Xt in device
        global memory.

        Requirements (these fix the choices; follow them exactly):
          - Device memory: cuda.mem_alloc() only, no gpuarray. Transfers:
            cuda.memcpy_htod() / cuda.memcpy_dtoh() only.
          - Launch fused_kernel once, and no other kernel.

        Time the operation twice, both with CUDA events:
          - kernel_time: the fused kernel duration ONLY.
          - total_time : end to end, including both transfers and all
            allocations.

        Arguments:
            X           :   2D numpy array of np.float32, shape (C, N),
                            C-contiguous
            gamma       :   1D numpy array of np.float32, shape (C,)
            beta        :   1D numpy array of np.float32, shape (C,)
        Returns:
            Y           :   2D numpy array of np.float32, shape (N, C)
            kernel_time :   kernel-only execution time, in seconds
            total_time  :   total end-to-end execution time, in seconds
        """
        C, N = X.shape
        C_u, N_u = np.uint32(C), np.uint32(N)

        # ============ STUDENT CODE STARTS HERE ============
        [total_start, total_end, kernel_start, kernel_end] = [cuda.Event() for _ in range(4)]

        fused = self.mod.get_function("fused_kernel")

        grid_size = ((N-1) // self.block[0]+1, (C-1) // self.block[1]+1, 1)

        total_start.record()

        Y = np.empty((N, C), dtype=np.float32)
        X_gpu = cuda.mem_alloc(X.nbytes)
        Y_gpu = cuda.mem_alloc(Y.nbytes)
        gamma_gpu = cuda.mem_alloc(gamma.nbytes)
        beta_gpu = cuda.mem_alloc(beta.nbytes)
        cuda.memcpy_htod(X_gpu, X)
        cuda.memcpy_htod(gamma_gpu, gamma)
        cuda.memcpy_htod(beta_gpu, beta)

        kernel_start.record()
        # This launch writes Y directly from X, avoiding both Xt and a second kernel launch.
        fused(Y_gpu, X_gpu, gamma_gpu, beta_gpu, C_u, N_u, block=self.block, grid=grid_size)
        kernel_end.record()

        cuda.memcpy_dtoh(Y, Y_gpu)

        total_end.record()
        total_end.synchronize()

        total_time = total_end.time_since(total_start) * 1e-3
        kernel_time = kernel_end.time_since(kernel_start) * 1e-3
        # ============= STUDENT CODE ENDS HERE =============

        return Y, kernel_time, total_time

    # ------------------------------------------------------------------
    # CPU reference
    # ------------------------------------------------------------------
    def cpu(self, X, gamma, beta):
        """
        **Do not modify this code**
        The numpy reference implementation. Read this before writing any
        kernel code; it states exactly what your three GPU versions must
        produce.

        Arguments:
            X       :   2D numpy array of np.float32, shape (C, N)
            gamma   :   1D numpy array of np.float32, shape (C,)
            beta    :   1D numpy array of np.float32, shape (C,)
        Returns:
            Y       :   2D numpy array of np.float32, shape (N, C)
            time    :   execution time in seconds
        """
        start = time.time()
        Y = gamma[None, :] * X.T + beta[None, :]
        end = time.time()
        return Y, end - start


###############################################################################
#                       TEST HARNESS - DO NOT MODIFY                          #
###############################################################################

# Tolerance for every correctness check in this file. See the note in the
# module docstring for why this is not exact equality.
RTOL = 1e-5
ATOL = 1e-6

# The fixed shape used by the NSight Compute mode, and the shape you take your
# required nsys timeline screenshots at. Bandwidth-bound and big enough that
# launch overhead is irrelevant: 256 * 65536 = 16,777,216 elements = 64 MiB
# per array.
PROFILE_C = 256
PROFILE_N = 65536

# The NSight Systems mode sweeps this ladder of N at C = PROFILE_C, so that a
# SINGLE nsys capture contains both the launch-overhead-dominated regime and
# the bandwidth-dominated regime. Note that PROFILE_N is one of the rungs, so
# your nsys and ncu numbers line up at that shape.
#   N=    256 ->     65,536 elements =  256 KiB per array
#   N=   4096 ->  1,048,576 elements =    4 MiB per array
#   N=  65536 -> 16,777,216 elements =   64 MiB per array   <-- PROFILE_N
#   N=1048576 -> 268,435,456 elements =   1 GiB per array
NSYS_N_LADDER = [256, 4096, 65536, 1048576]


def make_test_cases():
    """
    Build the correctness test suite: (name, X, gamma, beta) tuples.

    The shapes cover degenerate dimensions, dimensions that are not multiples
    of the 32 x 32 block, and both aspect ratios (tall-and-thin and
    short-and-wide). The gamma/beta choices isolate the shift and the scale
    independently.
    """
    rng = np.random.default_rng(4750)

    def rand(C, N):
        X = np.ascontiguousarray(rng.standard_normal((C, N)), dtype=np.float32)
        g = np.ascontiguousarray(rng.standard_normal(C), dtype=np.float32)
        b = np.ascontiguousarray(rng.standard_normal(C), dtype=np.float32)
        return X, g, b

    tests = []

    # The worked example straight out of the assignment text.
    tests.append((
        "worked_example_C2_N3",
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        np.array([2.0, 0.5], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ))

    # Degenerate shapes.
    tests.append(("C1_N1",       *rand(1, 1)))
    tests.append(("C1_N100003",  *rand(1, 100003)))
    tests.append(("C1024_N1",    *rand(1024, 1)))

    # Neither dimension a multiple of the 32 x 32 block.
    tests.append(("C37_N53",     *rand(37, 53)))
    tests.append(("C33_N1025",   *rand(33, 1025)))

    # Aspect ratio: tall-and-thin vs short-and-wide.
    tests.append(("C512_N7_tall",  *rand(512, 7)))
    tests.append(("C7_N512_wide",  *rand(7, 512)))

    # gamma = 1 everywhere, beta nonzero: isolates the SHIFT.
    C, N = 64, 1000
    X = np.ascontiguousarray(rng.standard_normal((C, N)), dtype=np.float32)
    tests.append((
        "shift_only_gamma1",
        X,
        np.ones(C, dtype=np.float32),
        np.ascontiguousarray(rng.standard_normal(C), dtype=np.float32),
    ))

    # beta = 0 everywhere, gamma varying: isolates the SCALE.
    tests.append((
        "scale_only_beta0",
        X,
        np.ascontiguousarray(rng.standard_normal(C), dtype=np.float32),
        np.zeros(C, dtype=np.float32),
    ))

    # A realistic inference-sized case.
    tests.append(("C128_N100000", *rand(128, 100000)))

    return tests


def run_tests(module):
    """Run the correctness suite against all three GPU implementations."""
    print("=" * 100)
    print("Part 1: Correctness tests (PyCUDA)")
    print(f"        tolerance: np.allclose(rtol={RTOL}, atol={ATOL})")
    print("=" * 100)

    all_passed = True
    for name, X, gamma, beta in make_test_cases():
        C, N = X.shape
        expected, _ = module.cpu(X, gamma, beta)

        y_hrt, _, _ = module.unfused_hostroundtrip(X, gamma, beta)
        y_unf, _, _ = module.unfused(X, gamma, beta)
        y_fus, _, _ = module.fused(X, gamma, beta)

        def check(y):
            return (y.shape == expected.shape) and np.allclose(
                y, expected, rtol=RTOL, atol=ATOL)

        ok_hrt, ok_unf, ok_fus = check(y_hrt), check(y_unf), check(y_fus)
        ok = ok_hrt and ok_unf and ok_fus
        all_passed = all_passed and ok

        print(f"[{'PASS' if ok else 'FAIL'}] {name:<22} C={C:>6,} N={N:>9,}   "
              f"hostroundtrip: {'PASS' if ok_hrt else 'FAIL'}   "
              f"unfused: {'PASS' if ok_unf else 'FAIL'}   "
              f"fused: {'PASS' if ok_fus else 'FAIL'}")

    print("-" * 100)
    if all_passed:
        print("All correctness tests PASSED.")
    else:
        print("Some correctness tests FAILED - fix your implementation before "
              "looking at any timing or profiler number.")
    print()
    return all_passed


def run_bench(module):
    """
    Benchmark sweep: C fixed at 128 (a plausible mid-depth CNN channel count),
    N swept across orders of magnitude.
    """
    print("=" * 100)
    print("Part 2: Benchmark sweep (PyCUDA), C fixed at 128")
    print("=" * 100)

    REPS = 5
    C = 128
    sizes = [10 ** k for k in range(1, 7)]  # N = 10 ... 1,000,000

    hrt_k, hrt_t = [], []
    unf_k, unf_t = [], []
    fus_k, fus_t = [], []
    cpu_t = []

    rng = np.random.default_rng(2026)
    for N in sizes:
        X = np.ascontiguousarray(rng.standard_normal((C, N)), dtype=np.float32)
        gamma = np.ascontiguousarray(rng.standard_normal(C), dtype=np.float32)
        beta = np.ascontiguousarray(rng.standard_normal(C), dtype=np.float32)

        a, b, c_, d, e, f, g = [], [], [], [], [], [], []
        for _ in range(REPS):
            _, kt, tt = module.unfused_hostroundtrip(X, gamma, beta)
            a.append(kt); b.append(tt)
            _, kt, tt = module.unfused(X, gamma, beta)
            c_.append(kt); d.append(tt)
            _, kt, tt = module.fused(X, gamma, beta)
            e.append(kt); f.append(tt)
            _, ct = module.cpu(X, gamma, beta)
            g.append(ct)

        hrt_k.append(np.average(a)); hrt_t.append(np.average(b))
        unf_k.append(np.average(c_)); unf_t.append(np.average(d))
        fus_k.append(np.average(e)); fus_t.append(np.average(f))
        cpu_t.append(np.average(g))

        print(f"N={N:>10,} | hostroundtrip {hrt_k[-1]:.3e}/{hrt_t[-1]:.3e}s "
              f"| unfused {unf_k[-1]:.3e}/{unf_t[-1]:.3e}s "
              f"| fused {fus_k[-1]:.3e}/{fus_t[-1]:.3e}s "
              f"| cpu {cpu_t[-1]:.3e}s      (kernel/total)")

    plt.figure(figsize=(8, 6))
    plt.title('Transpose + per-channel affine, C=128 (PyCUDA)')
    plt.loglog(sizes, hrt_k, 'o-', label='host round trip (kernel only)')
    plt.loglog(sizes, hrt_t, 'o--', label='host round trip (total)')
    plt.loglog(sizes, unf_k, 's-', label='unfused, device-only (kernel only)')
    plt.loglog(sizes, unf_t, 's--', label='unfused, device-only (total)')
    plt.loglog(sizes, fus_k, 'd-', label='fused (kernel only)')
    plt.loglog(sizes, fus_t, 'd--', label='fused (total)')
    plt.loglog(sizes, cpu_t, '^-', label='CPU (numpy)')
    plt.xlabel('N  (C fixed at 128)')
    plt.ylabel('Average runtime over %d runs (seconds)' % REPS)
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('a2_bench_cuda.png', dpi=300)
    print("\nPlot saved as 'a2_bench_cuda.png' - include it in your report.")
    print()


def run_ratio(module):
    """
    Aspect-ratio sweep: the TOTAL element count C*N is held fixed, only the
    C:N ratio changes. Every configuration therefore moves exactly the same
    number of bytes and launches exactly the same number of threads.

    Every C and every N below is a multiple of 32, so every 32 x 32 block is
    fully populated in both dimensions and no thread is discarded by a
    boundary check. Occupancy and thread count are constant across the sweep,
    so any variation in this plot is a memory effect rather than a
    thread-utilization effect. Theory question C1 is about this sweep.
    """
    print("=" * 100)
    print("Part 3: Aspect-ratio sweep (PyCUDA), C*N held fixed at 2^24")
    print("=" * 100)

    REPS = 5
    TOTAL = 2 ** 24  # 16,777,216 elements = 64 MiB per float32 array
    c_values = [32, 128, 512, 2048, 8192, 32768]

    unf_k, fus_k, unf_t, fus_t = [], [], [], []

    rng = np.random.default_rng(2026)
    for C in c_values:
        N = TOTAL // C
        X = np.ascontiguousarray(rng.standard_normal((C, N)), dtype=np.float32)
        gamma = np.ascontiguousarray(rng.standard_normal(C), dtype=np.float32)
        beta = np.ascontiguousarray(rng.standard_normal(C), dtype=np.float32)

        a, b, c_, d = [], [], [], []
        for _ in range(REPS):
            _, kt, tt = module.unfused(X, gamma, beta)
            a.append(kt); b.append(tt)
            _, kt, tt = module.fused(X, gamma, beta)
            c_.append(kt); d.append(tt)

        unf_k.append(np.average(a)); unf_t.append(np.average(b))
        fus_k.append(np.average(c_)); fus_t.append(np.average(d))

        print(f"C={C:>7,} N={N:>10,} | unfused kernel {unf_k[-1]:.3e}s "
              f"| fused kernel {fus_k[-1]:.3e}s "
              f"| unfused total {unf_t[-1]:.3e}s "
              f"| fused total {fus_t[-1]:.3e}s")

    plt.figure(figsize=(8, 6))
    plt.title('Same total elements (C*N = 2^24), varying aspect ratio (PyCUDA)')
    plt.loglog(c_values, unf_k, 's-', label='unfused (kernel only)')
    plt.loglog(c_values, unf_t, 's--', label='unfused (total)')
    plt.loglog(c_values, fus_k, 'd-', label='fused (kernel only)')
    plt.loglog(c_values, fus_t, 'd--', label='fused (total)')
    plt.xlabel('C  (N = 2^24 / C)')
    plt.ylabel('Average runtime over %d runs (seconds)' % REPS)
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('a2_ratio_cuda.png', dpi=300)
    print("\nPlot saved as 'a2_ratio_cuda.png' - include it in your report.")
    print()


def run_profile_nsys(module):
    """
    NSight Systems mode. Runs each of the three versions EXACTLY ONCE at each
    rung of NSYS_N_LADDER, with C fixed at PROFILE_C, in a fixed order.

    One capture, one report:

        nsys profile --trace=cuda --stats=true -o a2_nsys \\
            python3 pycuda_assignment2.py -m profile-nsys

    The ladder puts both regimes in a single timeline: at the smallest N the
    fixed per-launch overhead is comparable to the kernel durations, and at
    the largest N it is negligible beside them. Theory question A2 asks where
    the crossover between the two occurs.

    ------------------------------------------------------------------
    WHAT YOU SHOULD SEE, PER N, IN THIS ORDER
    ------------------------------------------------------------------
        [1] hostroundtrip : HtoD X -> transpose_kernel -> DtoH Xt ->
                            HtoD Xt -> affine_kernel -> DtoH Y
                            (6 GPU bars, 2 kernel launches)
        [2] unfused       : HtoD X -> transpose_kernel -> affine_kernel ->
                            DtoH Y
                            (4 GPU bars, 2 kernel launches)
        [3] fused         : HtoD X -> fused_kernel -> DtoH Y
                            (3 GPU bars, 1 kernel launch)

    ------------------------------------------------------------------
    CHECKING unfused() ON THE TIMELINE
    ------------------------------------------------------------------
    The [2] unfused timeline should read HtoD, kernel, kernel, DtoH, with only
    a short dispatch gap between the two kernel bars. A memcpy between the two
    kernel bars means unfused() is copying Xt through the host, which is an
    implementation error. The correctness tests still pass in that case, so
    the timeline is the only place this shows up. Verify it before recording
    timing numbers, because the Task-3 comparisons assume unfused() performs
    no host transfer of Xt.

    NSight Systems reports durations and launch counts, but not the number of
    bytes moved through DRAM. Use NSight Compute for the [2] vs [3] comparison.
    """
    print("=" * 100)
    print(f"NSight Systems profiling run: C={PROFILE_C}, "
          f"N ladder = {NSYS_N_LADDER}")
    print(f"Screenshot shape for your report: C={PROFILE_C}, N={PROFILE_N}")
    print("=" * 100)

    rng = np.random.default_rng(2026)
    gamma = np.ascontiguousarray(
        rng.standard_normal(PROFILE_C), dtype=np.float32)
    beta = np.ascontiguousarray(
        rng.standard_normal(PROFILE_C), dtype=np.float32)

    print(f"{'N':>10} {'version':<16} {'kernel(s)':>13} {'total(s)':>13} "
          f"{'total-kernel(s)':>16} {'launches':>9}")
    print("-" * 100)

    for N in NSYS_N_LADDER:
        X = np.ascontiguousarray(
            rng.standard_normal((PROFILE_C, N)), dtype=np.float32)

        _, k1, t1 = module.unfused_hostroundtrip(X, gamma, beta)
        _, k2, t2 = module.unfused(X, gamma, beta)
        _, k3, t3 = module.fused(X, gamma, beta)

        mark = "  <-- screenshot this shape" if N == PROFILE_N else ""
        print(f"{N:>10,} {'[1] hostroundtrip':<16} {k1:>13.6e} {t1:>13.6e} "
              f"{t1 - k1:>16.6e} {2:>9}")
        print(f"{'':>10} {'[2] unfused':<16} {k2:>13.6e} {t2:>13.6e} "
              f"{t2 - k2:>16.6e} {2:>9}")
        print(f"{'':>10} {'[3] fused':<16} {k3:>13.6e} {t3:>13.6e} "
              f"{t3 - k3:>16.6e} {1:>9}{mark}")
        print()

    print("The 'total-kernel' column is everything that is not kernel execution:")
    print("allocations, transfers, and per-launch dispatch overhead. Compare it")
    print("to the kernel time as N grows, and confirm these values against the")
    print("gaps on your nsys timeline.")
    print()


def run_profile_ncu(module):
    """
    NSight Compute mode. Runs ONLY unfused() and fused(), exactly once each,
    at the fixed profiling shape. That produces exactly three kernel launches
    in the report, with no duplicates to disambiguate:

        transpose_kernel, affine_kernel, fused_kernel

    unfused_hostroundtrip() is excluded because it launches the same two
    kernels a second time, which would produce duplicate entries, and because
    its additional cost is in transfers rather than kernels.

    Profile this with:

        ncu --set full -o a2_ncu python3 pycuda_assignment2.py -m profile-ncu

    then open a2_ncu.ncu-rep in the NSight Compute GUI. See the README for the
    exact metrics you must pull and report.
    """
    print("=" * 100)
    print(f"NSight Compute profiling run: C={PROFILE_C}, N={PROFILE_N}")
    print(f"        elements = {PROFILE_C * PROFILE_N:,}"
          f"   bytes per array = {PROFILE_C * PROFILE_N * 4:,}")
    print("=" * 100)

    rng = np.random.default_rng(2026)
    X = np.ascontiguousarray(
        rng.standard_normal((PROFILE_C, PROFILE_N)), dtype=np.float32)
    gamma = np.ascontiguousarray(
        rng.standard_normal(PROFILE_C), dtype=np.float32)
    beta = np.ascontiguousarray(
        rng.standard_normal(PROFILE_C), dtype=np.float32)

    _, k2, t2 = module.unfused(X, gamma, beta)
    print(f"unfused (transpose_kernel + affine_kernel): "
          f"kernel {k2:.6e}s   total {t2:.6e}s")
    _, k3, t3 = module.fused(X, gamma, beta)
    print(f"fused   (fused_kernel)                    : "
          f"kernel {k3:.6e}s   total {t3:.6e}s")
    print()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description="EECS E4750 Fall 2026 Assignment 2 - PyCUDA: layout "
                    "conversion fused with a per-channel affine transform")
    arg_parser.add_argument(
        '-m', '--mode', required=True,
        choices=['test', 'bench', 'ratio', 'profile-nsys', 'profile-ncu'],
        help="test: correctness suite. bench: N sweep at C=128. "
             "ratio: aspect-ratio sweep at fixed C*N. "
             "profile-nsys / profile-ncu: single fixed shape for the profilers.")
    args = arg_parser.parse_args()

    module = cudaModule()

    if args.mode == 'test':
        run_tests(module)
    elif args.mode == 'bench':
        run_bench(module)
    elif args.mode == 'ratio':
        run_ratio(module)
    elif args.mode == 'profile-nsys':
        run_profile_nsys(module)
    elif args.mode == 'profile-ncu':
        run_profile_ncu(module)
