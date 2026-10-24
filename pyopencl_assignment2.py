"""
EECS E4750 Fall 2026 - Assignment 2 - PyOpenCL template

##############################################################################
#                        READ THIS BEFORE YOU START                          #
##############################################################################
# 1. ONLY write code inside the blocks marked:                               #
#        # ============ STUDENT CODE STARTS HERE ============                #
#        # ============= STUDENT CODE ENDS HERE =============                #
#    and inside the kernel bodies in KERNEL_CODE.                            #
#    DO NOT touch anything else: the kernel signatures, the function         #
#    signatures/return values, cpu(), and the entire TEST HARNESS            #
#    section are used for grading and must remain exactly as provided.       #
# 2. Make all times returned in SECONDS and NOT milliseconds.                #
#    (Careful: OpenCL event profiling info is in NANOSECONDS.)               #
# 3. Run modes:                                                             #
#        python3 pyopencl_assignment2.py -m test                             #
#        python3 pyopencl_assignment2.py -m bench                            #
#        python3 pyopencl_assignment2.py -m ratio                            #
#    All three must run error-free, and every correctness test must print    #
#    PASS. The bench and ratio modes save the plots for your report.         #
# 4. There are no profiling modes here. NSight Systems and NSight Compute    #
#    profiling is required for the PyCUDA half of this assignment ONLY,      #
#    matching the convention from previous assignments in this course.       #
##############################################################################

You must complete:
    - transpose_kernel body in KERNEL_CODE
    - affine_kernel    body in KERNEL_CODE
    - fused_kernel     body in KERNEL_CODE
    - clModule.unfused_hostroundtrip()
    - clModule.unfused()
    - clModule.fused()

Do the PyCUDA half first. The kernels here are the same three kernels in a
different dialect, and the host functions are the same three sequences with
Buffers instead of mem_alloc.

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

You implement three versions:

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

The local work group size is fixed at 32 x 32 for every kernel. The mapping of
work-items onto (c, n) is not the same for all three kernels; theory question
B2 asks you to explain the difference:

    transpose_kernel and fused_kernel
        dimension 0 indexes n,  dimension 1 indexes c

    affine_kernel
        dimension 0 indexes c,  dimension 1 indexes n

Remember that in OpenCL the global work size must be an exact multiple of the
local work size, so you round it UP and then rely on the boundary check inside
the kernel to discard the surplus work-items. Both C and N may be ANY positive
integer, including 1 and including values that are not multiples of 32.

------------------------------------------------------------------------------
A NOTE ON THE CORRECTNESS TOLERANCE
------------------------------------------------------------------------------
The tests use np.allclose(..., rtol=1e-5, atol=1e-6), not exact equality.
numpy evaluates gamma * X.T and then adds beta as two separately-rounded
float32 operations. The OpenCL compiler may contract `gamma[c] * x + beta[c]`
into a single fused multiply-add, which rounds once instead of twice, so the
two results can differ in the last bit or two. This is expected. Do not pass
-cl-opt-disable; a kernel that fails at this tolerance has an error in an
index, a stride, or a boundary check.
"""

import time
import argparse
import numpy as np
import pyopencl as cl
import pyopencl.array

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt


# ===================== OPENCL KERNELS ====================
# Do NOT change the kernel names or their parameters - only fill in the bodies.
KERNEL_CODE = """
// ===================== transpose_kernel =====================
// Xt[n, c] = X[c, n]
//
// This kernel performs no arithmetic; it only moves data.
//
// Required mapping (see the header): dimension 0 indexes n, dimension 1
// indexes c.
// One work-item handles exactly one element. Work-items outside the (C, N)
// domain must do nothing.
__kernel void transpose_kernel(
    __global float* Xt, __global const float* X,
    const unsigned int C, const unsigned int N
) {
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
    
    if (x<N && y<C){
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
// Required mapping (see the header): dimension 0 indexes c, dimension 1
// indexes n. Note that dimension 0 indexes c here, NOT n.
// One work-item handles exactly one element. Work-items outside the (C, N)
// domain must do nothing.
__kernel void affine_kernel(
    __global float* Y, __global const float* Xt,
    __global const float* gamma, __global const float* beta,
    const unsigned int C, const unsigned int N
) {
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
    
    if (x<C && y<N){
        Y[y*C+x] = Xt[y*C+x] * gamma[x] + beta[x];
    }
}


// ===================== fused_kernel =====================
// Y[n, c] = gamma[c] * X[c, n] + beta[c]
//
// The whole operation in one kernel. Xt is never allocated, written or read.
// Note that the read index and the write index use different strides.
//
// Required mapping (same as transpose_kernel): dimension 0 indexes n,
// dimension 1 indexes c.
// One work-item handles exactly one element. Work-items outside the (C, N)
// domain must do nothing.
__kernel void fused_kernel(
    __global float* Y, __global const float* X,
    __global const float* gamma, __global const float* beta,
    const unsigned int C, const unsigned int N
) {
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
    
    if (x<N && y<C){
        Y[x*C+y] = X[y*N+x] * gamma[y] + beta[y];
    }
}
"""


class clModule:
    def __init__(self):
        """
        **Do not modify this code**
        Attributes for an instance of clModule: the OpenCL context, a command
        queue with profiling enabled, and the compiled program. The local work
        group size is fixed at 32 x 32 for every kernel; you compute the global
        work size from C and N inside each function, using the mapping
        documented in the file header.
        """
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()

        self.ctx = cl.Context(devs)
        self.queue = cl.CommandQueue(
            self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        self.prg = cl.Program(self.ctx, KERNEL_CODE).build()
        self.local = (32, 32)

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
        separately from the cost of the intermediate array itself. In the
        PyCUDA half of this assignment you profile the equivalent function in
        NSight Systems.

        Requirements (these fix the choices; follow them exactly):
          - Device memory: cl.Buffer only, no pyopencl.array. Host-to-device
            data may be supplied with COPY_HOST_PTR at buffer creation or with
            cl.enqueue_copy(); results come back with cl.enqueue_copy().
          - Xt comes back into a host numpy array of shape (N, C). Release
            the original device Xt buffer, then copy the host array into a NEW
            buffer; affine_kernel reads that new buffer.
          - Launch transpose_kernel and affine_kernel once each, and no other
            kernel.

        Time the operation twice:
          - kernel_time: the SUM of the two kernel durations ONLY, from OpenCL
            event profiling (evt.profile.end - evt.profile.start, in
            NANOSECONDS). Nothing else -- no copies, no buffer creation.
          - total_time : the whole sequence above, end to end, including every
            buffer creation and every transfer, using time.time().

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
        global_size = (((N-1) // self.local[0] + 1)*self.local[0], ((C-1) // self.local[1] + 1)*self.local[1])

        start_time = time.time()
        mf = cl.mem_flags

        Xt = np.empty((N, C), dtype=np.float32)
        X_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
        Xt_buffer = cl.Buffer(self.ctx, mf.WRITE_ONLY, Xt.nbytes)
        gamma_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gamma)
        beta_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=beta)

        evt = self.prg.transpose_kernel(self.queue, global_size, self.local, Xt_buffer, X_buffer, C_u, N_u)
        evt.wait()

        kernel1_time = (evt.profile.end - evt.profile.start) * 1e-9

        cl.enqueue_copy(self.queue, Xt, Xt_buffer)
        X_buffer.release(), Xt_buffer.release()

        global_size = (((C-1) // self.local[0] + 1)*self.local[0], ((N-1) // self.local[1] + 1)*self.local[1])

        Y = np.empty((N, C), dtype=np.float32)
        Y_buffer = cl.Buffer(self.ctx, mf.WRITE_ONLY, Y.nbytes)
        Xt_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Xt)

        evt = self.prg.affine_kernel(self.queue, global_size, self.local, Y_buffer, Xt_buffer, gamma_buffer, beta_buffer, C_u, N_u)
        evt.wait()

        kernel2_time = (evt.profile.end - evt.profile.start) * 1e-9
        Xt_buffer.release()

        cl.enqueue_copy(self.queue, Y, Y_buffer)

        total_time = time.time() - start_time
        kernel_time = kernel1_time + kernel2_time
        # ============= STUDENT CODE ENDS HERE =============

        return Y, kernel_time, total_time

    # ------------------------------------------------------------------
    # Version 2: two kernels, intermediate stays on the device
    # ------------------------------------------------------------------
    def unfused(self, X, gamma, beta):
        """
        Compute Y[n, c] = gamma[c] * X[c, n] + beta[c] with TWO kernels, with
        the intermediate Xt allocated on the device and kept there.

            H2D:   X, gamma, beta
            alloc: Xt on the device
            run:   transpose_kernel  ->  Xt
            run:   affine_kernel     ->  Y
            free:  Xt
            D2H:   Y  ->  host

        Note that Xt does not touch the host in this version. The host
        transfers here and in fused() are byte-for-byte identical, so the
        difference between the two is device-side global memory traffic only:
        one extra (N x C) write, one extra (N x C) read, one extra device
        buffer and one extra kernel launch. unfused_hostroundtrip() is what
        demonstrates the host-transfer cost separately.

        Requirements (these fix the choices; follow them exactly):
          - Device memory: cl.Buffer only, no pyopencl.array. Host-to-device
            data may be supplied with COPY_HOST_PTR at buffer creation or with
            cl.enqueue_copy(); results come back with cl.enqueue_copy().
          - Xt is a buffer with no host pointer, and is never the source or
            destination of a transfer. Release it after affine_kernel is
            launched and before Y is copied back.
          - Launch transpose_kernel and affine_kernel once each, and no other
            kernel.

        Time the operation twice:
          - kernel_time: the SUM of the two kernel durations ONLY, from OpenCL
            event profiling.
          - total_time : end to end with time.time(), including the H2D
            transfers, the creation and release of the Xt buffer, both kernels,
            and the D2H transfer.

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
        global_size = (((N-1) // self.local[0] + 1)*self.local[0], ((C-1) // self.local[1] + 1)*self.local[1])

        start_time = time.time()
        mf = cl.mem_flags

        X_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
        gamma_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gamma)
        beta_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=beta)

        Xt_buffer = cl.Buffer(self.ctx, mf.READ_WRITE, X.nbytes)

        evt = self.prg.transpose_kernel(self.queue, global_size, self.local, Xt_buffer, X_buffer, C_u, N_u)
        evt.wait()

        kernel1_time = (evt.profile.end - evt.profile.start) * 1e-9
        X_buffer.release()

        global_size = (((C-1) // self.local[0] + 1)*self.local[0], ((N-1) // self.local[1] + 1)*self.local[1])

        Y = np.empty((N, C), dtype=np.float32)
        Y_buffer = cl.Buffer(self.ctx, mf.WRITE_ONLY, Y.nbytes)

        evt = self.prg.affine_kernel(self.queue, global_size, self.local, Y_buffer, Xt_buffer, gamma_buffer,
                                     beta_buffer, C_u, N_u)
        evt.wait()

        kernel2_time = (evt.profile.end - evt.profile.start) * 1e-9
        Xt_buffer.release()

        cl.enqueue_copy(self.queue, Y, Y_buffer)

        total_time = time.time() - start_time
        kernel_time = kernel1_time + kernel2_time
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
          - Device memory: cl.Buffer only, no pyopencl.array. Host-to-device
            data may be supplied with COPY_HOST_PTR at buffer creation or with
            cl.enqueue_copy(); results come back with cl.enqueue_copy().
          - Launch fused_kernel once, and no other kernel.

        Time the operation twice:
          - kernel_time: the fused kernel duration ONLY, from OpenCL event
            profiling.
          - total_time : end to end with time.time(), including both transfers
            and all buffer creation.

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
        global_size = (((N-1) // self.local[0] + 1)*self.local[0], ((C-1) // self.local[1] + 1)*self.local[1])

        start_time = time.time()
        mf = cl.mem_flags

        X_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
        gamma_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gamma)
        beta_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=beta)

        Y = np.empty((N, C), dtype=np.float32)
        Y_buffer = cl.Buffer(self.ctx, mf.WRITE_ONLY, Y.nbytes)

        evt = self.prg.fused_kernel(self.queue, global_size, self.local, Y_buffer, X_buffer, gamma_buffer,
                                     beta_buffer, C_u, N_u)
        evt.wait()

        kernel_time = (evt.profile.end - evt.profile.start) * 1e-9
        X_buffer.release()

        cl.enqueue_copy(self.queue, Y, Y_buffer)

        total_time = time.time() - start_time
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


def make_test_cases():
    """
    Build the correctness test suite: (name, X, gamma, beta) tuples.

    The shapes cover degenerate dimensions, dimensions that are not multiples
    of the 32 x 32 work group, and both aspect ratios (tall-and-thin and
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

    # Neither dimension a multiple of the 32 x 32 work group.
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
    print("Part 1: Correctness tests (PyOpenCL)")
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
              "looking at any timing number.")
    print()
    return all_passed


def run_bench(module):
    """
    Benchmark sweep: C fixed at 128 (a plausible mid-depth CNN channel count),
    N swept across orders of magnitude.
    """
    print("=" * 100)
    print("Part 2: Benchmark sweep (PyOpenCL), C fixed at 128")
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
    plt.title('Transpose + per-channel affine, C=128 (PyOpenCL)')
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
    plt.savefig('a2_bench_opencl.png', dpi=300)
    print("\nPlot saved as 'a2_bench_opencl.png' - include it in your report.")
    print()


def run_ratio(module):
    """
    Aspect-ratio sweep: the TOTAL element count C*N is held fixed, only the
    C:N ratio changes. Every configuration therefore moves exactly the same
    number of bytes and launches exactly the same number of work-items.

    Every C and every N below is a multiple of 32, so every 32 x 32 work group
    is fully populated in both dimensions and no work-item is discarded by a
    boundary check. Occupancy and work-item count are constant across the
    sweep, so any variation in this plot is a memory effect rather than a
    work-item-utilization effect. Theory question C1 is about this sweep.
    """
    print("=" * 100)
    print("Part 3: Aspect-ratio sweep (PyOpenCL), C*N held fixed at 2^24")
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
    plt.title('Same total elements (C*N = 2^24), varying aspect ratio (PyOpenCL)')
    plt.loglog(c_values, unf_k, 's-', label='unfused (kernel only)')
    plt.loglog(c_values, unf_t, 's--', label='unfused (total)')
    plt.loglog(c_values, fus_k, 'd-', label='fused (kernel only)')
    plt.loglog(c_values, fus_t, 'd--', label='fused (total)')
    plt.xlabel('C  (N = 2^24 / C)')
    plt.ylabel('Average runtime over %d runs (seconds)' % REPS)
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('a2_ratio_opencl.png', dpi=300)
    print("\nPlot saved as 'a2_ratio_opencl.png' - include it in your report.")
    print()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description="EECS E4750 Fall 2026 Assignment 2 - PyOpenCL: layout "
                    "conversion fused with a per-channel affine transform")
    arg_parser.add_argument(
        '-m', '--mode', required=True,
        choices=['test', 'bench', 'ratio'],
        help="test: correctness suite. bench: N sweep at C=128. "
             "ratio: aspect-ratio sweep at fixed C*N.")
    args = arg_parser.parse_args()

    module = clModule()

    if args.mode == 'test':
        run_tests(module)
    elif args.mode == 'bench':
        run_bench(module)
    elif args.mode == 'ratio':
        run_ratio(module)
