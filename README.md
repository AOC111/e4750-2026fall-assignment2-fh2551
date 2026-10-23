# EECS E4750: Heterogeneous Computing for Signal and Data Processing (Fall 2026)

## Assignment-2: Kernel Fusion in PyCUDA & PyOpenCL — Layout Conversion with a Per-Channel Affine Transform

Due date: See in the courseworks.

Total points: 100

### <span style="color:red"><strong>TODO:</strong></span> (Re)naming of the student repository (for submitting the assignments)

***INSTRUCTIONS***
* You have to write the Code with thorough and clear comments, and write a report in PDF format which includes your plots, your profiler screenshots, and your answers to the theory questions.
* Students need to change and use the following name for the repository with their solutions: `e4750-2026fall-assignment2-UNI`
  * Good Example: `e4750-2026fall-assignment2-zz9999`
  * Bad example: `e4750-2026fall-assignments-e4750-2026fall-assign2-zz9999`
* This change can be done from the "Settings" tab which is located on the repository page.


### Primer

The goal of this assignment is to implement kernel fusion, and to use NVIDIA NSight Systems and NSight Compute to measure what fusion changes and what it does not.

Inference pipelines convert tensors between channel-major and batch-major layouts between stages. At inference time a BatchNorm layer reduces to a per-channel affine transform, because the running mean and variance are frozen after training; the trained parameters are folded on the host into two per-channel constants, `gamma` and `beta`.

The combined operation — convert the layout, then apply the per-channel scale and shift — can be written as two kernels or as one. The two-kernel version allocates an intermediate array in device global memory, writes it, and reads it back. The one-kernel version does not.

You will implement three versions of this operation in each API, benchmark them, and profile the PyCUDA versions with both profilers. NSight Systems shows what fusion does to the launch structure. NSight Compute shows what it does to the bytes moved.

The assignment should be completed in Google Cloud Hosted VMs only.

Please share any questions you come across on Ed Discussion for everyone to see.

### Relevant Documentation

For the profilers:
1. [NSight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
2. [NSight Compute User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
3. [nvprof to NSight Compute metric mapping](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvprof-metric-comparison)
4. [Tesla T4 hardware specification](https://www.techpowerup.com/gpu-specs/tesla-t4.c3316) — use the memory bandwidth figure when you interpret your achieved GB/s.

NSight Compute recitation session from the previous offering: [Slides](https://docs.google.com/presentation/d/1IIfkEgLrWg2hHBw2EWk1lFV9xAZiTqeyL4AcixYQMHM/edit?slide=id.g38407867325_0_11#slide=id.g38407867325_0_11)

Install the NSight Systems and NSight Compute GUIs on your own computer.

### **Submission Instructions on deliverables**

1. Complete the code templates provided in this folder for both PyOpenCL and PyCUDA. Keep the file names `pyopencl_assignment2.py` and `pycuda_assignment2.py` respectively.
    - All eight commands listed under **Running the code** must run without errors.
    - Every correctness test must print `PASS` for all three implementations in both files.
2. Create a report in PDF format which includes:
    - A screenshot of the correctness-test output (all `PASS`) for both files.
    - All four timing plots: `a2_bench_cuda.png`, `a2_ratio_cuda.png`, `a2_bench_opencl.png`, `a2_ratio_opencl.png`.
    - NSight Systems timeline screenshots of `unfused()` and `fused()`, both at `C = 256, N = 65536`, zoomed in far enough that the individual memcpy and kernel bars are legible. The `--stats=true` tables may be included as supporting evidence but do not replace the timeline screenshots.
    - NSight Compute screenshots covering the metrics listed in the profiling section: duration, DRAM bytes and store efficiency for all three kernels, plus throughput and load efficiency for `fused_kernel`.
    - A thorough analysis/comparison of the results shown in the plots.
    - Your answers to all six theory questions (A1, A2, B1, B2, C1, C2).
    - Report File Name: `Assignment2_Report_{First Name}_{Last Name}_{uni}.pdf`
    - Add the report into the github repo
3. Commit the four `.png` plots and your `.nsys-rep` and `.ncu-rep` profiler reports to your repository.
4. Go to Gradescope from the coursework menu to the correct assignment, and attach the repository corresponding to this assignment


## Programming Problem (80 points)

### Problem set up

| Symbol | Shape | dtype | Layout |
|---|---|---|---|
| `X`     | `(C, N)` | `float32` | channel-major: `C` channels, each of length `N` |
| `gamma` | `(C,)`   | `float32` | one scale per channel |
| `beta`  | `(C,)`   | `float32` | one shift per channel |
| `Y`     | `(N, C)` | `float32` | batch-major (the output) |

$$Y[n, c] = \gamma[c] \cdot X[c, n] + \beta[c]$$

Worked example (`C = 2`, `N = 3`):

```
X     = [[1, 2, 3],       gamma = [2, 0.5]      beta = [0, 1]
         [4, 5, 6]]

Y     = [[2,   3  ],
         [4,   3.5],
         [6,   4  ]]
```

Every array is row-major (C order, C-contiguous), as in C and in a default numpy array. The kernels receive flat pointers.

Read the `cpu()` reference implementation in the templates before writing any kernel code. It is one line of numpy and it states exactly what your GPU code must produce.

### The three kernels (required in both PyCUDA and PyOpenCL)

1. **`transpose_kernel`** — `Xt[n, c] = X[c, n]`. This kernel performs no arithmetic; it only moves data. One thread per element, boundary-checked.
2. **`affine_kernel`** — `Y[n, c] = gamma[c] * Xt[n, c] + beta[c]`. Reads the already-transposed matrix. Its input and output are both `(N, C)`, so it is an element-wise operation with a broadcast along the row.
3. **`fused_kernel`** — `Y[n, c] = gamma[c] * X[c, n] + beta[c]`. One kernel, with no intermediate array. The read index and the write index use different strides.

### Index mapping

The block / local work group size is fixed at **32 × 32** for every kernel. The mapping of threads onto `(c, n)` is not the same for all three kernels. Theory question B2 asks you to explain the difference.

| Kernel | dim 0 (`threadIdx.x` / `get_global_id(0)`) | dim 1 (`threadIdx.y` / `get_global_id(1)`) |
|---|---|---|
| `transpose_kernel` | `n` | `c` |
| `fused_kernel`     | `n` | `c` |
| `affine_kernel`    | `c` | `n` |

`C` and `N` may be any positive integers, including `1` and including values that are not multiples of 32. Every kernel needs a boundary check. In OpenCL the global work size must be an exact multiple of the local work size, so round up and let the boundary check discard the surplus work-items.

### The three host functions (required in both PyCUDA and PyOpenCL)

**1. `unfused_hostroundtrip(X, gamma, beta)`** — two kernels, with the intermediate `Xt` copied back to the host and then back to the device in between:

```
H2D:  X, gamma, beta
run:  transpose_kernel  ->  Xt on device
D2H:  Xt  ->  a host numpy array
H2D:  that host array  ->  a new device buffer
run:  affine_kernel     ->  Y on device
D2H:  Y   ->  host
```

The two `Xt` copies cancel out and have no effect on the result. This version exists so that the cost of a host round trip can be measured separately from the cost of the intermediate array itself.

**2. `unfused(X, gamma, beta)`** — two kernels, with `Xt` allocated on the device and kept there:

```
H2D:   X, gamma, beta
alloc: Xt on the device
run:   transpose_kernel  ->  Xt
run:   affine_kernel     ->  Y
free:  Xt
D2H:   Y  ->  host
```

`Xt` does not touch the host in this version. The host transfers in `unfused()` and `fused()` are byte-for-byte identical, so the difference between the two is device-side global memory traffic only: one extra `(N × C)` write, one extra `(N × C)` read, one extra device allocation and one extra kernel launch. Fusion in this assignment is not about host transfers, which is what `unfused_hostroundtrip()` is there to demonstrate.

**3. `fused(X, gamma, beta)`** — one kernel:

```
H2D:  X, gamma, beta
run:  fused_kernel  ->  Y
D2H:  Y  ->  host
```

### Implementation requirements

These fix the choices that would otherwise differ between submissions. Follow them exactly.

* **Device memory and transfers.**
  * PyCUDA: allocate every device array with `cuda.mem_alloc()` and move data with `cuda.memcpy_htod()` / `cuda.memcpy_dtoh()`. Do not use `gpuarray`.
  * PyOpenCL: allocate every device array as a `cl.Buffer`, and copy results back with `cl.enqueue_copy()`. Host-to-device data may be supplied either with the `COPY_HOST_PTR` flag at buffer creation or with `cl.enqueue_copy()`. Do not use `pyopencl.array`.
* **Launch configuration.** Every kernel uses the fixed 32 × 32 block / local work group size and the index mapping above, with one thread / work-item per element.
* **Kernel launches.** Each host function launches exactly the kernels in its sequence above, once each, and no others. `-m profile-ncu` relies on there being exactly three kernel launches in total.
* **`unfused_hostroundtrip()`.** Copy `Xt` into a host numpy array of shape `(N, C)`, free the original device `Xt` buffer, then copy the host array into a **new** device buffer. `affine_kernel` reads that new buffer.
* **`unfused()`.** `Xt` is device-only: it is never the source or destination of a host transfer. Free it after `affine_kernel` is launched and before `Y` is copied back.

### Timing requirements

Every host function returns `(Y, kernel_time, total_time)`:

* **`kernel_time`** — kernel duration only. For the two unfused versions this is the sum of the two kernel durations, each timed individually. No copies, no allocations.
* **`total_time`** — the entire sequence end to end: every allocation, every transfer, every kernel.

All times must be returned in **seconds**, NOT milliseconds. CUDA events' `time_till()` returns milliseconds; OpenCL `evt.profile` values are in nanoseconds. In PyCUDA use CUDA events for both measurements. In PyOpenCL use OpenCL event profiling for `kernel_time` and `time.time()` for `total_time`.

### Correctness tolerance

The test harness uses `np.allclose(y, expected, rtol=1e-5, atol=1e-6)`, not exact equality.

numpy evaluates `gamma * X.T` and then adds `beta` as two separately-rounded float32 operations. The device compiler may contract `gamma[c] * x + beta[c]` into a single fused multiply-add, which rounds once instead of twice. The two results can differ in the last bit or two, and this is expected. Do not disable FMA contraction with `-fmad=false` or `-cl-opt-disable`. A kernel that fails at this tolerance has an error in an index, a stride, or a boundary check.

### Running the code

PyCUDA, five modes:

```bash
python3 pycuda_assignment2.py -m test           # correctness suite
python3 pycuda_assignment2.py -m bench          # N sweep at C=128       -> a2_bench_cuda.png
python3 pycuda_assignment2.py -m ratio          # aspect-ratio sweep     -> a2_ratio_cuda.png
python3 pycuda_assignment2.py -m profile-nsys   # C=256, ladder of N, all three versions
python3 pycuda_assignment2.py -m profile-ncu    # C=256 N=65536, unfused + fused only
```

PyOpenCL, three modes:

```bash
python3 pyopencl_assignment2.py -m test         # correctness suite
python3 pyopencl_assignment2.py -m bench        # N sweep at C=128       -> a2_bench_opencl.png
python3 pyopencl_assignment2.py -m ratio        # aspect-ratio sweep     -> a2_ratio_opencl.png
```

All eight commands must run error-free. Profiling is required for the PyCUDA half only, matching the convention from previous assignments in this course.

The two sweeps:

* **`-m bench`** — `C` fixed at 128, a representative mid-depth CNN channel count, with `N` swept from 10 to 1,000,000. Produces a log-log plot with seven curves: all three versions in kernel-only and total form, plus the CPU baseline.
* **`-m ratio`** — the total element count `C*N` is held fixed at 2^24 and only the `C:N` ratio changes. Every `C` and every `N` in this sweep is a multiple of 32, so every work group is fully populated and no thread is discarded by a boundary check. Thread count and occupancy are constant across the sweep, so any variation in the plot is a memory effect rather than a thread-utilization effect. Theory question C1 is about this sweep.

### Task-1: PyCUDA (40 points)

| Item | Points |
|---|---|
| `transpose_kernel` correct and boundary-checked | 5 |
| `affine_kernel` correct and boundary-checked | 5 |
| `fused_kernel` correct and boundary-checked | 8 |
| `unfused_hostroundtrip()` implementing the full host round trip | 5 |
| `unfused()` with correct dual timing (kernel-only = sum of both kernels; total = everything) | 8 |
| `fused()` with correct dual timing | 4 |
| All correctness tests PASS, including every edge-case shape | 5 |

### Task-2: PyOpenCL (40 points)

Identical split to Task 1, for `pyopencl_assignment2.py`.

### Hints

1. Complete the PyCUDA code first, then the PyOpenCL code. The three OpenCL kernels are the same three kernels in a different dialect, and the three OpenCL host functions are the same three sequences with `cl.Buffer` in place of `mem_alloc`.
2. Read `cpu()` before writing kernel code. It is one line: `gamma[None, :] * X.T + beta[None, :]`.
3. Get `transpose_kernel`, `affine_kernel` and `unfused()` passing the correctness suite before writing `fused_kernel`.
4. Confusing `X`'s row length (`N`) with `Y`'s row length (`C`) is a common error. Write the flat index out for the `C=2, N=3` worked example before compiling.
5. Edge cases in the test suite include `C=1`, `N=1`, `C=37, N=53` (neither a multiple of 32), and both extreme aspect ratios.
6. In PyOpenCL the global work size must be an exact multiple of the local work size. Round up and rely on the in-kernel boundary check.
7. Synchronization: in PyCUDA, synchronize the event rather than the context, and note that `time_till()` returns milliseconds. In PyOpenCL, call `evt.wait()` before reading `evt.profile`, whose values are in nanoseconds.
8. `-m bench`, `-m ratio` and the top rung of the `-m profile-nsys` ladder allocate up to about 1 GiB per array, or roughly 3 GiB in total for `X`, `Xt` and `Y`. This fits on a T4 provided you free the device `Xt` buffer in `unfused()`. An out-of-memory error at the largest sizes usually means it is not being freed.


## Profiling (PyCUDA only)

Run NSight Systems first, then NSight Compute.

### Step 1 — NSight Systems

```bash
nsys profile --trace=cuda --stats=true -o a2_nsys python3 pycuda_assignment2.py -m profile-nsys
```

`--stats=true` prints the summary tables (`cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`, `cuda_api_sum`) to the terminal. Open `a2_nsys.nsys-rep` in the NSight Systems GUI for the timeline view; the timeline is required, and the summary tables alone are not sufficient.

The mode runs all three versions, in a fixed order, at each rung of a ladder of `N` (`N = 256, 4096, 65536, 1048576`, with `C` fixed at 256). One command produces one capture containing both the launch-overhead-dominated regime and the bandwidth-dominated regime. Per rung, the timeline reads left to right as:

```
[1] hostroundtrip : HtoD X -> transpose_kernel -> DtoH Xt -> HtoD Xt -> affine_kernel -> DtoH Y
                    6 GPU bars, 2 kernel launches
[2] unfused       : HtoD X -> transpose_kernel -> affine_kernel -> DtoH Y
                    4 GPU bars, 2 kernel launches
[3] fused         : HtoD X -> fused_kernel -> DtoH Y
                    3 GPU bars, 1 kernel launch
```

#### Checking `unfused()` on the timeline

The `unfused()` timeline should read HtoD, kernel, kernel, DtoH, with only a short dispatch gap between the two kernel bars. A memcpy between the two kernel bars means `unfused()` is copying `Xt` through the host, which is an implementation error. The correctness tests still pass in that case, so the timeline is the only place this shows up. Verify this before recording timing numbers, because the Task-3 comparisons assume `unfused()` performs no host transfer of `Xt`.

#### Required NSight Systems deliverables

1. Timeline screenshots of `unfused()` and `fused()`, both at `C = 256, N = 65536`, which is the same shape profiled in NSight Compute.
2. The kernel launch count for each of the three versions, read off the timeline or `cuda_gpu_kern_sum`.
3. The dispatch gap in `unfused()`: the interval between the end of the `transpose_kernel` bar and the start of the `affine_kernel` bar, measured off the timeline and reported in microseconds.

Two observations that the theory questions build on:

* Between `[1]` and `[2]`, a large amount of wall clock is removed and no kernel duration changes.
* In `[2]`, `transpose_kernel` performs no floating-point operations and still accounts for a substantial fraction of the GPU time. NSight Systems cannot report how many bytes were moved through DRAM, which is what Step 2 is for.

### Step 2 — NSight Compute

```bash
ncu --set full -o a2_ncu python3 pycuda_assignment2.py -m profile-ncu
```

Open `a2_ncu.ncu-rep` in the NSight Compute GUI.

`-m profile-ncu` runs only `unfused()` and `fused()`, once each, at `C = 256`, `N = 65536`. This produces exactly three kernel launches in the report — `transpose_kernel`, `affine_kernel`, `fused_kernel` — with no duplicate kernel names. `unfused_hostroundtrip()` is excluded because it launches the same two kernels a second time and its additional cost is in transfers, which is an NSight Systems measurement.

Collect the following:

| Metric | Collect for | Location in the NSight Compute GUI | CLI metric name |
|---|---|---|---|
| Kernel duration | all three kernels | Speed Of Light → *Duration* | `gpu__time_duration.sum` |
| DRAM bytes read / written | all three kernels | Memory Workload Analysis → *Device Memory* row | `dram__bytes_read.sum`, `dram__bytes_write.sum` |
| Global **store** efficiency | all three kernels | Memory Workload Analysis → *Global Store*, sectors per request | `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct` |
| Achieved memory throughput (GB/s) | `fused_kernel` only | Speed Of Light → *Memory Throughput* | `dram__bytes.sum.per_second` |
| Global **load** efficiency | `fused_kernel` only | Memory Workload Analysis → *Global Load*, sectors per request | `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` |

The last two are only required for `fused_kernel`. Load efficiency comes out close to 100% in all three kernels, so one reading is enough to establish the contrast with the store side.

For a faster targeted run instead of `--set full`:

```bash
ncu --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__bytes.sum.per_second,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct \
    python3 pycuda_assignment2.py -m profile-ncu
```

CLI metric names change between NSight Compute versions. If a metric name is rejected on your VM, use `ncu --query-metrics | grep <keyword>` to find the current name, and fall back on the GUI section named in the table above, since the section names are stable. State which version of `ncu` you used (`ncu --version`) in your report.


## Theory Problems (20 points)

Answer these using your own profiled numbers, timeline screenshots and plots.

The 20 points are divided between the two profilers, 8 each, plus 4 for the general questions.

### Part A — NSight Systems (8 points)

**A1. Launch structure across the three versions. (4 pts)**

Using your NSight Systems timeline at `C = 256, N = 65536`:

* State the kernel launch count for each of the three versions.
* Report the dispatch gap in `unfused()` — the interval between the end of the `transpose_kernel` bar and the start of the `affine_kernel` bar — in microseconds. This is the only number in this assignment you have to measure by hand off the timeline; everything else is printed or read from a table.
* Report the **total memcpy time** for each of the three versions (the sum of their HtoD and DtoH bars, available directly from `cuda_gpu_mem_time_sum`). A per-bar breakdown is not required.
* A large amount of time is removed between `unfused_hostroundtrip()` and `unfused()` while no kernel duration changes. State what was removed, and explain why the remaining difference between `unfused()` and `fused()` cannot have the same cause.

Include both required timeline screenshots with this answer, and confirm that your `unfused()` timeline has no memcpy between the two kernel bars.

**A2. Launch-overhead crossover. (4 pts)**

`-m profile-nsys` sweeps `N = 256, 4096, 65536, 1048576` at `C = 256` within a single capture.

Measure the difference between kernel-only time and total time at small `N` and at large `N`, for both `unfused()` and `fused()`. State the value of `N` at which the per-launch overhead is no longer visible relative to the kernel duration itself, give the numbers on either side of that point, and explain why the crossover exists: what is fixed, what scales with `N`, and what this implies about when fusion is worth doing.

### Part B — NSight Compute (8 points)

**B1. Predicted versus measured DRAM bytes. (4 pts)**

Before opening the profiler, write down the total DRAM bytes moved by the unfused pair and by the fused kernel in terms of `C` and `N`, counting each component separately (unfused: `X` read + `Xt` written + `Xt` read + `Y` written; fused: `X` read + `Y` written). Evaluate both at `C = 256`, `N = 65536`.

Compare these predictions against `dram__bytes_read.sum + dram__bytes_write.sum` from your report. State the predicted value, the measured value and the ratio, and explain the difference between prediction and measurement.

Also report the achieved memory throughput in GB/s for `fused_kernel`, and compare it against the T4's theoretical peak memory bandwidth.

**B2. Global store efficiency. (4 pts)**

Report global **store** efficiency for all three kernels, and global **load** efficiency for `fused_kernel`. `transpose_kernel` and `fused_kernel` should show similar store efficiency, and `affine_kernel` should differ from both.

* Explain this result at warp granularity. For `fused_kernel` under its required mapping (`n ← threadIdx.x`, `c ← threadIdx.y`), give the byte stride between the addresses touched by consecutive threads in a warp, for the read from `X` and for the write to `Y`. Do the same for `affine_kernel` under its mapping (`c ← threadIdx.x`), and explain why the two kernels use different mappings.
* In one paragraph, state what would have to change in order to improve the store efficiency. Refer to a shared-memory tiled transpose and describe what it provides. You do not implement it.
* State in one sentence what fusion changes, and what it does not change, about the transpose's access pattern.

### Part C — General (4 points)

**C1. Aspect-ratio sweep. (2 pts)**

From your `-m ratio` plot: `C*N`, thread count and occupancy are all constant, and every dimension is a multiple of 32. Describe how the fused kernel's runtime varies as `C` grows from 32 to 32768 and explain the mechanism behind what you observe. If the curve is flat, explain why it is flat.

**C2. Kernel fusion in general. (2 pts)**

Give the general definition of kernel fusion, and describe one real use case other than this assignment, stating what the fused version stops materializing in that case.


## Suggested workflow

1. Read `cpu()`.
2. Implement `transpose_kernel` and `affine_kernel`, and get `unfused()` and `unfused_hostroundtrip()` passing `-m test`.
3. Implement `fused_kernel` and get `fused()` passing `-m test`. All tests must print `PASS`.
4. Port everything to PyOpenCL and get `-m test` passing there.
5. Run `-m bench` and `-m ratio` for both APIs and save the four plots.
6. Run `-m profile-nsys` under `nsys`. Check that your `unfused()` timeline has nothing between the two kernel bars before continuing. Take the two required screenshots, count launches, measure the dispatch gap, and walk the `N` ladder for the crossover.
7. Write down your DRAM byte prediction before opening NSight Compute.
8. Run `-m profile-ncu` under `ncu` and collect the metrics listed in the profiling section. Compare the DRAM byte totals against your prediction.
9. Answer the theory questions using your own numbers.


## Code Templates

You **must** adhere to the templates given in the starter code — this is essential for all assignments to be graded fairly and equally. You are free to add helper functions and to modify comments, but the class structure, the kernel signatures, the function signatures/return values, `cpu()`, and the entire `TEST HARNESS` section must remain as provided.

Write your code **only** inside the blocks marked `STUDENT CODE STARTS HERE` / `STUDENT CODE ENDS HERE`, and inside the kernel bodies in `CUDA_CODE` / `KERNEL_CODE`. The docstring of each function states what it must do; the sections above state the required sequences, index mapping and implementation requirements.

#### PyCuda Starter Code
[pycuda_assignment2.py](./pycuda_assignment2.py)

#### PyOpenCL Starter Code
[pyopencl_assignment2.py](./pyopencl_assignment2.py)
