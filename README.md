# SYCL-Accelerated L-BFGS on Intel Arc GPUs

This repository demonstrates a **fully GPU-accelerated L-BFGS optimizer implemented in SYCL (DPC++)**, targeting Intel GPUs.  
The implementation is evaluated on standard nonlinear optimization benchmarks and compared against optimized CPU baselines.

The results show **order-of-magnitude speedups** on an Intel Arc discrete GPU, validating SYCL as a viable high-performance alternative to CUDA for numerical optimization workloads.

---

## Hardware & Software Configuration

### GPU (SYCL backend)
- **Device:** Intel® Arc™ B580  
- **Memory:** 12 GB VRAM  
- **Backend:** Level Zero  
- **Compiler:** Intel oneAPI DPC++ (`icpx`)  
- **Optimization flags:** `/O3 /fsycl`

### CPU Baseline
- **Processor:** Intel® Core™ i5-14600K  
- **Cores / Threads:** 14C / 20T  
- **Max Turbo Frequency:** 5.2 GHz  
- **Implementation:** Optimized CPU version (same algorithm)

---

## Benchmarks

All GPU results are **averaged over 10 runs**.  
Speedups are computed against **fixed CPU baseline runtimes**.

### CPU Baselines (ms)
- **Ackley:** 2039.391 ms  
- **Rastrigin:** 2326.843 ms  
- **Rosenbrock:** 57395.111 ms  

**Total CPU time (Ackley + Rastrigin + Rosenbrock):**  
**61,761.345 ms**

---

## Total Runtime & Speedup

| Version               | GPU Total Avg (ms) | Speedup vs CPU |
|-----------------------|--------------------|----------------|
| **lbfgs_sycl_opt.exe** | **4525.515**        | **13.65×**     |

---

## Per-Test Performance

### Ackley Function

| Version               | GPU Avg (ms) | Samples | CPU Baseline (ms) | Speedup |
|-----------------------|--------------|---------|-------------------|---------|
| **lbfgs_sycl_opt.exe** | **30.696**   | 10      | 2039.391          | **66.44×** |

---

### Quadratic Function

| Version               | GPU Avg (ms) | Samples | CPU Baseline | Speedup |
|-----------------------|--------------|---------|--------------|---------|
| **lbfgs_sycl_opt.exe** | **14.899**   | 10      | n/a          | n/a     |

> The Quadratic function is included primarily as a convergence and correctness sanity check.

---

### Rastrigin Function

| Version               | GPU Avg (ms) | Samples | CPU Baseline (ms) | Speedup |
|-----------------------|--------------|---------|-------------------|---------|
| **lbfgs_sycl_opt.exe** | **15.218**   | 10      | 2326.843          | **152.90×** |

---

### Rosenbrock Function

| Version               | GPU Avg (ms) | Samples | CPU Baseline (ms) | Speedup |
|-----------------------|--------------|---------|-------------------|---------|
| **lbfgs_sycl_opt.exe** | **4479.601** | 10      | 57395.111         | **12.81×** |

---

## Key Observations

- **Massive speedups on highly non-convex functions**  
  Rastrigin and Ackley benefit enormously from GPU parallelism.

- **Rosenbrock remains memory- and dependency-heavy**  
  Even so, a **~13× speedup** is achieved on a traditionally difficult benchmark.

- **SYCL overhead is amortized effectively**  
  Once kernel fusion, USM discipline, and synchronization are handled correctly, SYCL performance is competitive.

- **No CUDA required**  
  The entire pipeline runs on Intel hardware using open standards.

---

## Conclusion

This project demonstrates that:

- SYCL + Intel Arc GPUs can deliver **double-digit to triple-digit speedups** for numerical optimization  
- L-BFGS is a strong candidate for GPU acceleration when implemented carefully  
- Intel oneAPI provides a **production-ready toolchain** for serious compute workloads

---

## Reproducibility

To reproduce these results:

1. Use an Intel Arc GPU with Level Zero support
2. Compile with:
   ```bat
    icpx -std:c++20 -O3 -fsycl lbfgs_sycl_opt.cpp -o lbfgs_sycl.exe 
    set ONEAPI_DEVICE_SELECTOR=level_zero:gpu
    lbfgs_sycl.exe