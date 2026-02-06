// lbfgs_sycl_optimized.cpp
// Major optimizations:
// - Persistent reduction buffers (no malloc/free per operation)
// - Removed unnecessary synchronization points
// - Fused kernels where possible
// - Better line search with Wolfe conditions
// - Improved memory access patterns

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>

static constexpr double PI = 3.141592653589793238462643383279502884;
static constexpr double E  = 2.718281828459045235360287471352662497;

// ----------------------- USM wrappers -----------------------

template <typename T>
struct GpuMatrix {
    int n, m;
    T* elems;
    sycl::queue* q;

    GpuMatrix(sycl::queue& qu, int n_, int m_) : n(n_), m(m_), elems(nullptr), q(&qu) {
        elems = (T*)sycl::malloc_device(sizeof(T) * (size_t)n * (size_t)m, qu);
        if (!elems) throw std::bad_alloc();
    }
    ~GpuMatrix() {
        if (elems) sycl::free(elems, *q);
    }
};

template <typename T>
struct GpuVector {
    int n;
    T* elems;
    sycl::queue* q;

    GpuVector(sycl::queue& qu, int n_) : n(n_), elems(nullptr), q(&qu) {
        elems = (T*)sycl::malloc_device(sizeof(T) * (size_t)n, qu);
        if (!elems) throw std::bad_alloc();
    }
    ~GpuVector() {
        if (elems) sycl::free(elems, *q);
    }
};

// ----------------------- Persistent reduction buffer -----------------------

struct ReductionBuffer {
    double* shared_buf;
    sycl::queue* q;

    ReductionBuffer(sycl::queue& qu) : q(&qu) {
        shared_buf = (double*)sycl::malloc_shared(sizeof(double), qu);
        if (!shared_buf) throw std::bad_alloc();
        *shared_buf = 0.0;
    }
    ~ReductionBuffer() {
        if (shared_buf) sycl::free(shared_buf, *q);
    }
};

// ----------------------- Optimized kernels -----------------------

template <typename T>
inline void setVectorScalar(sycl::queue& q, T* r, const T* x, T factor, int n) {
    q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> i) {
        r[i] = x[i] * factor;
    });
    // NO WAIT - let operations overlap
}

template <typename T>
inline void axpy(sycl::queue& q, T alpha, const T* x, T* y, int n) {
    q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> i) {
        y[i] += alpha * x[i];
    });
    // NO WAIT
}

// Optimized sum with persistent buffer
template <typename T>
double gpu_sum(sycl::queue& q, const T* d_in, int n, ReductionBuffer& rbuf) {
    double* out = rbuf.shared_buf;
    *out = 0.0;

    q.submit([&](sycl::handler& h) {
        auto red = sycl::reduction(out, sycl::plus<double>());
        h.parallel_for(sycl::range<1>((size_t)n), red,
                       [=](sycl::id<1> i, auto& sum) {
                           sum += (double)d_in[i];
                       });
    }).wait();

    return *out;
}

// Optimized dot with persistent buffer
template <typename T>
double dot(sycl::queue& q, const T* a, const T* b, int n, ReductionBuffer& rbuf) {
    double* out = rbuf.shared_buf;
    *out = 0.0;

    q.submit([&](sycl::handler& h) {
        auto red = sycl::reduction(out, sycl::plus<double>());
        h.parallel_for(sycl::range<1>((size_t)n), red,
                       [=](sycl::id<1> i, auto& sum) {
                           sum += (double)a[i] * (double)b[i];
                       });
    }).wait();

    return *out;
}

// Fused kernel: compute both function value and gradient terms
template <typename T>
void quadratic_fused(sycl::queue& q, T* d_x, T* d_g, T* d_temp, int n) {
    q.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> i) {
        T xi = d_x[i];
        d_temp[i] = xi * xi;
        d_g[i] = (T)2 * xi;
    });
}

// ----------------------- Tests (SYCL versions) -----------------------

template <typename T>
struct QuadraticTest {
    sycl::queue* q;
    T* d_temp_f;
    ReductionBuffer* rbuf;

    QuadraticTest(sycl::queue& qu, int n, ReductionBuffer& rb) 
        : q(&qu), d_temp_f(nullptr), rbuf(&rb) {
        d_temp_f = (T*)sycl::malloc_device(sizeof(T) * (size_t)n, qu);
        if (!d_temp_f) throw std::bad_alloc();
    }
    ~QuadraticTest() {
        if (d_temp_f) sycl::free(d_temp_f, *q);
    }

    double f(T* d_x, int n) {
        T* temp = d_temp_f;
        sycl::queue& qu = *q;

        qu.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> i) {
            T xi = d_x[i];
            temp[i] = xi * xi;
        }).wait();

        return gpu_sum(qu, temp, n, *rbuf);
    }

    void df(T* d_x, T* d_g, int n) {
        sycl::queue& qu = *q;
        qu.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> i) {
            T xi = d_x[i];
            d_g[i] = (T)2 * xi;
        });
    }
};

template <typename T>
struct RosenbrockTest {
    sycl::queue* q;
    T* d_temp_f;
    ReductionBuffer* rbuf;

    RosenbrockTest(sycl::queue& qu, int n, ReductionBuffer& rb) 
        : q(&qu), d_temp_f(nullptr), rbuf(&rb) {
        d_temp_f = (T*)sycl::malloc_device(sizeof(T) * (size_t)n, qu);
        if (!d_temp_f) throw std::bad_alloc();
    }
    ~RosenbrockTest() {
        if (d_temp_f) sycl::free(d_temp_f, *q);
    }

    double f(T* d_x, int n) {
        sycl::queue& qu = *q;
        T* temp = d_temp_f;

        qu.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> i) {
            temp[i] = (T)0;
        });

        if (n > 1) {
            qu.parallel_for(sycl::range<1>((size_t)(n - 1)), [=](sycl::id<1> i) {
                int idx = (int)i[0];
                T xi = d_x[idx];
                T xj = d_x[idx + 1];
                T t1 = xj - xi * xi;
                T t2 = (T)1 - xi;
                temp[idx] = (T)100 * t1 * t1 + t2 * t2;
            }).wait();
        }

        return gpu_sum(qu, temp, n, *rbuf);
    }

    void df(T* d_x, T* d_g, int n) {
        sycl::queue& qu = *q;

        if (n <= 0) return;

        qu.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> i) {
            int idx = (int)i[0];

            if (n == 1) {
                T x0 = d_x[0];
                d_g[0] = (T)(-2) * ((T)1 - x0);
                return;
            }

            T gi = (T)0;

            if (idx == 0) {
                T x0 = d_x[0], x1 = d_x[1];
                T t1 = x1 - x0 * x0;
                T t2 = (T)1 - x0;
                gi = (T)(-400) * x0 * t1 - (T)2 * t2;
            } else if (idx == n - 1) {
                T xm2 = d_x[n - 2];
                T xm1 = d_x[n - 1];
                gi = (T)200 * (xm1 - xm2 * xm2);
            } else {
                T xim1 = d_x[idx - 1];
                T xi   = d_x[idx];
                T xip1 = d_x[idx + 1];

                T t_left  = xi - xim1 * xim1;
                T t_right = xip1 - xi * xi;
                T t2      = (T)1 - xi;

                gi = (T)200 * t_left + (T)(-400) * xi * t_right - (T)2 * t2;
            }

            d_g[idx] = gi;
        });
    }
};

template <typename T>
struct RastriginTest {
    sycl::queue* q;
    T* d_temp_f;
    ReductionBuffer* rbuf;

    RastriginTest(sycl::queue& qu, int n, ReductionBuffer& rb) 
        : q(&qu), d_temp_f(nullptr), rbuf(&rb) {
        d_temp_f = (T*)sycl::malloc_device(sizeof(T) * (size_t)n, qu);
        if (!d_temp_f) throw std::bad_alloc();
    }
    ~RastriginTest() {
        if (d_temp_f) sycl::free(d_temp_f, *q);
    }

    double f(T* d_x, int n) {
        sycl::queue& qu = *q;
        T* temp = d_temp_f;
        const T two_pi = (T)(2.0 * PI);

        qu.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> i) {
            T xi = d_x[i];
            temp[i] = xi * xi - (T)10 * (T)sycl::cos(two_pi * xi);
        }).wait();

        return 10.0 * (double)n + gpu_sum(qu, temp, n, *rbuf);
    }

    void df(T* d_x, T* d_g, int n) {
        sycl::queue& qu = *q;
        const T two_pi = (T)(2.0 * PI);

        qu.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> i) {
            T xi = d_x[i];
            d_g[i] = (T)2 * xi + (T)(20.0 * PI) * (T)sycl::sin(two_pi * xi);
        });
    }
};

template <typename T>
struct AckleyTest {
    sycl::queue* q;
    T* d_temp_f;
    ReductionBuffer* rbuf;

    AckleyTest(sycl::queue& qu, int n, ReductionBuffer& rb) 
        : q(&qu), d_temp_f(nullptr), rbuf(&rb) {
        d_temp_f = (T*)sycl::malloc_device(sizeof(T) * (size_t)n, qu);
        if (!d_temp_f) throw std::bad_alloc();
    }
    ~AckleyTest() {
        if (d_temp_f) sycl::free(d_temp_f, *q);
    }

    double f(T* d_x, int n) {
        sycl::queue& qu = *q;
        T* temp = d_temp_f;
        const T two_pi = (T)(2.0 * PI);

        qu.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> i) {
            T xi = d_x[i];
            T ax = (T)sycl::fabs((double)xi);

            T tA = (T)(-20.0) * (T)sycl::exp((T)(-0.2) * ax);
            T c  = (T)sycl::cos(two_pi * xi);
            T tB = (T)(-1.0) * (T)sycl::exp(c);

            temp[i] = tA + tB + (T)20 + (T)E;
        }).wait();

        return gpu_sum(qu, temp, n, *rbuf);
    }

    void df(T* d_x, T* d_g, int n) {
        sycl::queue& qu = *q;
        const T two_pi = (T)(2.0 * PI);

        qu.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> i) {
            T xi = d_x[i];
            T ax = (T)sycl::fabs((double)xi);

            T sgn = (xi > (T)0) ? (T)1 : ((xi < (T)0) ? (T)-1 : (T)0);

            T gA = (T)4 * sgn * (T)sycl::exp((T)(-0.2) * ax);

            T s = (T)sycl::sin(two_pi * xi);
            T c = (T)sycl::cos(two_pi * xi);
            T gB = two_pi * s * (T)sycl::exp(c);

            d_g[i] = gA + gB;
        });
    }
};

// ----------------------- Improved LBFGS -----------------------

template <typename Func, typename T>
double lbfgs_sycl(sycl::queue& q, int n, int m, T* x0_host, int max_itr, 
                  Func& func, ReductionBuffer& rbuf, double eps = 1e-9) {
    GpuVector<T> x(q, n), g(q, n);
    GpuVector<T> x_old(q, n), g_old(q, n), d(q, n), qv(q, n), r(q, n);
    GpuMatrix<T> S(q, n, m), Y(q, n, m);

    std::vector<T> rho(m, (T)0);
    std::vector<T> alpha_hist(m, (T)0);

    q.memcpy(x.elems, x0_host, sizeof(T) * (size_t)n).wait();

    double val = func.f(x.elems, n);
    func.df(x.elems, g.elems, n);
    q.wait();

    int mem = 0;
    bool restart = false;

    for (int k = 0; k < max_itr; ++k) {
        double g_norm = std::sqrt(dot(q, g.elems, g.elems, n, rbuf));
        if (g_norm < eps) {
            std::cout << "Converged at iteration " << k << "\n";
            break;
        }

        // 1) direction
        if (restart || mem == 0) {
            restart = false;
            setVectorScalar(q, d.elems, g.elems, (T)-1, n);
        } else {
            q.memcpy(qv.elems, g.elems, sizeof(T) * (size_t)n);

            int bound = (mem < m) ? mem : m;

            for (int i = bound - 1; i >= 0; --i) {
                int idx = (k - 1 - i) % m;
                if (idx < 0) idx += m;

                if (rho[idx] == (T)0) { alpha_hist[idx] = (T)0; continue; }

                double s_dot_q = dot(q, S.elems + (size_t)idx * n, qv.elems, n, rbuf);
                alpha_hist[idx] = (T)((double)rho[idx] * s_dot_q);

                axpy(q, (T)(-alpha_hist[idx]), Y.elems + (size_t)idx * n, qv.elems, n);
            }

            int last_idx = (k - 1) % m; if (last_idx < 0) last_idx += m;
            double gamma = 1.0;

            if (rho[last_idx] != (T)0) {
                double s_dot_y = dot(q, S.elems + (size_t)last_idx * n, Y.elems + (size_t)last_idx * n, n, rbuf);
                double y_dot_y = dot(q, Y.elems + (size_t)last_idx * n, Y.elems + (size_t)last_idx * n, n, rbuf);
                gamma = (y_dot_y > 1e-18) ? (s_dot_y / y_dot_y) : 1.0;
            }

            setVectorScalar(q, r.elems, qv.elems, (T)gamma, n);

            for (int i = 0; i < bound; ++i) {
                int idx = (k - bound + i) % m;
                if (idx < 0) idx += m;
                if (rho[idx] == (T)0) continue;

                double y_dot_r = dot(q, Y.elems + (size_t)idx * n, r.elems, n, rbuf);
                double beta = (double)rho[idx] * y_dot_r;

                axpy(q, (T)(alpha_hist[idx] - (T)beta), S.elems + (size_t)idx * n, r.elems, n);
            }

            setVectorScalar(q, d.elems, r.elems, (T)-1, n);
        }

        q.wait(); // Ensure direction is ready

        // 2) Improved line search with Wolfe conditions
        double g_dot_d = dot(q, g.elems, d.elems, n, rbuf);

        if (g_dot_d >= 0.0) {
            restart = true;
            mem = 0;
            std::fill(rho.begin(), rho.end(), (T)0);
            setVectorScalar(q, d.elems, g.elems, (T)-1, n);
            q.wait();
            g_dot_d = dot(q, g.elems, d.elems, n, rbuf);
        }

        const double c1 = 1e-4;
        const double c2 = 0.9;
        const double tau = 0.5;
        double step = 1.0;
        bool success = false;

        q.memcpy(x_old.elems, x.elems, sizeof(T) * (size_t)n);
        q.memcpy(g_old.elems, g.elems, sizeof(T) * (size_t)n);
        q.wait();

        double f_old = val;
        double g_dot_d_old = g_dot_d;

        // Backtracking with strong Wolfe
        for (int ls = 0; ls < 40; ++ls) {
            q.memcpy(x.elems, x_old.elems, sizeof(T) * (size_t)n);
            axpy(q, (T)step, d.elems, x.elems, n);
            q.wait();

            double f_new = func.f(x.elems, n);

            // Armijo condition
            if (f_new <= f_old + c1 * step * g_dot_d_old) {
                // Check curvature (strong Wolfe)
                func.df(x.elems, g.elems, n);
                q.wait();
                
                double new_g_dot_d = dot(q, g.elems, d.elems, n, rbuf);
                
                if (std::fabs(new_g_dot_d) <= c2 * std::fabs(g_dot_d_old)) {
                    val = f_new;
                    success = true;
                    break;
                }
                
                // Armijo satisfied but not curvature - accept anyway if sufficient decrease
                if (f_new < f_old - 1e-10) {
                    val = f_new;
                    success = true;
                    break;
                }
            }

            step *= tau;
            if (step < 1e-20) break;
        }

        if (!success) {
            // More aggressive fallback
            step = 1e-3 / std::max(1.0, g_norm);
            q.memcpy(x.elems, x_old.elems, sizeof(T) * (size_t)n);
            axpy(q, (T)step, d.elems, x.elems, n);
            q.wait();
            
            double f_try = func.f(x.elems, n);
            if (f_try < val) {
                val = f_try;
                func.df(x.elems, g.elems, n);
                q.wait();
            } else {
                std::cout << "Line search failed at iteration " << k << ", terminating.\n";
                q.memcpy(x.elems, x_old.elems, sizeof(T) * (size_t)n);
                q.memcpy(g.elems, g_old.elems, sizeof(T) * (size_t)n);
                q.wait();
                break;
            }
            
            restart = true;
            mem = 0;
            std::fill(rho.begin(), rho.end(), (T)0);
            continue;
        }

        // 3) Update history
        int cur = k % m;

        q.memcpy(S.elems + (size_t)cur * n, x.elems, sizeof(T) * (size_t)n);
        axpy(q, (T)-1, x_old.elems, S.elems + (size_t)cur * n, n);

        q.memcpy(Y.elems + (size_t)cur * n, g.elems, sizeof(T) * (size_t)n);
        axpy(q, (T)-1, g_old.elems, Y.elems + (size_t)cur * n, n);
        q.wait();

        double sy = dot(q, S.elems + (size_t)cur * n, Y.elems + (size_t)cur * n, n, rbuf);
        if (sy > 1e-12) {
            rho[cur] = (T)(1.0 / sy);
            if (mem < m) mem++;
        } else {
            rho[cur] = (T)0;
        }
    }

    q.memcpy(x0_host, x.elems, sizeof(T) * (size_t)n).wait();
    return val;
}

// ----------------------- main -----------------------

int main() {
    sycl::async_handler asyncHandler = [](sycl::exception_list el) {
        for (auto& e : el) {
            try {
                std::rethrow_exception(e);
            } catch (const sycl::exception& ex) {
                std::cerr << "SYCL async exception: " << ex.what() << "\n";
                std::terminate();
            }
        }
    };

    sycl::queue q(
        sycl::default_selector{},
        asyncHandler,
        sycl::property_list{ sycl::property::queue::in_order() }
    );

    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n\n";

    // Create persistent reduction buffer
    ReductionBuffer rbuf(q);

    using T = float;
    int N = 1 << 12; // 4 million dimensions
    int M = 10;

    std::vector<T> x0(N, (T)0);

    // --- Quadratic ---
    std::fill(x0.begin(), x0.end(), (T)8);
    QuadraticTest<T> quad(q, N, rbuf);
    std::cout << "Starting: Quadratic Test (N=" << N << ")...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    double f_quad = lbfgs_sycl(q, N, M, x0.data(), 2000, quad, rbuf, 1e-6);
    q.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Quadratic Final F: " << std::scientific << f_quad << " (Target: 0.0)\n";
    std::cout << "Time elapsed: " << std::fixed << ms << " ms\n\n";

    /*
    // --- Rosenbrock ---
    std::fill(x0.begin(), x0.end(), (T)-1.2);
    std::cout << "Starting: Rosenbrock Test (N=" << N << ")...\n";
    RosenbrockTest<T> rosen(q, N, rbuf);
    t0 = std::chrono::high_resolution_clock::now();
    double f_rosen = lbfgs_sycl(q, N, M, x0.data(), 5000, rosen, rbuf, 1e-6);
    q.wait_and_throw();
    t1 = std::chrono::high_resolution_clock::now();

    double ms_rosen = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Rosenbrock Final F: " << std::scientific << f_rosen << " (Target: 0.0)\n";
    std::cout << "Rosenbrock Time elapsed: " << std::fixed << ms_rosen << " ms\n\n";
    */

    // --- Rastrigin ---
    std::fill(x0.begin(), x0.end(), (T)2.5);
    std::cout << "Starting: Rastrigin Test (N=" << N << ")...\n";
    RastriginTest<T> ras(q, N, rbuf);
    t0 = std::chrono::high_resolution_clock::now();
    double f_ras = lbfgs_sycl(q, N, M, x0.data(), 5000, ras, rbuf, 1e-6);
    q.wait_and_throw();
    t1 = std::chrono::high_resolution_clock::now();
    double ms_ras = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Rastrigin Final F: " << std::scientific << f_ras << " (Target: 0.0)\n";
    std::cout << "Rastrigin Time elapsed: " << std::fixed << ms_ras << " ms\n\n";

    // --- Ackley ---
    std::fill(x0.begin(), x0.end(), (T)2.0);
    AckleyTest<T> ack(q, N, rbuf);
    std::cout << "Starting: Ackley Test (N=" << N << ")...\n";
    t0 = std::chrono::high_resolution_clock::now();
    double f_ack = lbfgs_sycl(q, N, M, x0.data(), 5000, ack, rbuf, 1e-6);
    q.wait_and_throw();
    t1 = std::chrono::high_resolution_clock::now();
    double ms_ack = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Ackley Final F: " << std::scientific << f_ack << " (Target: 0.0)\n";
    std::cout << "Ackley Time elapsed: " << std::fixed << ms_ack << " ms\n";

    return 0;
}