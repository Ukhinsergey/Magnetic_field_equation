#include <fftw3-mpi.h>
#include <mpi.h>
#include <functional>
#include <vector>
#include <cmath>


const double EPSILON = 1e-8;

struct Info
{   
    const ptrdiff_t N;
    const ptrdiff_t indl, indr, alloc_local, local_n0, local_0_start;
    double *inds;
    int rank, size;

    Info(const ptrdiff_t N_, const ptrdiff_t alloc_local_, const ptrdiff_t local_n0_,
         const ptrdiff_t local_0_start_, int rank_, int size_):
    N{N_},
    indl{-N / 2 + 1},
    indr{N / 2 + 1},
    alloc_local{alloc_local_},
    local_n0{local_n0_},
    local_0_start{local_0_start_},
    rank{rank_},
    size{size_} {
        inds = new double[N];
        for(ptrdiff_t i = 0 ; i <= N / 2; ++i) {
            inds[i] = i;
        }
        for(ptrdiff_t i = N / 2 + 1; i < N; ++i) {
            inds[i] = i - N;
        }
    }
    ~Info() {
        delete[] inds;
    }
};


void fill_func(double * vec[3], const Info &info) {
    std::vector<std::function<double(const double, const double, const double)>> func;
    func.push_back([](const double x1, const double x2, const double x3) {
        return std::exp(std::sin(x1 - 2 * x2 + 3 * x3));
    });
    func.push_back([](const double x1, const double x2, const double x3) {
        return std::exp(std::cos(-x1 - x3));
    });
    func.push_back([](const double x1, const double x2, const double x3) {
        return std::exp(std::sin(-3 * x1 - x2 + x3));
    });

    for (ptrdiff_t q = 0; q < 3; ++q)
        for (ptrdiff_t i = 0; i < info.local_n0; ++i)
            for (ptrdiff_t j = 0; j < info.N; ++j)
                for (ptrdiff_t k = 0; k < info.N; ++k) {
                    const double cur_x = 2 * M_PI * (info.local_0_start + i) / info.N;
                    const double cur_y = 2 * M_PI * j / info.N;
                    const double cur_z = 2 * M_PI * k / info.N;
                    vec[q][(i * info.N + j) * (2 * (info.N / 2 + 1)) + k] = func[q](cur_x, cur_y, cur_z);
                }


    return;
}
void deriv_func(fftw_complex *ptr, const Info &info, int dimension) {
    double coef;
    for(ptrdiff_t i = 0; i < info.local_n0; ++i) {
        for(ptrdiff_t j = 0 ; j < info.N; ++j) {
            for(ptrdiff_t k = 0 ; k < info.indr; ++k) {
                std::swap(  ptr[(i * info.N + j) * (info.N / 2 + 1) + k][0],
                            ptr[(i * info.N + j) * (info.N / 2 + 1) + k][1]);
                if (dimension == 0) {
                    coef = info.inds[info.local_0_start + i];
                } else if (dimension == 1) {
                    coef = info.inds[j];
                } else {
                    coef = k;
                }
                ptr[(i * info.N + j) * (info.N / 2 + 1) + k][0] *= -coef;
                ptr[(i * info.N + j) * (info.N / 2 + 1) + k][1] *=  coef;
            }
        }
    }
    return;
}

void test_deriv(const Info &info) {
    const ptrdiff_t N = info.N;
    fftw_plan forward_pl[3], backward_pl[3];
    double *vec_r[3];
    fftw_complex *vec_c[3];
    for(int i = 0 ; i < 3; ++i) {
        vec_r[i] = fftw_alloc_real(2 * info.alloc_local);
        vec_c[i] = fftw_alloc_complex(info.alloc_local);
        forward_pl[i] = fftw_mpi_plan_dft_r2c_3d(N, N, N, vec_r[i], vec_c[i], MPI_COMM_WORLD, FFTW_MEASURE);
        backward_pl[i] = fftw_mpi_plan_dft_c2r_3d(N, N, N, vec_c[i], vec_r[i], MPI_COMM_WORLD, FFTW_MEASURE);
    }
    std::vector<std::vector<std::function<double(const double, const double, const double)>>> real_deriv_func(3);
    real_deriv_func[0].push_back([](const double x1, const double x2, const double x3) {
        return std::exp(std::sin(x1 - 2 * x2 + 3 * x3)) * std::cos(1 * x1 - 2 * x2 + 3 * x3);
    });
    real_deriv_func[0].push_back([](const double x1, const double x2, const double x3) {
        return std::exp(std::sin(x1 - 2 * x2 + 3 * x3)) * -2 * std::cos(1 * x1 - 2 * x2 + 3 * x3);
    });
    real_deriv_func[0].push_back([](const double x1, const double x2, const double x3) {
        return std::exp(std::sin(x1 - 2 * x2 + 3 * x3)) *  3 * std::cos(1 * x1 - 2 * x2 + 3 * x3);
    });
    real_deriv_func[1].push_back([](const double x1, const double x2, const double x3) {
        return std::exp(std::cos(-x1 - x3)) * std::sin(-x1 -x3);
    });
    real_deriv_func[1].push_back([](const double x1, const double x2, const double x3) {
        return 0;
    });
    real_deriv_func[1].push_back([](const double x1, const double x2, const double x3) {
        return std::exp(std::cos(-x1 - x3)) * std::sin(-x1 -x3);
    });
    real_deriv_func[2].push_back([](const double x1, const double x2, const double x3) {
        return std::exp(std::sin(-3 * x1 - x2 + x3)) * -3 * std::cos(-3 * x1 - x2 + x3);
    });
    real_deriv_func[2].push_back([](const double x1, const double x2, const double x3) {
        return std::exp(std::sin(-3 * x1 - x2 + x3)) * -std::cos(-3 * x1 - x2 + x3);
    });
    real_deriv_func[2].push_back([](const double x1, const double x2, const double x3) {
        return std::exp(std::sin(-3 * x1 - x2 + x3)) * std::cos(-3 * x1 - x2 + x3);
    });
    for(int n = 0 ; n < 3; ++n) {
        fill_func(vec_r, info);

        fftw_execute(forward_pl[0]);
        fftw_execute(forward_pl[1]);
        fftw_execute(forward_pl[2]);

        deriv_func(vec_c[0], info, n);
        deriv_func(vec_c[1], info, n);
        deriv_func(vec_c[2], info, n);

        fftw_execute(backward_pl[0]);
        fftw_execute(backward_pl[1]);
        fftw_execute(backward_pl[2]);
        for (int q = 0; q < 3; ++q) {
            for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
                for (ptrdiff_t j = 0; j < info.N; ++j) {
                    for (ptrdiff_t k = 0; k < info.N; ++k) {
                        vec_r[q][(i * N + j) * (2 * (N / 2 + 1)) + k] /= N * N * N;
                    }
                }
            }
        }
        double max_diff = 0.;
        for (int q = 0; q < 3; ++q) {
            for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
                for (ptrdiff_t j = 0; j < info.N; ++j) {
                    for (ptrdiff_t k = 0; k < info.N; ++k) {
                        const double cur_x = 2 * M_PI * (info.local_0_start + i) / info.N;
                        const double cur_y = 2 * M_PI * j / info.N;
                        const double cur_z = 2 * M_PI * k / info.N;
                        max_diff = std::max(max_diff, std::abs(vec_r[q][(i * N + j) * (2 * (N / 2 + 1)) + k] - real_deriv_func[q][n](cur_x, cur_y, cur_z)));
                    }
                }
            }
        }

        if (info.rank == 0) {
            MPI_Reduce(MPI_IN_PLACE, &max_diff, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(&max_diff, nullptr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        }

        if (info.rank == 0) {
            if (n == 0) {
                std::cout << "max deviation from correct ans(derivetive) for var:" << '\n'; 
            }
            std::cout << n + 1 << " -- " << max_diff << '\n';
        }

    }

    for (int i = 0; i < 3; ++i) {
        fftw_free(vec_r[i]);
        fftw_free(vec_c[i]);
        fftw_destroy_plan(forward_pl[i]);
        fftw_destroy_plan(backward_pl[i]);
    }

    return;
}

int main(int argc, char **argv)
{
    const ptrdiff_t N = std::atoi(argv[1]);
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    fftw_mpi_init();


    ptrdiff_t alloc_local, local_n0, local_0_start;
    alloc_local = fftw_mpi_local_size_3d(N, N, N/2 + 1, MPI_COMM_WORLD, &local_n0, &local_0_start);
    const Info info(N, alloc_local, local_n0, local_0_start, rank, size);
    test_deriv(info);
    

    MPI_Finalize();
    return 0;
}