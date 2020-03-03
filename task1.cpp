#include <fftw3-mpi.h>
#include <mpi.h>
#include <functional>
#include <vector>
#include <cmath>


const double EPSILON = 1e-8;

struct Field
{   
    const ptrdiff_t N;
    const ptrdiff_t indl, indr, alloc_local, local_n0, local_0_start;
    double *inds;
    int rank, size;
    double *vec_r[3];
    fftw_complex *vec_c[3];
    fftw_plan forward_pl[3];
    fftw_plan backward_pl[3];
    const double NORM;

    Field(const ptrdiff_t N_, const ptrdiff_t alloc_local_, const ptrdiff_t local_n0_,
         const ptrdiff_t local_0_start_, int rank_, int size_):
    N{N_},
    indl{-N / 2 + 1},
    indr{N / 2 + 1},
    alloc_local{alloc_local_},
    local_n0{local_n0_},
    local_0_start{local_0_start_},
    rank{rank_},
    size{size_},
    NORM{std::sqrt(N*N*N)}
    {
        inds = new double[N];
        for(ptrdiff_t i = 0 ; i <= N / 2; ++i) {
            inds[i] = i;
        }
        for(ptrdiff_t i = N / 2 + 1; i < N; ++i) {
            inds[i] = i - N;
        }
        for(int i = 0 ; i < 3; ++i) {
            vec_r[i] = fftw_alloc_real(2 * alloc_local);
            vec_c[i] = fftw_alloc_complex(alloc_local);
            forward_pl[i] = fftw_mpi_plan_dft_r2c_3d(N, N, N, vec_r[i], vec_c[i], MPI_COMM_WORLD, FFTW_MEASURE);
            backward_pl[i] = fftw_mpi_plan_dft_c2r_3d(N, N, N, vec_c[i], vec_r[i], MPI_COMM_WORLD, FFTW_MEASURE);
        }
    }
    ~Field() {
        //std::cout << "destr" << std::endl;
        delete[] inds;
        for(int i = 0; i < 3; ++i) {
            fftw_free(vec_r[i]);
            fftw_free(vec_c[i]);
            fftw_destroy_plan(forward_pl[i]);
            fftw_destroy_plan(backward_pl[i]);
        }
    }

    void fill_func() {
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
            for (ptrdiff_t i = 0; i < local_n0; ++i)
                for (ptrdiff_t j = 0; j < N; ++j)
                    for (ptrdiff_t k = 0; k < N; ++k) {
                        const double cur_x = 2 * M_PI * (local_0_start + i) / N;
                        const double cur_y = 2 * M_PI * j / N;
                        const double cur_z = 2 * M_PI * k / N;
                        vec_r[q][(i * N + j) * (2 * (N / 2 + 1)) + k] = func[q](cur_x, cur_y, cur_z);
                    }
        return;
    }

    void forward_transform() {
        fftw_execute(forward_pl[0]);
        fftw_execute(forward_pl[1]);
        fftw_execute(forward_pl[2]);
        ptrdiff_t idx;
        for(ptrdiff_t i = 0; i < local_n0; ++i){
            for(ptrdiff_t j = 0; j < N; ++j) {
                for(ptrdiff_t k = 0; k < indr; ++k) {
                    idx = (i * N + j) * (N / 2 + 1) +k;
                    vec_c[0][idx][0] /= NORM;
                    vec_c[0][idx][1] /= NORM;
                    vec_c[1][idx][0] /= NORM;
                    vec_c[1][idx][1] /= NORM;
                    vec_c[2][idx][0] /= NORM;
                    vec_c[2][idx][1] /= NORM;
                }
            }
        }
    }

    void backward_transform() {
        fftw_execute(backward_pl[0]);
        fftw_execute(backward_pl[1]);
        fftw_execute(backward_pl[2]);
        ptrdiff_t idx;
        for(ptrdiff_t i = 0; i < local_n0; ++i){
            for(ptrdiff_t j = 0; j < N; ++j) {
                for(ptrdiff_t k = 0; k < N; ++k) {
                    idx = (i * N + j) * ( 2 * (N / 2 + 1)) +k;
                    vec_r[0][idx] /= NORM;
                    vec_r[1][idx] /= NORM;
                    vec_r[2][idx] /= NORM;
                }
            }
        }
    }

    void deriv_func(fftw_complex *vec, int dimension) {
        double coef;
        for(ptrdiff_t i = 0; i < local_n0; ++i) {
            for(ptrdiff_t j = 0 ; j < N; ++j) {
                for(ptrdiff_t k = 0 ; k < indr; ++k) {
                    std::swap(  vec[(i * N + j) * (N / 2 + 1) + k][0],
                                vec[(i * N + j) * (N / 2 + 1) + k][1]);
                    if (dimension == 0) {
                        coef = inds[local_0_start + i];
                    } else if (dimension == 1) {
                        coef = inds[j];
                    } else {
                        coef = k;
                    }
                    vec[(i * N + j) * (N / 2 + 1) + k][0] *= -coef;
                    vec[(i * N + j) * (N / 2 + 1) + k][1] *=  coef;
                }
            }
        }
    
        return;
}
};




void test_deriv(Field &field) {
    ptrdiff_t N = field.N;
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
        field.fill_func();
        field.forward_transform();
        field.deriv_func(field.vec_c[0], n);
        field.deriv_func(field.vec_c[1], n);
        field.deriv_func(field.vec_c[2], n);
        field.backward_transform();

        double max_diff = 0.;
        for (int q = 0; q < 3; ++q) {
            for (ptrdiff_t i = 0; i < field.local_n0; ++i) {
                for (ptrdiff_t j = 0; j < field.N; ++j) {
                    for (ptrdiff_t k = 0; k < field.N; ++k) {
                        const double cur_x = 2 * M_PI * (field.local_0_start + i) / field.N;
                        const double cur_y = 2 * M_PI * j / field.N;
                        const double cur_z = 2 * M_PI * k / field.N;
                        max_diff = std::max(max_diff, std::abs(field.vec_r[q][(i * N + j) * (2 * (N / 2 + 1)) + k] - real_deriv_func[q][n](cur_x, cur_y, cur_z)));
                    }
                }
            }
        }

        if (field.rank == 0) {
            MPI_Reduce(MPI_IN_PLACE, &max_diff, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(&max_diff, nullptr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        }

        if (field.rank == 0) {
            if (n == 0) {
                std::cout << "max deviation from correct ans(derivetive) for var:" << '\n'; 
            }
            std::cout << n + 1 << " -- " << max_diff << '\n';
        }

    }

    return;
}

void test_divergence(Field& field) {
    const ptrdiff_t N = field.N;

    std::vector<std::vector<std::function<double(const double, const double, const double)>>> real_deriv_func(3); // i - func num, j - var num
    real_deriv_func[0].push_back([](const double x1, const double x2, const double x3) {
        return std::exp(std::sin(x1 - 2 * x2 + 3 * x3)) * std::cos(1 * x1 - 2 * x2 + 3 * x3);
    });
    real_deriv_func[0].push_back([](const double x1, const double x2, const double x3) {
        return std::exp(std::sin(x1 - 2 * x2 + 3 * x3)) * -2 * std::cos(1 * x1 - 2 * x2 + 3 * x3);
    });
    real_deriv_func[0].push_back([](const double x1, const double x2, const double x3) {
        return std::exp(std::sin(x1 - 2 * x2 + 3 * x3)) * 3 * std::cos(1 * x1 - 2 * x2 + 3 * x3);
    });
    real_deriv_func[1].push_back([](const double x1, const double x2, const double x3) {
        return std::exp(std::cos(-x1 - x3)) * std::sin(-x1 - x3);
    });
    real_deriv_func[1].push_back([](const double x1, const double x2, const double x3) {
        return 0;
    });
    real_deriv_func[1].push_back([](const double x1, const double x2, const double x3) {
        return std::exp(std::cos(-x1 - x3)) * std::sin(-x1 - x3);
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

    field.fill_func();
    field.forward_transform();

    field.deriv_func(field.vec_c[0], 0);
    field.deriv_func(field.vec_c[1], 1);
    field.deriv_func(field.vec_c[2], 2);

    field.backward_transform();


    double max_diff = 0.;
    for (ptrdiff_t i = 0; i < field.local_n0; ++i) {
        for (ptrdiff_t j = 0; j < N; ++j) {
            for (ptrdiff_t k = 0; k < N; ++k) {
                const double cur_x = 2 * M_PI * (field.local_0_start + i) / N;
                const double cur_y = 2 * M_PI * j / N;
                const double cur_z = 2 * M_PI * k / N;
                max_diff = std::max(max_diff, std::abs(
                    field.vec_r[0][(i * N + j) * (2 * (N / 2 + 1)) + k] +
                    field.vec_r[1][(i * N + j) * (2 * (N / 2 + 1)) + k] +
                    field.vec_r[2][(i * N + j) * (2 * (N / 2 + 1)) + k] -
                    (real_deriv_func[0][0](cur_x, cur_y, cur_z) +
                     real_deriv_func[1][1](cur_x, cur_y, cur_z) +
                     real_deriv_func[2][2](cur_x, cur_y, cur_z))));
            }
        }
    }

    if (field.rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &max_diff, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(&max_diff, nullptr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    if (field.rank == 0) {
        std::cout << "max deviation from correct ans(div):" << '\n';
        std::cout << max_diff << '\n';
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
    {
        Field field{N, alloc_local, local_n0, local_0_start, rank, size};
        test_deriv(field);
        test_divergence(field);
     //   MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}