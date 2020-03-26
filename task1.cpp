#include <fftw3-mpi.h>
#include <mpi.h>
#include <functional>
#include <vector>
#include <cmath>
#include <iostream>


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
            return std::exp(std::cos(-x1 + x2 + x3));
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
        return std::exp(std::cos(-x1 + x2 + x3)) * std::sin(-x1 + x2 + x3);
    });
    real_deriv_func[1].push_back([](const double x1, const double x2, const double x3) {
        return -std::exp(std::cos(-x1 + x2 + x3)) * std::sin(-x1 + x2 + x3);
    });
    real_deriv_func[1].push_back([](const double x1, const double x2, const double x3) {
        return -std::exp(std::cos(-x1 + x2 + x3)) * std::sin(-x1 + x2 + x3);
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
        return std::exp(std::cos(-x1 + x2 + x3)) * std::sin(-x1 + x2 + x3);
    });
    real_deriv_func[1].push_back([](const double x1, const double x2, const double x3) {
        return -std::exp(std::cos(-x1 + x2 + x3)) * std::sin(-x1 + x2 + x3);
    });
    real_deriv_func[1].push_back([](const double x1, const double x2, const double x3) {
        return -std::exp(std::cos(-x1 + x2 + x3)) * std::sin(-x1 + x2 + x3);
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
    double starttime, endtime;
    starttime = MPI_Wtime();
    field.forward_transform();
    endtime = MPI_Wtime();
    float restime = endtime - starttime;
    //std::cout << restime << std::endl;

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
        MPI_Reduce(MPI_IN_PLACE, &restime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(MPI_IN_PLACE, &max_diff, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(&restime, nullptr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&max_diff, nullptr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    if (field.rank == 0) {
        std::cout << "max time for TTF " << restime << std::endl;
        std::cout << "max deviation from correct ans(div):" << '\n';
        std::cout << max_diff << '\n';
    }

    return;
}


void rotor(fftw_complex *rot,
           const fftw_complex *cross_p_l, const fftw_complex *cross_p_r,
           Field& field, const int num_of_dimension) {
    double coef_l, coef_r;
    for (ptrdiff_t i = 0; i < field.local_n0; ++i) {
        for (ptrdiff_t j = 0; j < field.N; ++j) {
            for (ptrdiff_t k = 0; k < field.indr; ++k) {
                const ptrdiff_t idx = (i * field.N + j) * (field.N / 2 + 1) + k;
                if (num_of_dimension == 0) {
                    coef_l = field.inds[j];
                    coef_r = k;
                } else if (num_of_dimension == 1) {
                    coef_l = k;
                    coef_r = field.inds[field.local_0_start + i];
                } else {
                    coef_l = field.inds[field.local_0_start + i];
                    coef_r = field.inds[j];
                }
                rot[idx][0] = -cross_p_l[idx][1] * coef_l + cross_p_r[idx][1] * coef_r;
                rot[idx][1] =  cross_p_l[idx][0] * coef_l - cross_p_r[idx][0] * coef_r;
            }
        }
    }
    return;
}


void test_rotor(Field& field) {
    const ptrdiff_t N = field.N;
    
    field.fill_func();

    std::vector<std::function<double(const double, const double, const double)>> rotor_functions;
    rotor_functions.push_back([](const double x1, const double x2, const double x3) {
        return -std::exp(std::sin(-3 * x1 - x2 + x3)) * std::cos(-3 * x1 - x2 + x3) + std::exp(std::cos(-x1 + x2 + x3)) * std::sin(-x1 + x2 + x3);
    });
    rotor_functions.push_back([](const double x1, const double x2, const double x3) {
        return std::exp(std::sin(x1 - 2 * x2 + 3 * x3)) * 3 * std::cos(x1 - 2 * x2 + 3 * x3) + std::exp(std::sin(-3 * x1 - x2 + x3)) * 3 * std::cos(-3 * x1 - x2 + x3);
    });
    rotor_functions.push_back([](const double x1, const double x2, const double x3) {
        return std::exp(std::cos(-x1 + x2 + x3)) * std::sin(-x1 + x2 + x3) + std::exp(std::sin(x1 - 2 * x2 + 3 * x3)) * 2 * std::cos(x1 - 2 * x2 + 3 * x3);
    });

    fftw_complex *rotor_c[3];
    double* rotor_r[3];
    fftw_plan rot_c_to_r[3];
    for (int q = 0; q < 3; ++q) {
        rotor_r[q] = fftw_alloc_real(2 * field.alloc_local);
        rotor_c[q] = fftw_alloc_complex(field.alloc_local);
        rot_c_to_r[q] = fftw_mpi_plan_dft_c2r_3d(N, N, N, rotor_c[q], rotor_r[q], MPI_COMM_WORLD, FFTW_MEASURE);
    }

    field.forward_transform();

    rotor(rotor_c[0], field.vec_c[2], field.vec_c[1], field, 0);
    rotor(rotor_c[1], field.vec_c[0], field.vec_c[2], field, 1);
    rotor(rotor_c[2], field.vec_c[1], field.vec_c[0], field, 2);

    fftw_execute(rot_c_to_r[0]);
    fftw_execute(rot_c_to_r[1]);
    fftw_execute(rot_c_to_r[2]);

    for (int q = 0; q < 3; ++q) {
        for (ptrdiff_t i = 0; i < field.local_n0; ++i) {
            for (ptrdiff_t j = 0; j < field.N; ++j) {
                for (ptrdiff_t k = 0; k < field.N; ++k) {
                    rotor_r[q][(i * N + j) * (2 * (N / 2 + 1)) + k] /= N * std::sqrt(N);
                }
            }
        }
    }

    double max_diff = 0.;
    for (int q = 0; q < 3; ++q) {
        for (ptrdiff_t i = 0; i < field.local_n0; ++i) {
            for (ptrdiff_t j = 0; j < field.N; ++j) {
                for (ptrdiff_t k = 0; k < field.N; ++k) {
                    const double cur_x = 2 * M_PI * (field.local_0_start + i) / N;
                    const double cur_y = 2 * M_PI * j / N;
                    const double cur_z = 2 * M_PI * k / N;
                    max_diff = std::max(max_diff, std::abs(rotor_r[q][(i * N + j) * (2 * (N / 2 + 1)) + k] - rotor_functions[q](cur_x, cur_y, cur_z)));
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
        std::cout << "max deviation from correct ans(rot):" << '\n';
        std::cout << max_diff << '\n';
    }


    for (int q = 0; q < 3; ++q) {
        fftw_free(rotor_r[q]);
        fftw_free(rotor_c[q]);
        fftw_destroy_plan(rot_c_to_r[q]);
    }

    return;
}

double field_energy_phi(const double *ptr_1,
                        const double *ptr_2,
                        const double *ptr_3,
                        Field& field) {
    double energy = 0.;
    for (ptrdiff_t i = 0; i < field.local_n0; ++i) {
        for (ptrdiff_t j = 0; j < field.N; ++j) {
            for (ptrdiff_t k = 0; k < field.N; ++k) {
                const ptrdiff_t idx = (i * field.N + j) * (2 * (field.N / 2 + 1)) + k;
                energy += ptr_1[idx] * ptr_1[idx] +
                          ptr_2[idx] * ptr_2[idx] +
                          ptr_3[idx] * ptr_3[idx];
            }
        }
    }
    energy /= 2;
    energy /=  field.N * field.N * field.N;
    return energy;
}

double field_energy_fourie( const fftw_complex *ptr_1,
                            const fftw_complex *ptr_2,
                            const fftw_complex *ptr_3,
                            Field& field) {
    double energy = 0.;
    for (ptrdiff_t i = 0; i < field.local_n0; ++i) {
        for (ptrdiff_t j = 0; j < field.N; ++j) {
            ptrdiff_t idx = (i * field.N + j) * (field.N / 2 + 1);
            energy += 0.5 * (ptr_1[idx][0] * ptr_1[idx][0] + ptr_1[idx][1] * ptr_1[idx][1] +
                             ptr_2[idx][0] * ptr_2[idx][0] + ptr_2[idx][1] * ptr_2[idx][1] +
                             ptr_3[idx][0] * ptr_3[idx][0] + ptr_3[idx][1] * ptr_3[idx][1]);
            for (ptrdiff_t k = 1; k < field.indr; ++k) {
                ++idx;
                energy += (ptr_1[idx][0] * ptr_1[idx][0] + ptr_1[idx][1] * ptr_1[idx][1] +
                           ptr_2[idx][0] * ptr_2[idx][0] + ptr_2[idx][1] * ptr_2[idx][1] +
                           ptr_3[idx][0] * ptr_3[idx][0] + ptr_3[idx][1] * ptr_3[idx][1]);
            }
        }
    }
    energy /=  field.N * field.N * field.N;
    return energy;
}

void test_energy(Field& field) {
    field.fill_func();

    double energy = field_energy_phi(field.vec_r[0], field.vec_r[1], field.vec_r[2], field);
    if (field.rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(&energy, nullptr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (field.rank == 0) {
        std::cout << "energy real space" << '\n';
        std::cout << energy << '\n';
    }

    field.forward_transform();

    energy = field_energy_fourie(field.vec_c[0], field.vec_c[1], field.vec_c[2], field);
    if (field.rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(&energy, nullptr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (field.rank == 0) {
        std::cout << "energy fourie space" << '\n';
        std::cout << energy << '\n';
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
        test_rotor(field);
        test_energy(field);
    }
    MPI_Finalize();
    return 0;
}