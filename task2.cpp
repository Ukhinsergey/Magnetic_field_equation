#include <fftw3-mpi.h>
#include <mpi.h>
#include <functional>
#include <vector>
#include <cmath>
#include <fstream>


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
    const double TAU, ETA;

    Field(const ptrdiff_t N_, const ptrdiff_t alloc_local_, const ptrdiff_t local_n0_,
         const ptrdiff_t local_0_start_, int rank_, int size_, double TAU_, double ETA_):
    N{N_},
    indl{-N / 2 + 1},
    indr{N / 2 + 1},
    alloc_local{alloc_local_},
    local_n0{local_n0_},
    local_0_start{local_0_start_},
    rank{rank_},
    size{size_},
    NORM{std::sqrt(N*N*N)},
    TAU{TAU_},
    ETA{ETA_}
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

    void fill_velocity_field() {
        std::vector<std::function<double(const double, const double, const double)>> func;
        func.push_back([](const double x1, const double x2, const double x3) {
            return 2./std::sqrt(3) * std::sin(x2) * std::cos(x3);
        });
        func.push_back([](const double x1, const double x2, const double x3) {
            return 2./std::sqrt(3) * std::sin(x3) * std::cos(x1);
        });
        func.push_back([](const double x1, const double x2, const double x3) {
            return 2./std::sqrt(3) * std::sin(x1) * std::cos(x2);
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

    void fill_magnetic_field() {
        std::vector<std::function<double(const double, const double, const double)>> func;
        func.push_back([](const double x1, const double x2, const double x3) {
            return std::sin(x1 - 2 * x2 + 3 * x3);
        });
        func.push_back([](const double x1, const double x2, const double x3) {
            return std::cos(-x1 - x3 + 5 * x2);
        });
        func.push_back([](const double x1, const double x2, const double x3) {
            return std::sin(-3 * x1 - x2 + x3);
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

    void divergence(const Field& source_field) {
        double coef_0, coef_1, coef_2;
        ptrdiff_t idx;
        for (ptrdiff_t i = 0; i < local_n0; ++i) {
            for (ptrdiff_t j = 0; j < N; ++j) {
                for (ptrdiff_t k = 0; k < indr; ++k) {
                    idx = (i * N + j) * (N / 2 + 1) + k;
                    coef_0 = inds[local_0_start + i];
                    coef_1 = inds[j];
                    coef_2 = k;
                    vec_c[0][idx][0] =-(coef_0 * source_field.vec_c[0][idx][1] +
                                        coef_1 * source_field.vec_c[1][idx][1] +
                                        coef_2 * source_field.vec_c[2][idx][1]);

                    vec_c[0][idx][1] =  coef_0 * source_field.vec_c[0][idx][0] +
                                        coef_1 * source_field.vec_c[1][idx][0] +
                                        coef_2 * source_field.vec_c[2][idx][0];
                }
            }
        }

        return;
    }


    void correction(Field& tmp_field) {
        tmp_field.divergence(*this);

        ptrdiff_t idx;
        double local_max = 0.;
        for (ptrdiff_t i = 0; i < local_n0; ++i) {
            for (ptrdiff_t j = 0; j < N; ++j) {
                for (ptrdiff_t k = 0; k < indr; ++k) {
                    idx = (i * N + j) * (N / 2 + 1) + k;
                    local_max = std::max(local_max, std::sqrt(
                                                    tmp_field.vec_c[0][idx][0] *
                                                    tmp_field.vec_c[0][idx][0] +
                                                    tmp_field.vec_c[0][idx][1] *
                                                    tmp_field.vec_c[0][idx][1]));
                }
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, &local_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (local_max >= EPSILON) {
            double coef_0, coef_1, coef_2, sum_coef;
            for (ptrdiff_t i = 0; i < local_n0; ++i) {
                for (ptrdiff_t j = 0; j < N; ++j) {
                    for (ptrdiff_t k = 0; k < indr; ++k) {
                        idx = (i * N + j) * (N / 2 + 1) + k;
                        coef_0 = inds[local_0_start + i];
                        coef_1 = inds[j];
                        coef_2 = k;
                        if (std::abs(coef_0) + std::abs(coef_1) + std::abs(coef_2) < EPSILON) {
                            vec_c[0][idx][0] = 0;
                            vec_c[0][idx][1] = 0;
                            vec_c[1][idx][0] = 0;
                            vec_c[1][idx][1] = 0;
                            vec_c[2][idx][0] = 0;
                            vec_c[2][idx][1] = 0;
                        } else {
                            sum_coef = coef_0 * coef_0 + coef_1 * coef_1 + coef_2 * coef_2;
                            vec_c[0][idx][0] += -coef_0 * tmp_field.vec_c[0][idx][1] / sum_coef;
                            vec_c[0][idx][1] +=  coef_0 * tmp_field.vec_c[0][idx][0] / sum_coef;
                            vec_c[1][idx][0] += -coef_1 * tmp_field.vec_c[0][idx][1] / sum_coef;
                            vec_c[1][idx][1] +=  coef_1 * tmp_field.vec_c[0][idx][0] / sum_coef;
                            vec_c[2][idx][0] += -coef_2 * tmp_field.vec_c[0][idx][1] / sum_coef;
                            vec_c[2][idx][1] +=  coef_2 * tmp_field.vec_c[0][idx][0] / sum_coef;
                        }
                    }
                }
            }
        }

        return;
    }

};



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


int main(int argc, char **argv)
{
    const int iters = std::atoi(argv[1]);
    const ptrdiff_t N = std::atoi(argv[2]);
    const double tau = std::strtod(argv[3], nullptr);
    const double eta = std::strtod(argv[4], nullptr);
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    fftw_mpi_init();
    ptrdiff_t alloc_local, local_n0, local_0_start;
    alloc_local = fftw_mpi_local_size_3d(N, N, N/2 + 1, MPI_COMM_WORLD, &local_n0, &local_0_start);
    std::ofstream fout("file_energy.data");
    {
        Field magnetic{N, alloc_local, local_n0, local_0_start, rank, size, tau, eta};
        Field velocity{N, alloc_local, local_n0, local_0_start, rank, size, tau, eta};
        Field tmp{N, alloc_local, local_n0, local_0_start, rank, size, tau, eta};
        velocity.fill_velocity_field();
        magnetic.fill_velocity_field();
        double energy;
        magnetic.forward_transform();
        for(int i = 0; i < iters; ++i) {
            magnetic.correction(tmp);
            if (rank == 0) {
                fout << energy << "\n";
            }
        }
    }
    fout.close();
    MPI_Finalize();
    return 0;
}