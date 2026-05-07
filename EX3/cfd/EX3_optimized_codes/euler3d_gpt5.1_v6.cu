// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

#include <helper_cuda.h>
#include <helper_timer.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>


/*
 * Options
 *
 */
#define GAMMA 1.4f
#define iterations 2000

#define NDIM 3
#define NNB 4

#define RK 3 // 3rd order RK
#define ff_mach 1.2f
#define deg_angle_of_attack 0.0f

/*
 * not options
 */

#define BLOCK_SIZE 192
#define VAR_DENSITY 0
#define VAR_MOMENTUM 1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM + NDIM)
#define NVAR (VAR_DENSITY_ENERGY + 1)


/*
 * Generic functions
 */
template <typename T> T *alloc(int N) {
    T *t;
    checkCudaErrors(cudaMalloc((void **)&t, sizeof(T) * N));
    return t;
}

template <typename T> void dealloc(T *array) {
    checkCudaErrors(cudaFree((void *)array));
}

template <typename T> void copy(T *dst, T *src, int N) {
    checkCudaErrors(cudaMemcpy((void *)dst, (void *)src, N * sizeof(T),
                               cudaMemcpyDeviceToDevice));
}

template <typename T> void upload(T *dst, T *src, int N) {
    checkCudaErrors(cudaMemcpy((void *)dst, (void *)src, N * sizeof(T),
                               cudaMemcpyHostToDevice));
}

template <typename T> void download(T *dst, T *src, int N) {
    checkCudaErrors(cudaMemcpy((void *)dst, (void *)src, N * sizeof(T),
                               cudaMemcpyDeviceToHost));
}

void dump(float *variables, int nel, int nelr, const char *output_file = NULL) {
    float *h_variables = new float[nelr * NVAR];
    download(h_variables, variables, nelr * NVAR);

    if (output_file != NULL) {
        // Write all data to a single output file
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open output file " << output_file << std::endl;
            delete[] h_variables;
            return;
        }
        
        // Write density
        file << "density" << std::endl;
        file << nel << std::endl;
        for (int i = 0; i < nel; i++)
            file << h_variables[i + VAR_DENSITY * nelr] << std::endl;
        
        // Write momentum
        file << "momentum" << std::endl;
        file << nel << std::endl;
        for (int i = 0; i < nel; i++) {
            for (int j = 0; j != NDIM; j++)
                file << h_variables[i + (VAR_MOMENTUM + j) * nelr] << " ";
            file << std::endl;
        }
        
        // Write density_energy
        file << "density_energy" << std::endl;
        file << nel << std::endl;
        for (int i = 0; i < nel; i++)
            file << h_variables[i + VAR_DENSITY_ENERGY * nelr] << std::endl;
        
        file.close();
        std::cout << "Results written to " << output_file << std::endl;
    } else {
        // Original behavior: write to separate files
        {
            std::ofstream file("density");
            file << nel << std::endl;
            for (int i = 0; i < nel; i++)
                file << h_variables[i + VAR_DENSITY * nelr] << std::endl;
        }

        {
        std::ofstream file("momentum");
        file << nel << std::endl;
        for (int i = 0; i < nel; i++) {
            for (int j = 0; j != NDIM; j++)
                file << h_variables[i + (VAR_MOMENTUM + j) * nelr] << " ";
            file << std::endl;
        }
    }

    {
        std::ofstream file("density_energy");
        file << nel << std::endl;
        for (int i = 0; i < nel; i++)
            file << h_variables[i + VAR_DENSITY_ENERGY * nelr] << std::endl;
        }
    }
    delete[] h_variables;
}

/*
 * Element-based Cell-centered FVM solver functions
 */
__constant__ float ff_variable[NVAR];
__constant__ float3 ff_flux_contribution_momentum_x[1];
__constant__ float3 ff_flux_contribution_momentum_y[1];
__constant__ float3 ff_flux_contribution_momentum_z[1];
__constant__ float3 ff_flux_contribution_density_energy[1];

__global__ void cuda_initialize_variables(int nelr, float *variables) {
    const int i = (blockDim.x * blockIdx.x + threadIdx.x);
    for (int j = 0; j < NVAR; j++)
        variables[i + j * nelr] = ff_variable[j];
}
void initialize_variables(int nelr, float *variables) {
    dim3 Dg(nelr / BLOCK_SIZE), Db(BLOCK_SIZE);
    cuda_initialize_variables<<<Dg, Db>>>(nelr, variables);
    getLastCudaError("initialize_variables failed");
}

__device__ __host__ inline void compute_flux_contribution(
    float &density, float3 &momentum, float &density_energy, float &pressure,
    float3 &velocity, float3 &fc_momentum_x, float3 &fc_momentum_y,
    float3 &fc_momentum_z, float3 &fc_density_energy) {
    fc_momentum_x.x = velocity.x * momentum.x + pressure;
    fc_momentum_x.y = velocity.x * momentum.y;
    fc_momentum_x.z = velocity.x * momentum.z;


    fc_momentum_y.x = fc_momentum_x.y;
    fc_momentum_y.y = velocity.y * momentum.y + pressure;
    fc_momentum_y.z = velocity.y * momentum.z;

    fc_momentum_z.x = fc_momentum_x.z;
    fc_momentum_z.y = fc_momentum_y.z;
    fc_momentum_z.z = velocity.z * momentum.z + pressure;

    float de_p = density_energy + pressure;
    fc_density_energy.x = velocity.x * de_p;
    fc_density_energy.y = velocity.y * de_p;
    fc_density_energy.z = velocity.z * de_p;
}

__device__ inline void compute_velocity(float &density, float3 &momentum,
                                        float3 &velocity) {
    velocity.x = momentum.x / density;
    velocity.y = momentum.y / density;
    velocity.z = momentum.z / density;
}

__device__ inline float compute_speed_sqd(float3 &velocity) {
    return velocity.x * velocity.x + velocity.y * velocity.y +
           velocity.z * velocity.z;
}

__device__ inline float compute_pressure(float &density, float &density_energy,
                                         float &speed_sqd) {
    return (float(GAMMA) - float(1.0f)) *
           (density_energy - float(0.5f) * density * speed_sqd);
}

__device__ inline float compute_speed_of_sound(float &density,
                                               float &pressure) {
    return sqrtf(float(GAMMA) * pressure / density);
}

__global__ void cuda_compute_step_factor(int nelr, float *variables,
                                         float *areas, float *step_factors) {
    const int i = (blockDim.x * blockIdx.x + threadIdx.x);

    float density = variables[i + VAR_DENSITY * nelr];
    float3 momentum;
    momentum.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
    momentum.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
    momentum.z = variables[i + (VAR_MOMENTUM + 2) * nelr];

    float density_energy = variables[i + VAR_DENSITY_ENERGY * nelr];

    float3 velocity;
    compute_velocity(density, momentum, velocity);
    float speed_sqd = compute_speed_sqd(velocity);
    float pressure = compute_pressure(density, density_energy, speed_sqd);
    float speed_of_sound = compute_speed_of_sound(density, pressure);

    // dt = float(0.5f) * sqrtf(areas[i]) /  (||v|| + c).... but when we do time
    // stepping, this later would need to be divided by the area, so we just do
    // it all at once
    step_factors[i] =
        float(0.5f) / (sqrtf(areas[i]) * (sqrtf(speed_sqd) + speed_of_sound));
}
void compute_step_factor(int nelr, float *variables, float *areas,
                         float *step_factors) {
    dim3 Dg(nelr / BLOCK_SIZE), Db(BLOCK_SIZE);
    cuda_compute_step_factor<<<Dg, Db>>>(nelr, variables, areas, step_factors);
    getLastCudaError("compute_step_factor failed");
}

/*
 *
 *
*/
__global__ void cuda_compute_flux(int nelr, int * __restrict__ elements_surrounding_elements,
                                  const float * __restrict__ normals, const float * __restrict__ variables,
                                  float * __restrict__ fluxes) {
    const float smoothing_coefficient = 0.2f;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= nelr) return;

    const int var_density_offset        = VAR_DENSITY * nelr;
    const int var_momentum0_offset      = (VAR_MOMENTUM + 0) * nelr;
    const int var_momentum1_offset      = (VAR_MOMENTUM + 1) * nelr;
    const int var_momentum2_offset      = (VAR_MOMENTUM + 2) * nelr;
    const int var_density_energy_offset = VAR_DENSITY_ENERGY * nelr;

    float density_i        = __ldg(&variables[i + var_density_offset]);
    float3 momentum_i;
    momentum_i.x           = __ldg(&variables[i + var_momentum0_offset]);
    momentum_i.y           = __ldg(&variables[i + var_momentum1_offset]);
    momentum_i.z           = __ldg(&variables[i + var_momentum2_offset]);
    float density_energy_i = __ldg(&variables[i + var_density_energy_offset]);

    float3 velocity_i;
    compute_velocity(density_i, momentum_i, velocity_i);
    float speed_sqd_i      = compute_speed_sqd(velocity_i);
    float speed_i          = sqrtf(speed_sqd_i);
    float pressure_i       = compute_pressure(density_i, density_energy_i, speed_sqd_i);
    float speed_of_sound_i = compute_speed_of_sound(density_i, pressure_i);

    float3 flux_contribution_i_momentum_x;
    float3 flux_contribution_i_momentum_y;
    float3 flux_contribution_i_momentum_z;
    float3 flux_contribution_i_density_energy;
    compute_flux_contribution(
        density_i, momentum_i, density_energy_i, pressure_i, velocity_i,
        flux_contribution_i_momentum_x, flux_contribution_i_momentum_y,
        flux_contribution_i_momentum_z, flux_contribution_i_density_energy);

    float  flux_i_density         = 0.0f;
    float3 flux_i_momentum;
    flux_i_momentum.x             = 0.0f;
    flux_i_momentum.y             = 0.0f;
    flux_i_momentum.z             = 0.0f;
    float  flux_i_density_energy  = 0.0f;

    float3 velocity_nb;
    float  density_nb, density_energy_nb;
    float3 momentum_nb;
    float3 flux_contribution_nb_momentum_x;
    float3 flux_contribution_nb_momentum_y;
    float3 flux_contribution_nb_momentum_z;
    float3 flux_contribution_nb_density_energy;
    float  speed_sqd_nb, speed_of_sound_nb, pressure_nb;

#pragma unroll
    for (int j = 0; j < NNB; j++) {
        const int  base_idx = i + j * nelr;
        const int  nb       = elements_surrounding_elements[base_idx];

        float3 normal;
        normal.x = normals[i + (j + 0 * NNB) * nelr];
        normal.y = normals[i + (j + 1 * NNB) * nelr];
        normal.z = normals[i + (j + 2 * NNB) * nelr];

        const float normal_x = normal.x;
        const float normal_y = normal.y;
        const float normal_z = normal.z;

        const float normal_len = rsqrtf(normal_x * normal_x +
                                        normal_y * normal_y +
                                        normal_z * normal_z);
        float factor;

        if (nb >= 0) {
            const int nb_density_offset        = nb + var_density_offset;
            const int nb_momentum0_offset      = nb + var_momentum0_offset;
            const int nb_momentum1_offset      = nb + var_momentum1_offset;
            const int nb_momentum2_offset      = nb + var_momentum2_offset;
            const int nb_density_energy_offset = nb + var_density_energy_offset;

            density_nb        = variables[nb_density_offset];
            momentum_nb.x     = variables[nb_momentum0_offset];
            momentum_nb.y     = variables[nb_momentum1_offset];
            momentum_nb.z     = variables[nb_momentum2_offset];
            density_energy_nb = variables[nb_density_energy_offset];

            compute_velocity(density_nb, momentum_nb, velocity_nb);
            speed_sqd_nb      = compute_speed_sqd(velocity_nb);
            pressure_nb       = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
            speed_of_sound_nb = compute_speed_of_sound(density_nb, pressure_nb);

            compute_flux_contribution(
                density_nb, momentum_nb, density_energy_nb, pressure_nb,
                velocity_nb, flux_contribution_nb_momentum_x,
                flux_contribution_nb_momentum_y,
                flux_contribution_nb_momentum_z,
                flux_contribution_nb_density_energy);

            // artificial viscosity
            const float speed_nb = sqrtf(speed_sqd_nb);
            factor = -normal_len * smoothing_coefficient * 0.5f *
                     (speed_i + speed_nb + speed_of_sound_i + speed_of_sound_nb);

            const float density_diff        = density_i        - density_nb;
            const float density_energy_diff = density_energy_i - density_energy_nb;
            const float momentum_x_diff     = momentum_i.x     - momentum_nb.x;
            const float momentum_y_diff     = momentum_i.y     - momentum_nb.y;
            const float momentum_z_diff     = momentum_i.z     - momentum_nb.z;

            flux_i_density        += factor * density_diff;
            flux_i_density_energy += factor * density_energy_diff;
            flux_i_momentum.x     += factor * momentum_x_diff;
            flux_i_momentum.y     += factor * momentum_y_diff;
            flux_i_momentum.z     += factor * momentum_z_diff;

            // accumulate cell-centered fluxes
            const float factor_x = 0.5f * normal_x;
            const float factor_y = 0.5f * normal_y;
            const float factor_z = 0.5f * normal_z;

            const float mom_x_sum_x = momentum_nb.x + momentum_i.x;
            const float mom_x_sum_y = momentum_nb.y + momentum_i.y;
            const float mom_x_sum_z = momentum_nb.z + momentum_i.z;

            const float de_x_sum = flux_contribution_nb_density_energy.x +
                                   flux_contribution_i_density_energy.x;
            const float de_y_sum = flux_contribution_nb_density_energy.y +
                                   flux_contribution_i_density_energy.y;
            const float de_z_sum = flux_contribution_nb_density_energy.z +
                                   flux_contribution_i_density_energy.z;

            const float mom_flux_xx_sum = flux_contribution_nb_momentum_x.x +
                                          flux_contribution_i_momentum_x.x;
            const float mom_flux_xy_sum = flux_contribution_nb_momentum_x.y +
                                          flux_contribution_i_momentum_x.y;
            const float mom_flux_xz_sum = flux_contribution_nb_momentum_x.z +
                                          flux_contribution_i_momentum_x.z;

            const float mom_flux_yx_sum = flux_contribution_nb_momentum_y.x +
                                          flux_contribution_i_momentum_y.x;
            const float mom_flux_yy_sum = flux_contribution_nb_momentum_y.y +
                                          flux_contribution_i_momentum_y.y;
            const float mom_flux_yz_sum = flux_contribution_nb_momentum_y.z +
                                          flux_contribution_i_momentum_y.z;

            const float mom_flux_zx_sum = flux_contribution_nb_momentum_z.x +
                                          flux_contribution_i_momentum_z.x;
            const float mom_flux_zy_sum = flux_contribution_nb_momentum_z.y +
                                          flux_contribution_i_momentum_z.y;
            const float mom_flux_zz_sum = flux_contribution_nb_momentum_z.z +
                                          flux_contribution_i_momentum_z.z;

            // x-normal contribution
            flux_i_density        += factor_x * mom_x_sum_x;
            flux_i_density_energy += factor_x * de_x_sum;
            flux_i_momentum.x     += factor_x * mom_flux_xx_sum;
            flux_i_momentum.y     += factor_x * mom_flux_yx_sum;
            flux_i_momentum.z     += factor_x * mom_flux_zx_sum;

            // y-normal contribution
            flux_i_density        += factor_y * mom_x_sum_y;
            flux_i_density_energy += factor_y * de_y_sum;
            flux_i_momentum.x     += factor_y * mom_flux_xy_sum;
            flux_i_momentum.y     += factor_y * mom_flux_yy_sum;
            flux_i_momentum.z     += factor_y * mom_flux_zy_sum;

            // z-normal contribution
            flux_i_density        += factor_z * mom_x_sum_z;
            flux_i_density_energy += factor_z * de_z_sum;
            flux_i_momentum.x     += factor_z * mom_flux_xz_sum;
            flux_i_momentum.y     += factor_z * mom_flux_yz_sum;
            flux_i_momentum.z     += factor_z * mom_flux_zz_sum;

        } else if (nb == -1) {
            flux_i_momentum.x += normal_x * pressure_i;
            flux_i_momentum.y += normal_y * pressure_i;
            flux_i_momentum.z += normal_z * pressure_i;
        } else if (nb == -2) {
            const float factor_x = 0.5f * normal_x;
            const float factor_y = 0.5f * normal_y;
            const float factor_z = 0.5f * normal_z;

            // x-normal contribution
            flux_i_density        += factor_x * (ff_variable[VAR_MOMENTUM + 0] + momentum_i.x);
            flux_i_density_energy += factor_x * (ff_flux_contribution_density_energy[0].x +
                                                 flux_contribution_i_density_energy.x);
            flux_i_momentum.x     += factor_x * (ff_flux_contribution_momentum_x[0].x +
                                                 flux_contribution_i_momentum_x.x);
            flux_i_momentum.y     += factor_x * (ff_flux_contribution_momentum_y[0].x +
                                                 flux_contribution_i_momentum_y.x);
            flux_i_momentum.z     += factor_x * (ff_flux_contribution_momentum_z[0].x +
                                                 flux_contribution_i_momentum_z.x);

            // y-normal contribution
            flux_i_density        += factor_y * (ff_variable[VAR_MOMENTUM + 1] + momentum_i.y);
            flux_i_density_energy += factor_y * (ff_flux_contribution_density_energy[0].y +
                                                 flux_contribution_i_density_energy.y);
            flux_i_momentum.x     += factor_y * (ff_flux_contribution_momentum_x[0].y +
                                                 flux_contribution_i_momentum_x.y);
            flux_i_momentum.y     += factor_y * (ff_flux_contribution_momentum_y[0].y +
                                                 flux_contribution_i_momentum_y.y);
            flux_i_momentum.z     += factor_y * (ff_flux_contribution_momentum_z[0].y +
                                                 flux_contribution_i_momentum_z.y);

            // z-normal contribution
            flux_i_density        += factor_z * (ff_variable[VAR_MOMENTUM + 2] + momentum_i.z);
            flux_i_density_energy += factor_z * (ff_flux_contribution_density_energy[0].z +
                                                 flux_contribution_i_density_energy.z);
            flux_i_momentum.x     += factor_z * (ff_flux_contribution_momentum_x[0].z +
                                                 flux_contribution_i_momentum_x.z);
            flux_i_momentum.y     += factor_z * (ff_flux_contribution_momentum_y[0].z +
                                                 flux_contribution_i_momentum_y.z);
            flux_i_momentum.z     += factor_z * (ff_flux_contribution_momentum_z[0].z +
                                                 flux_contribution_i_momentum_z.z);
        }
    }

    fluxes[i + var_density_offset]        = flux_i_density;
    fluxes[i + var_momentum0_offset]      = flux_i_momentum.x;
    fluxes[i + var_momentum1_offset]      = flux_i_momentum.y;
    fluxes[i + var_momentum2_offset]      = flux_i_momentum.z;
    fluxes[i + var_density_energy_offset] = flux_i_density_energy;
}

void compute_flux(int nelr, int *elements_surrounding_elements, float *normals,
                  float *variables, float *fluxes) {
    dim3 Dg(nelr / BLOCK_SIZE), Db(BLOCK_SIZE);
    cuda_compute_flux<<<Dg, Db>>>(nelr, elements_surrounding_elements, normals,
                                  variables, fluxes);
    getLastCudaError("compute_flux failed");
}

__global__ void cuda_time_step(int j, int nelr, float *old_variables,
                               float *variables, float *step_factors,
                               float *fluxes) {
    const int i = (blockDim.x * blockIdx.x + threadIdx.x);

    float factor = step_factors[i] / float(RK + 1 - j);

    variables[i + VAR_DENSITY * nelr] = old_variables[i + VAR_DENSITY * nelr] +
                                        factor * fluxes[i + VAR_DENSITY * nelr];
    variables[i + VAR_DENSITY_ENERGY * nelr] =
        old_variables[i + VAR_DENSITY_ENERGY * nelr] +
        factor * fluxes[i + VAR_DENSITY_ENERGY * nelr];
    variables[i + (VAR_MOMENTUM + 0) * nelr] =
        old_variables[i + (VAR_MOMENTUM + 0) * nelr] +
        factor * fluxes[i + (VAR_MOMENTUM + 0) * nelr];
    variables[i + (VAR_MOMENTUM + 1) * nelr] =
        old_variables[i + (VAR_MOMENTUM + 1) * nelr] +
        factor * fluxes[i + (VAR_MOMENTUM + 1) * nelr];
    variables[i + (VAR_MOMENTUM + 2) * nelr] =
        old_variables[i + (VAR_MOMENTUM + 2) * nelr] +
        factor * fluxes[i + (VAR_MOMENTUM + 2) * nelr];
}
void time_step(int j, int nelr, float *old_variables, float *variables,
               float *step_factors, float *fluxes) {
    dim3 Dg(nelr / BLOCK_SIZE), Db(BLOCK_SIZE);
    cuda_time_step<<<Dg, Db>>>(j, nelr, old_variables, variables, step_factors,
                               fluxes);
    getLastCudaError("update failed");
}

/*
 * Main function
 */
int main(int argc, char **argv) {
    struct timespec main_start, main_end;
    clock_gettime(CLOCK_MONOTONIC, &main_start);
    
    printf("WG size of kernel:initialize = %d, WG size of "
           "kernel:compute_step_factor = %d, WG size of kernel:compute_flux = "
           "%d, WG size of kernel:time_step = %d\n",
           BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <input_file> [output_file]" << std::endl;
        return 0;
    }
    const char *data_file_name = argv[1];
    const char *output_file = (argc >= 3) ? argv[2] : NULL;

    cudaDeviceProp prop;
    int dev;

    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaGetDevice(&dev));
    checkCudaErrors(cudaGetDeviceProperties(&prop, dev));

    printf("Name:                     %s\n", prop.name);

    // set far field conditions and load them into constant memory on the gpu
    {
        float h_ff_variable[NVAR];
        const float angle_of_attack =
            float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);

        h_ff_variable[VAR_DENSITY] = float(1.4);

        float ff_pressure = float(1.0f);
        float ff_speed_of_sound =
            sqrt(GAMMA * ff_pressure / h_ff_variable[VAR_DENSITY]);
        float ff_speed = float(ff_mach) * ff_speed_of_sound;

        float3 ff_velocity;
        ff_velocity.x = ff_speed * float(cos((float)angle_of_attack));
        ff_velocity.y = ff_speed * float(sin((float)angle_of_attack));
        ff_velocity.z = 0.0f;

        h_ff_variable[VAR_MOMENTUM + 0] =
            h_ff_variable[VAR_DENSITY] * ff_velocity.x;
        h_ff_variable[VAR_MOMENTUM + 1] =
            h_ff_variable[VAR_DENSITY] * ff_velocity.y;
        h_ff_variable[VAR_MOMENTUM + 2] =
            h_ff_variable[VAR_DENSITY] * ff_velocity.z;

        h_ff_variable[VAR_DENSITY_ENERGY] =
            h_ff_variable[VAR_DENSITY] * (float(0.5f) * (ff_speed * ff_speed)) +
            (ff_pressure / float(GAMMA - 1.0f));

        float3 h_ff_momentum;
        h_ff_momentum.x = *(h_ff_variable + VAR_MOMENTUM + 0);
        h_ff_momentum.y = *(h_ff_variable + VAR_MOMENTUM + 1);
        h_ff_momentum.z = *(h_ff_variable + VAR_MOMENTUM + 2);
        float3 h_ff_flux_contribution_momentum_x;
        float3 h_ff_flux_contribution_momentum_y;
        float3 h_ff_flux_contribution_momentum_z;
        float3 h_ff_flux_contribution_density_energy;
        compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_momentum,
                                  h_ff_variable[VAR_DENSITY_ENERGY],
                                  ff_pressure, ff_velocity,
                                  h_ff_flux_contribution_momentum_x,
                                  h_ff_flux_contribution_momentum_y,
                                  h_ff_flux_contribution_momentum_z,
                                  h_ff_flux_contribution_density_energy);

        // copy far field conditions to the gpu
        checkCudaErrors(cudaMemcpyToSymbol(ff_variable, h_ff_variable,
                                           NVAR * sizeof(float)));
        checkCudaErrors(cudaMemcpyToSymbol(ff_flux_contribution_momentum_x,
                                           &h_ff_flux_contribution_momentum_x,
                                           sizeof(float3)));
        checkCudaErrors(cudaMemcpyToSymbol(ff_flux_contribution_momentum_y,
                                           &h_ff_flux_contribution_momentum_y,
                                           sizeof(float3)));
        checkCudaErrors(cudaMemcpyToSymbol(ff_flux_contribution_momentum_z,
                                           &h_ff_flux_contribution_momentum_z,
                                           sizeof(float3)));

        checkCudaErrors(cudaMemcpyToSymbol(
            ff_flux_contribution_density_energy,
            &h_ff_flux_contribution_density_energy, sizeof(float3)));
    }
    int nel;
    int nelr;

    // read in domain geometry
    float *areas;
    int *elements_surrounding_elements;
    float *normals;
    {
        std::ifstream file(data_file_name);

        file >> nel;
        nelr =
            BLOCK_SIZE * ((nel / BLOCK_SIZE) + std::min(1, nel % BLOCK_SIZE));

        float *h_areas = new float[nelr];
        int *h_elements_surrounding_elements = new int[nelr * NNB];
        float *h_normals = new float[nelr * NDIM * NNB];


        // read in data
        for (int i = 0; i < nel; i++) {
            file >> h_areas[i];
            for (int j = 0; j < NNB; j++) {
                file >> h_elements_surrounding_elements[i + j * nelr];
                if (h_elements_surrounding_elements[i + j * nelr] < 0)
                    h_elements_surrounding_elements[i + j * nelr] = -1;
                h_elements_surrounding_elements[i + j * nelr]--; // it's coming
                                                                 // in with
                                                                 // Fortran
                                                                 // numbering

                for (int k = 0; k < NDIM; k++) {
                    file >> h_normals[i + (j + k * NNB) * nelr];
                    h_normals[i + (j + k * NNB) * nelr] =
                        -h_normals[i + (j + k * NNB) * nelr];
                }
            }
        }

        // fill in remaining data
        int last = nel - 1;
        for (int i = nel; i < nelr; i++) {
            h_areas[i] = h_areas[last];
            for (int j = 0; j < NNB; j++) {
                // duplicate the last element
                h_elements_surrounding_elements[i + j * nelr] =
                    h_elements_surrounding_elements[last + j * nelr];
                for (int k = 0; k < NDIM; k++)
                    h_normals[last + (j + k * NNB) * nelr] =
                        h_normals[last + (j + k * NNB) * nelr];
            }
        }

        areas = alloc<float>(nelr);
        upload<float>(areas, h_areas, nelr);

        elements_surrounding_elements = alloc<int>(nelr * NNB);
        upload<int>(elements_surrounding_elements,
                    h_elements_surrounding_elements, nelr * NNB);

        normals = alloc<float>(nelr * NDIM * NNB);
        upload<float>(normals, h_normals, nelr * NDIM * NNB);

        delete[] h_areas;
        delete[] h_elements_surrounding_elements;
        delete[] h_normals;
    }

    // Create arrays and set initial conditions
    float *variables = alloc<float>(nelr * NVAR);
    initialize_variables(nelr, variables);

    float *old_variables = alloc<float>(nelr * NVAR);
    float *fluxes = alloc<float>(nelr * NVAR);
    float *step_factors = alloc<float>(nelr);

    // make sure all memory is doublely allocated before we start timing
    initialize_variables(nelr, old_variables);
    initialize_variables(nelr, fluxes);
    cudaMemset((void *)step_factors, 0, sizeof(float) * nelr);
    // make sure CUDA isn't still doing something before we start timing
    cudaDeviceSynchronize();

    // these need to be computed the first time in order to compute time step
    std::cout << "Starting..." << std::endl;

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    struct timespec kernel_start, kernel_end;
    double total_kernel_time = 0.0;

    // Begin iterations
    for (int i = 0; i < iterations; i++) {
        copy<float>(old_variables, variables, nelr * NVAR);

        // for the first iteration we compute the time step
        compute_step_factor(nelr, variables, areas, step_factors);
        getLastCudaError("compute_step_factor failed");

        for (int j = 0; j < RK; j++) {
            // Time only cuda_compute_flux kernel
            clock_gettime(CLOCK_MONOTONIC, &kernel_start);
            compute_flux(nelr, elements_surrounding_elements, normals,
                         variables, fluxes);
            cudaDeviceSynchronize();
            clock_gettime(CLOCK_MONOTONIC, &kernel_end);
            total_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) + 
                                 (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
            getLastCudaError("compute_flux failed");
            
            time_step(j, nelr, old_variables, variables, step_factors, fluxes);
            getLastCudaError("time_step failed");
        }
    }

    sdkStopTimer(&timer);

    std::cout << (sdkGetAverageTimerValue(&timer) / 1000.0) / iterations
              << " seconds per iteration" << std::endl;

    if (output_file != NULL) {
        std::cout << "Saving solution..." << std::endl;
        dump(variables, nel, nelr, output_file);
        std::cout << "Saved solution..." << std::endl;
    } else if (getenv("OUTPUT")) {
        std::cout << "Saving solution..." << std::endl;
        dump(variables, nel, nelr);
        std::cout << "Saved solution..." << std::endl;
    }

    std::cout << "Cleaning up..." << std::endl;
    dealloc<float>(areas);
    dealloc<int>(elements_surrounding_elements);
    dealloc<float>(normals);

    dealloc<float>(variables);
    dealloc<float>(old_variables);
    dealloc<float>(fluxes);
    dealloc<float>(step_factors);

    std::cout << "Done..." << std::endl;

    // Timing output
    clock_gettime(CLOCK_MONOTONIC, &main_end);
    double main_time = (main_end.tv_sec - main_start.tv_sec) + 
                       (main_end.tv_nsec - main_start.tv_nsec) / 1e9;
    
    FILE *timing_file = stderr;
    const char *timing_path = getenv("TIMING_LOG_FILE");
    if (timing_path && timing_path[0] != '\0') {
        FILE *tmp = fopen(timing_path, "w");
        if (tmp)
            timing_file = tmp;
    }
    
    fprintf(timing_file, "KERNEL_TIME: %.9f\n", total_kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);
    
    if (timing_file != stderr)
        fclose(timing_file);

    return 0;
}
