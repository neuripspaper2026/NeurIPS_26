#include "lavaMD.h"
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel_gpu_cuda(par_str d_par_gpu, dim_str d_dim_gpu,
                                box_str *d_box_gpu, FOUR_VECTOR *d_rv_gpu,
                                fp *d_qv_gpu, FOUR_VECTOR *d_fv_gpu) {

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
    //	THREAD PARAMETERS
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

    const int bx = blockIdx.x;   // get current horizontal block index (0-n)
    const int tx = threadIdx.x;  // get current horizontal thread index (0-n)
    int wtx = tx;

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
    //	DO FOR THE NUMBER OF BOXES
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

    if (bx < d_dim_gpu.number_boxes) {

        //------------------------------------------------------------------------------------------------------------------------------------------------------160
        //	Extract input parameters
        //------------------------------------------------------------------------------------------------------------------------------------------------------160

        // parameters
        const fp a2 = (fp)2.0 * d_par_gpu.alpha * d_par_gpu.alpha;

        // home box
        int first_i;
        FOUR_VECTOR *rA;
        FOUR_VECTOR *fA;
        __shared__ FOUR_VECTOR rA_shared[NUMBER_PAR_PER_BOX];

        // nei box
        int pointer;
        int k = 0;
        int first_j;
        FOUR_VECTOR *rB;
        fp *qB;
        int j = 0;
        __shared__ FOUR_VECTOR rB_shared[NUMBER_PAR_PER_BOX];
        __shared__ fp qB_shared[NUMBER_PAR_PER_BOX];

        // common
        fp r2;
        fp u2;
        fp vij;
        fp fs;
        fp fxij;
        fp fyij;
        fp fzij;
        THREE_VECTOR d;

        //------------------------------------------------------------------------------------------------------------------------------------------------------160
        //	Home box
        //------------------------------------------------------------------------------------------------------------------------------------------------------160

        //----------------------------------------------------------------------------------------------------------------------------------140
        //	Setup parameters
        //----------------------------------------------------------------------------------------------------------------------------------140

        // home box - box parameters
        first_i = d_box_gpu[bx].offset;

        // home box - distance, force, charge and type parameters
        rA = &d_rv_gpu[first_i];
        fA = &d_fv_gpu[first_i];

        //----------------------------------------------------------------------------------------------------------------------------------140
        //	Copy to shared memory
        //----------------------------------------------------------------------------------------------------------------------------------140

        // home box - shared memory
        for (int idx = wtx; idx < NUMBER_PAR_PER_BOX; idx += NUMBER_THREADS) {
            rA_shared[idx] = rA[idx];
        }

        // synchronize threads  - not strictly needed if all threads participate, but kept for safety
        __syncthreads();

        //------------------------------------------------------------------------------------------------------------------------------------------------------160
        //	nei box loop
        //------------------------------------------------------------------------------------------------------------------------------------------------------160

        // loop over neighboring boxes of home box
        const int nn = d_box_gpu[bx].nn;
        for (k = 0; k < (1 + nn); k++) {

            //----------------------------------------50
            //	nei box - get pointer to the right box
            //----------------------------------------50

            if (k == 0) {
                pointer = bx; // set first box to be processed to home box
            } else {
                pointer = d_box_gpu[bx].nei[k - 1].number; // remaining boxes are nei boxes
            }

            //----------------------------------------------------------------------------------------------------------------------------------140
            //	Setup parameters
            //----------------------------------------------------------------------------------------------------------------------------------140

            // nei box - box parameters
            first_j = d_box_gpu[pointer].offset;

            // nei box - distance, (force), charge and (type) parameters
            rB = &d_rv_gpu[first_j];
            qB = &d_qv_gpu[first_j];

            //----------------------------------------------------------------------------------------------------------------------------------140
            //	Copy nei box to shared memory
            //----------------------------------------------------------------------------------------------------------------------------------140

            for (int idx = wtx; idx < NUMBER_PAR_PER_BOX; idx += NUMBER_THREADS) {
                rB_shared[idx] = rB[idx];
                qB_shared[idx] = qB[idx];
            }

            // synchronize threads because in next section each thread accesses
            // data brought in by different threads here
            __syncthreads();

            //----------------------------------------------------------------------------------------------------------------------------------140
            //	Calculation
            //----------------------------------------------------------------------------------------------------------------------------------140

            // loop for the number of particles in the home box
            for (int i = wtx; i < NUMBER_PAR_PER_BOX; i += NUMBER_THREADS) {

                // cache home particle in registers
                const fp rA_v = rA_shared[i].v;
                const fp rA_x = rA_shared[i].x;
                const fp rA_y = rA_shared[i].y;
                const fp rA_z = rA_shared[i].z;

                // local accumulators in fp for better ILP and to avoid repeated global updates
                fp acc_v = 0.0f;
                fp acc_x = 0.0f;
                fp acc_y = 0.0f;
                fp acc_z = 0.0f;

                // unroll inner loop for better throughput
#pragma unroll 4
                for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {

                    const fp rB_v = rB_shared[j].v;
                    const fp rB_x = rB_shared[j].x;
                    const fp rB_y = rB_shared[j].y;
                    const fp rB_z = rB_shared[j].z;
                    const fp qB_j = qB_shared[j];

                    const fp dx = rA_x - rB_x;
                    const fp dy = rA_y - rB_y;
                    const fp dz = rA_z - rB_z;

                    // r2 = rA[i].v + rB[j].v - DOT(rA[i],rB[j]);
                    r2 = rA_v + rB_v - (rA_x * rB_x + rA_y * rB_y + rA_z * rB_z);
                    u2 = a2 * r2;

                    // use fast single-precision exponential
                    vij = __expf(-u2);
                    fs = (fp)2.0 * vij;

                    fxij = fs * dx;
                    fyij = fs * dy;
                    fzij = fs * dz;

                    acc_v += qB_j * vij;
                    acc_x += qB_j * fxij;
                    acc_y += qB_j * fyij;
                    acc_z += qB_j * fzij;
                }

                // single global memory update per particle per neighbor box
                fA[i].v += (fp)acc_v;
                fA[i].x += (fp)acc_x;
                fA[i].y += (fp)acc_y;
                fA[i].z += (fp)acc_z;
            }

            // synchronize after finishing force contributions from current nei
            // box not to cause conflicts when starting next box
            __syncthreads();

            //----------------------------------------------------------------------------------------------------------------------------------140
            //	Calculation END
            //----------------------------------------------------------------------------------------------------------------------------------140
        }
    }
}
