#include "lavaMD.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void kernel_gpu_cuda(par_str d_par_gpu, dim_str d_dim_gpu,
                                box_str *d_box_gpu, FOUR_VECTOR *d_rv_gpu,
                                fp *d_qv_gpu, FOUR_VECTOR *d_fv_gpu) {

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
    //  THREAD PARAMETERS
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

    int bx = blockIdx.x;  // get current horizontal block index (0-n)
    int tx = threadIdx.x; // get current horizontal thread index (0-n)
    int wtx = tx;

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
    //  DO FOR THE NUMBER OF BOXES
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

    if (bx < d_dim_gpu.number_boxes) {

        //------------------------------------------------------------------------------------------------------------------------------------------------------160
        //  Extract input parameters
        //------------------------------------------------------------------------------------------------------------------------------------------------------160

        // parameters
        fp a2 = (fp)2.0 * d_par_gpu.alpha * d_par_gpu.alpha;

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
        //  Home box
        //------------------------------------------------------------------------------------------------------------------------------------------------------160

        //----------------------------------------------------------------------------------------------------------------------------------140
        //  Setup parameters
        //----------------------------------------------------------------------------------------------------------------------------------140

        // home box - box parameters
        first_i = d_box_gpu[bx].offset;

        // home box - distance, force, charge and type parameters
        rA = &d_rv_gpu[first_i];
        fA = &d_fv_gpu[first_i];

        //----------------------------------------------------------------------------------------------------------------------------------140
        //  Copy to shared memory
        //----------------------------------------------------------------------------------------------------------------------------------140

        // home box - shared memory
        for (int idx = wtx; idx < NUMBER_PAR_PER_BOX; idx += blockDim.x) {
            rA_shared[idx] = rA[idx];
        }
        __syncthreads();

        //------------------------------------------------------------------------------------------------------------------------------------------------------160
        //  nei box loop
        //------------------------------------------------------------------------------------------------------------------------------------------------------160

        // loop over neighboring boxes of home box
        int nnei = 1 + d_box_gpu[bx].nn;
        for (k = 0; k < nnei; k++) {

            //----------------------------------------50
            //  nei box - get pointer to the right box
            //----------------------------------------50

            if (k == 0) {
                pointer = bx; // set first box to be processed to home box
            } else {
                pointer = d_box_gpu[bx].nei[k - 1].number; // remaining boxes are nei boxes
            }

            //----------------------------------------------------------------------------------------------------------------------------------140
            //  Setup parameters
            //----------------------------------------------------------------------------------------------------------------------------------140

            // nei box - box parameters
            first_j = d_box_gpu[pointer].offset;

            // nei box - distance, (force), charge and (type) parameters
            rB = &d_rv_gpu[first_j];
            qB = &d_qv_gpu[first_j];

            //----------------------------------------------------------------------------------------------------------------------------------140
            //  Copy neighbor box to shared memory
            //----------------------------------------------------------------------------------------------------------------------------------140

            for (int idx = wtx; idx < NUMBER_PAR_PER_BOX; idx += blockDim.x) {
                rB_shared[idx] = rB[idx];
                qB_shared[idx] = qB[idx];
            }

            // synchronize threads because in next section each thread accesses
            // data brought in by different threads here
            __syncthreads();

            //----------------------------------------------------------------------------------------------------------------------------------140
            //  Calculation
            //----------------------------------------------------------------------------------------------------------------------------------140

            // loop for the number of particles in the home box
            for (int i = wtx; i < NUMBER_PAR_PER_BOX; i += blockDim.x) {

                // private accumulator in registers to avoid repeated global memory atomics
                fp acc_v = (fp)0.0f;
                fp acc_x = (fp)0.0f;
                fp acc_y = (fp)0.0f;
                fp acc_z = (fp)0.0f;

                FOUR_VECTOR rA_i = rA_shared[i];

#pragma unroll 4
                for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {

                    FOUR_VECTOR rB_j = rB_shared[j];
                    fp qB_j = qB_shared[j];

                    fp dot = rA_i.x * rB_j.x + rA_i.y * rB_j.y + rA_i.z * rB_j.z;
                    r2 = rA_i.v + rB_j.v - dot;
                    u2 = a2 * r2;

                    // use fast intrinsic for expf
                    vij = __expf(-u2);
                    fs = (fp)2.0f * vij;

                    d.x = rA_i.x - rB_j.x;
                    d.y = rA_i.y - rB_j.y;
                    d.z = rA_i.z - rB_j.z;

                    fxij = fs * d.x;
                    fyij = fs * d.y;
                    fzij = fs * d.z;

                    fp qv = qB_j;
                    acc_v += qv * vij;
                    acc_x += qv * fxij;
                    acc_y += qv * fyij;
                    acc_z += qv * fzij;
                }

                // accumulate results to global memory
                fA[i].v += (double)acc_v;
                fA[i].x += (double)acc_x;
                fA[i].y += (double)acc_y;
                fA[i].z += (double)acc_z;
            }

            // synchronize after finishing force contributions from current nei
            // box not to cause conflicts when starting next box
            __syncthreads();

            //----------------------------------------------------------------------------------------------------------------------------------140
            //  Calculation END
            //----------------------------------------------------------------------------------------------------------------------------------140
        }
    }
}
