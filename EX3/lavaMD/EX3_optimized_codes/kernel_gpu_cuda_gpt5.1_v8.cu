#include "lavaMD.h"
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void kernel_gpu_cuda(par_str d_par_gpu, dim_str d_dim_gpu,
                                box_str * __restrict__ d_box_gpu,
                                FOUR_VECTOR * __restrict__ d_rv_gpu,
                                fp * __restrict__ d_qv_gpu,
                                FOUR_VECTOR * __restrict__ d_fv_gpu) {

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
    //  THREAD PARAMETERS
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

    int bx = blockIdx.x;   // get current horizontal block index (0-n)
    int tx = threadIdx.x;  // get current horizontal thread index (0-n)
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
        FOUR_VECTOR * __restrict__ rA;
        FOUR_VECTOR * __restrict__ fA;
        __shared__ FOUR_VECTOR rA_shared[NUMBER_PAR_PER_BOX];

        // nei box
        int pointer;
        int k = 0;
        int first_j;
        FOUR_VECTOR * __restrict__ rB;
        fp * __restrict__ qB;
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
        for (int ii = wtx; ii < NUMBER_PAR_PER_BOX; ii += NUMBER_THREADS) {
            rA_shared[ii] = rA[ii];
        }
        wtx = tx;

        // synchronize threads  - not needed, but just to be safe
        __syncthreads();

        //------------------------------------------------------------------------------------------------------------------------------------------------------160
        //  nei box loop
        //------------------------------------------------------------------------------------------------------------------------------------------------------160

        // loop over neighboring boxes of home box (including itself)
        int nnei = d_box_gpu[bx].nn;
        for (k = 0; k < (1 + nnei); k++) {

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
            //  Copy neighbor box data to shared memory
            //----------------------------------------------------------------------------------------------------------------------------------140

            for (int jj = wtx; jj < NUMBER_PAR_PER_BOX; jj += NUMBER_THREADS) {
                rB_shared[jj] = rB[jj];
                qB_shared[jj] = qB[jj];
            }
            wtx = tx;

            // synchronize threads because in next section each thread accesses
            // data brought in by different threads here
            __syncthreads();

            //----------------------------------------------------------------------------------------------------------------------------------140
            //  Calculation
            //----------------------------------------------------------------------------------------------------------------------------------140

            // loop for the number of particles in the home box
            for (int ii = wtx; ii < NUMBER_PAR_PER_BOX; ii += NUMBER_THREADS) {

                FOUR_VECTOR rAi = rA_shared[ii];
                FOUR_VECTOR fAi = fA[ii];

                // unroll inner loop for better ILP
                #pragma unroll 4
                for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {

                    FOUR_VECTOR rBj = rB_shared[j];
                    fp qBj = qB_shared[j];

                    // r2 = rAi.v + rBj.v - DOT(rAi,rBj);
                    fp dot = rAi.x * rBj.x + rAi.y * rBj.y + rAi.z * rBj.z;
                    r2 = rAi.v + rBj.v - dot;
                    u2 = a2 * r2;
                    vij = __expf(-u2);
                    fs = (fp)2.0 * vij;

                    d.x = rAi.x - rBj.x;
                    d.y = rAi.y - rBj.y;
                    d.z = rAi.z - rBj.z;

                    fxij = fs * d.x;
                    fyij = fs * d.y;
                    fzij = fs * d.z;

                    fp qv = qBj * vij;
                    fp qfx = qBj * fxij;
                    fp qfy = qBj * fyij;
                    fp qfz = qBj * fzij;

                    fAi.v += qv;
                    fAi.x += qfx;
                    fAi.y += qfy;
                    fAi.z += qfz;
                }

                // write back accumulated forces for this particle
                fA[ii] = fAi;
            }

            // reset work index
            wtx = tx;

            // synchronize after finishing force contributions from current nei
            // box not to cause conflicts when starting next box
            __syncthreads();

            //----------------------------------------------------------------------------------------------------------------------------------140
            //  Calculation END
            //----------------------------------------------------------------------------------------------------------------------------------140
        }
    }
}
