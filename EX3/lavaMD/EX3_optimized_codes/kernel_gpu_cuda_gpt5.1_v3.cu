#include "lavaMD.h"
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
        fp alpha = d_par_gpu.alpha;
        fp a2 = 2.0f * alpha * alpha;

        // home box
        int first_i;
        FOUR_VECTOR *rA;
        FOUR_VECTOR *fA;

        extern __shared__ unsigned char shmem[];
        FOUR_VECTOR *rA_shared = (FOUR_VECTOR *)shmem;
        FOUR_VECTOR *rB_shared = (FOUR_VECTOR *)(shmem + sizeof(FOUR_VECTOR) * NUMBER_PAR_PER_BOX);
        fp *qB_shared = (fp *)(shmem + sizeof(FOUR_VECTOR) * NUMBER_PAR_PER_BOX * 2);

        // nei box
        int pointer;
        int k = 0;
        int first_j;
        FOUR_VECTOR *rB;
        fp *qB;
        int j = 0;

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
        for (int idx = wtx; idx < NUMBER_PAR_PER_BOX; idx += NUMBER_THREADS) {
            rA_shared[idx] = rA[idx];
        }

        // synchronize threads  - not needed, but just to be safe
        __syncthreads();

        //------------------------------------------------------------------------------------------------------------------------------------------------------160
        //  nei box loop
        //------------------------------------------------------------------------------------------------------------------------------------------------------160

        // loop over neiing boxes of home box
        int nn_local = d_box_gpu[bx].nn;
        for (k = 0; k < (1 + nn_local); k++) {

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
            //  Copy nei box to shared memory
            //----------------------------------------------------------------------------------------------------------------------------------140

            for (int idx = wtx; idx < NUMBER_PAR_PER_BOX; idx += NUMBER_THREADS) {
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
            for (int i = wtx; i < NUMBER_PAR_PER_BOX; i += NUMBER_THREADS) {

                // cache rA particle in registers
                FOUR_VECTOR rAi = rA_shared[i];

                fp fvv = 0.0f;
                fp fvx = 0.0f;
                fp fvy = 0.0f;
                fp fvz = 0.0f;

                // loop for the number of particles in the current nei box
#pragma unroll 4
                for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {

                    FOUR_VECTOR rBj = rB_shared[j];
                    fp qBj = qB_shared[j];

                    fp dot = rAi.x * rBj.x + rAi.y * rBj.y + rAi.z * rBj.z;
                    r2 = rAi.v + rBj.v - dot;
                    u2 = a2 * r2;
                    vij = __expf(-u2);
                    fs = 2.0f * vij;

                    fp dx = rAi.x - rBj.x;
                    fp dy = rAi.y - rBj.y;
                    fp dz = rAi.z - rBj.z;

                    fxij = fs * dx;
                    fyij = fs * dy;
                    fzij = fs * dz;

                    fp qv = qBj;

                    fvv += qv * vij;
                    fvx += qv * fxij;
                    fvy += qv * fyij;
                    fvz += qv * fzij;
                }

                // store back to global memory
                FOUR_VECTOR fAv = fA[i];
                fAv.v += (fp)fvv;
                fAv.x += (fp)fvx;
                fAv.y += (fp)fvy;
                fAv.z += (fp)fvz;
                fA[i] = fAv;
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
