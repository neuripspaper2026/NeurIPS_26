#include "lavaMD.h"
#include <cuda_runtime.h>
#include <math_constants.h>

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
        int k;
        int first_j;
        FOUR_VECTOR *rB;
        fp *qB;
        int j;
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

        // home box - shared memory, use unrolled loop for better ILP
        for (int idx = wtx; idx < NUMBER_PAR_PER_BOX; idx += NUMBER_THREADS) {
            rA_shared[idx] = rA[idx];
        }

        // synchronize threads  - not strictly needed, but just to be safe
        __syncthreads();

        //------------------------------------------------------------------------------------------------------------------------------------------------------160
        //	nei box loop
        //------------------------------------------------------------------------------------------------------------------------------------------------------160

        // loop over neiing boxes of home box
        const int nnei = d_box_gpu[bx].nn;
        for (k = 0; k < (1 + nnei); k++) {

            //----------------------------------------50
            //	nei box - get pointer to the right box
            //----------------------------------------50

            if (k == 0) {
                pointer = bx; // set first box to be processed to home box
            } else {
                pointer = d_box_gpu[bx].nei[k - 1].number; // remaining boxes
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
            //	Copy nei box data to shared memory
            //----------------------------------------------------------------------------------------------------------------------------------140

            // load neighbor box positions and charges into shared memory
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

                // keep local copy of rA to reduce shared memory traffic
                const FOUR_VECTOR rAi = rA_shared[i];

                // loop for the number of particles in the current nei box
                // unroll for better ILP and to help compiler schedule
                #pragma unroll 4
                for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {

                    const FOUR_VECTOR rBj = rB_shared[j];
                    const fp qBj = qB_shared[j];

                    // r2 = rA[i].v + rB[j].v - DOT(rA[i],rB[j]);
                    r2 = rAi.v + rBj.v -
                         (rAi.x * rBj.x + rAi.y * rBj.y + rAi.z * rBj.z);
                    u2 = a2 * r2;

                    // using fast expf for single precision; sufficient for fp=float
                    vij = __expf(-u2);
                    fs = (fp)2.0 * vij;

                    d.x = rAi.x - rBj.x;
                    d.y = rAi.y - rBj.y;
                    d.z = rAi.z - rBj.z;

                    fxij = fs * d.x;
                    fyij = fs * d.y;
                    fzij = fs * d.z;

                    const fp qv = qBj;

                    // accumulate to global force array; keep casts minimal
                    fA[i].v += (fp)(qv * vij);
                    fA[i].x += (fp)(qv * fxij);
                    fA[i].y += (fp)(qv * fyij);
                    fA[i].z += (fp)(qv * fzij);
                }
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
