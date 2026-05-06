#include "lavaMD.h"
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel_gpu_cuda(par_str d_par_gpu, dim_str d_dim_gpu,
                                box_str *d_box_gpu, FOUR_VECTOR *d_rv_gpu,
                                fp *d_qv_gpu, FOUR_VECTOR *d_fv_gpu) {

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
    //	THREAD PARAMETERS
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

    int bx = blockIdx.x;  // get current horizontal block index (0-n)
    int tx = threadIdx.x; // get current horizontal thread index (0-n)
    int wtx = tx;

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
    //	DO FOR THE NUMBER OF BOXES
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

    if (bx < d_dim_gpu.number_boxes) {

        //------------------------------------------------------------------------------------------------------------------------------------------------------160
        //	Extract input parameters
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
        __shared__ fp qB_shared_fp[NUMBER_PAR_PER_BOX];

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
        while (wtx < NUMBER_PAR_PER_BOX) {
            rA_shared[wtx] = rA[wtx];
            wtx = wtx + NUMBER_THREADS;
        }
        wtx = tx;

        // synchronize threads  - not needed, but just to be safe
        __syncthreads();

        //------------------------------------------------------------------------------------------------------------------------------------------------------160
        //	nei box loop
        //------------------------------------------------------------------------------------------------------------------------------------------------------160

        // loop over neiing boxes of home box
        int nnei = 1 + d_box_gpu[bx].nn;
        for (k = 0; k < nnei; k++) {

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
            //	Setup parameters
            //----------------------------------------------------------------------------------------------------------------------------------140

            // nei box - shared memory
            while (wtx < NUMBER_PAR_PER_BOX) {
                rB_shared[wtx] = rB[wtx];
                qB_shared_fp[wtx] = qB[wtx];
                wtx = wtx + NUMBER_THREADS;
            }
            wtx = tx;

            // synchronize threads because in next section each thread accesses
            // data brought in by different threads here
            __syncthreads();

            //----------------------------------------------------------------------------------------------------------------------------------140
            //	Calculation
            //----------------------------------------------------------------------------------------------------------------------------------140

            // loop for the number of particles in the home box
            while (wtx < NUMBER_PAR_PER_BOX) {

                FOUR_VECTOR rA_loc = rA_shared[wtx];
                fp fAv = 0.0f;
                fp fAx = 0.0f;
                fp fAy = 0.0f;
                fp fAz = 0.0f;

#pragma unroll 4
                for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {

                    FOUR_VECTOR rB_loc = rB_shared[j];
                    fp qB_val = qB_shared_fp[j];

                    r2 = rA_loc.v + rB_loc.v -
                         (rA_loc.x * rB_loc.x + rA_loc.y * rB_loc.y +
                          rA_loc.z * rB_loc.z);

                    u2 = a2 * r2;
                    vij = __expf(-u2);
                    fs = (fp)2.0 * vij;

                    d.x = rA_loc.x - rB_loc.x;
                    fxij = fs * d.x;
                    d.y = rA_loc.y - rB_loc.y;
                    fyij = fs * d.y;
                    d.z = rA_loc.z - rB_loc.z;
                    fzij = fs * d.z;

                    fp scaled = qB_val * vij;
                    fAv += scaled;
                    fAx += qB_val * fxij;
                    fAy += qB_val * fyij;
                    fAz += qB_val * fzij;
                }

                // accumulate to global memory
                fA[wtx].v += (double)fAv;
                fA[wtx].x += (double)fAx;
                fA[wtx].y += (double)fAy;
                fA[wtx].z += (double)fAz;

                // increment work thread index
                wtx = wtx + NUMBER_THREADS;
            }

            // reset work index
            wtx = tx;

            // synchronize after finishing force contributions from current nei
            // box not to cause conflicts when starting next box
            __syncthreads();

            //----------------------------------------------------------------------------------------------------------------------------------140
            //	Calculation END
            //----------------------------------------------------------------------------------------------------------------------------------140
        }
    }
}
