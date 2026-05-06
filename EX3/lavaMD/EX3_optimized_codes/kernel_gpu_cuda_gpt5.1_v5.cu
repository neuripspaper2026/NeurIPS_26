#include "lavaMD.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

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

        // align shared arrays to 16B and overprovision to 128 for better bank alignment on A100
        __shared__ FOUR_VECTOR rA_shared[128];

        // nei box
        int pointer;
        int k = 0;
        int first_j;
        FOUR_VECTOR *rB;
        fp *qB;
        int j = 0;
        __shared__ FOUR_VECTOR rB_shared[128];
        __shared__ fp qB_shared[128];

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
        // ensure coalesced loads: consecutive threads load consecutive elements
        for (int i = tx; i < NUMBER_PAR_PER_BOX; i += blockDim.x) {
            rA_shared[i] = rA[i];
        }

        // synchronize threads  - not needed, but just to be safe
        __syncthreads();

        //------------------------------------------------------------------------------------------------------------------------------------------------------160
        //	nei box loop
        //------------------------------------------------------------------------------------------------------------------------------------------------------160

        // loop over neighboring boxes of home box
        int nNei = 1 + d_box_gpu[bx].nn;
        for (k = 0; k < nNei; k++) {

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
            //	Load neighbor box into shared memory
            //----------------------------------------------------------------------------------------------------------------------------------140

            // coalesced loads for rB and qB
            for (int i = tx; i < NUMBER_PAR_PER_BOX; i += blockDim.x) {
                rB_shared[i] = rB[i];
                qB_shared[i] = qB[i];
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
                FOUR_VECTOR ri = rA_shared[i];

                // local accumulators in registers to reduce global memory traffic
                fp acc_v = 0.0f;
                fp acc_x = 0.0f;
                fp acc_y = 0.0f;
                fp acc_z = 0.0f;

                // loop for the number of particles in the current neighbor box
#pragma unroll 4
                for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {

                    FOUR_VECTOR rBj = rB_shared[j];
                    fp qBj = qB_shared[j];

                    // r2 = ri.v + rBj.v - DOT(ri, rBj);
                    fp dotp = ri.x * rBj.x + ri.y * rBj.y + ri.z * rBj.z;
                    r2 = ri.v + rBj.v - dotp;

                    u2 = a2 * r2;

                    // use fast single-precision exp for FP32
                    vij = __expf(-u2);
                    fs = (fp)2.0 * vij;

                    d.x = ri.x - rBj.x;
                    d.y = ri.y - rBj.y;
                    d.z = ri.z - rBj.z;

                    fxij = fs * d.x;
                    fyij = fs * d.y;
                    fzij = fs * d.z;

                    fp qv = qBj;
                    acc_v += qv * vij;
                    acc_x += qv * fxij;
                    acc_y += qv * fyij;
                    acc_z += qv * fzij;
                }

                // write back to global memory once per i, using double-precision accumulators as in original
                fA[i].v += (double)acc_v;
                fA[i].x += (double)acc_x;
                fA[i].y += (double)acc_y;
                fA[i].z += (double)acc_z;
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
