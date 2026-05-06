#include "lavaMD.h"
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel_gpu_cuda(par_str d_par_gpu, dim_str d_dim_gpu,
                                box_str *d_box_gpu, FOUR_VECTOR *d_rv_gpu,
                                fp *d_qv_gpu, FOUR_VECTOR *d_fv_gpu) {

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
    //  THREAD PARAMETERS
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

    const int bx = blockIdx.x;   // get current horizontal block index (0-n)
    const int tx = threadIdx.x;  // get current horizontal thread index (0-n)
    int wtx = tx;

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
    //  DO FOR THE NUMBER OF BOXES
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

    if (bx < d_dim_gpu.number_boxes) {

        //------------------------------------------------------------------------------------------------------------------------------------------------------160
        //  Extract input parameters
        //------------------------------------------------------------------------------------------------------------------------------------------------------160

        // parameters
        const fp alpha  = d_par_gpu.alpha;
        const fp a2     = (fp)2.0f * alpha * alpha;

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
        while (wtx < NUMBER_PAR_PER_BOX) {
            rA_shared[wtx] = rA[wtx];
            wtx += NUMBER_THREADS;
        }
        wtx = tx;

        // synchronize threads  - not needed, but just to be safe
        __syncthreads();

        //------------------------------------------------------------------------------------------------------------------------------------------------------160
        //  nei box loop
        //------------------------------------------------------------------------------------------------------------------------------------------------------160

        // loop over neighboring boxes of home box
        const int nn = d_box_gpu[bx].nn;
        for (k = 0; k < (1 + nn); k++) {

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

            while (wtx < NUMBER_PAR_PER_BOX) {
                rB_shared[wtx] = rB[wtx];
                qB_shared[wtx] = qB[wtx];
                wtx += NUMBER_THREADS;
            }
            wtx = tx;

            // synchronize threads because in next section each thread accesses
            // data brought in by different threads here
            __syncthreads();

            //----------------------------------------------------------------------------------------------------------------------------------140
            //  Calculation
            //----------------------------------------------------------------------------------------------------------------------------------140

            // loop for the number of particles in the home box
            while (wtx < NUMBER_PAR_PER_BOX) {

                // keep local copy to registers
                FOUR_VECTOR rA_loc = rA_shared[wtx];
                FOUR_VECTOR fA_loc = fA[wtx];

                // unroll inner j-loop for better ILP and math throughput
                #pragma unroll 4
                for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {

                    const FOUR_VECTOR rB_loc = rB_shared[j];
                    const fp qB_val = qB_shared[j];

                    // r2 = rA[wtx].v + rB[j].v - DOT(rA[wtx],rB[j]);
                    r2 = rA_loc.v + rB_loc.v - DOT(rA_loc, rB_loc);
                    u2 = a2 * r2;
                    vij = __expf(-u2);
                    fs = (fp)2.0f * vij;

                    d.x = rA_loc.x - rB_loc.x;
                    fxij = fs * d.x;
                    d.y = rA_loc.y - rB_loc.y;
                    fyij = fs * d.y;
                    d.z = rA_loc.z - rB_loc.z;
                    fzij = fs * d.z;

                    const fp qv = qB_val;

                    fA_loc.v += (fp)(qv * vij);
                    fA_loc.x += (fp)(qv * fxij);
                    fA_loc.y += (fp)(qv * fyij);
                    fA_loc.z += (fp)(qv * fzij);
                }

                // write back accumulated forces to global memory
                fA[wtx] = fA_loc;

                // increment work thread index
                wtx += NUMBER_THREADS;
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
