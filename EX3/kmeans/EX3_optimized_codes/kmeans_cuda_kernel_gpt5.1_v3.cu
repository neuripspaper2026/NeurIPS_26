#ifndef _KMEANS_CUDA_KERNEL_H_
#define _KMEANS_CUDA_KERNEL_H_

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "kmeans.h"

// FIXME: Make this a runtime selectable variable!
#define ASSUMED_NR_CLUSTERS 32

#define SDATA(index) CUT_BANK_CHECKER(sdata, index)

__constant__ float c_clusters[ASSUMED_NR_CLUSTERS *
                              34]; /* constant memory for cluster centers */

/* ----------------- invert_mapping() --------------------- */
/* inverts data array from row-major to column-major.

   [p0,dim0][p0,dim1][p0,dim2] ...
   [p1,dim0][p1,dim1][p1,dim2] ...
   [p2,dim0][p2,dim1][p2,dim2] ...
                                                                                to
   [dim0,p0][dim0,p1][dim0,p2] ...
   [dim1,p0][dim1,p1][dim1,p2] ...
   [dim2,p0][dim2,p1][dim2,p2] ...
*/
__global__ void invert_mapping(float *input,  /* original */
                               float *output, /* inverted */
                               int npoints,   /* npoints */
                               int nfeatures) /* nfeatures */
{
    // Use a 2D grid-stride loop for better scalability and occupancy
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_y = blockDim.y * gridDim.y;

    for (int p = tid_x; p < npoints; p += stride_x) {
        int base_in = p * nfeatures;
        for (int f = tid_y; f < nfeatures; f += stride_y) {
            output[p + npoints * f] = input[base_in + f];
        }
    }
}
/* ----------------- invert_mapping() end --------------------- */

/* to turn on the GPU delta and center reduction */
//#define GPU_DELTA_REDUCTION
//#define GPU_NEW_CENTER_REDUCTION


/* ----------------- kmeansPoint() --------------------- */
/* find the index of nearest cluster centers and change membership*/
__global__ void kmeansPoint(float *features, /* in: [npoints*nfeatures] */
                            int nfeatures, int npoints, int nclusters,
                            int *membership, float *clusters,
                            float *block_clusters, int *block_deltas,
                            cudaTextureObject_t texObj_features,
                            cudaTextureObject_t texObj_features_flipped) {

    // block ID
    const unsigned int block_id = gridDim.x * blockIdx.y + blockIdx.x;
    // point/thread ID
    const unsigned int point_id =
        block_id * blockDim.x * blockDim.y + threadIdx.x;

    int index = -1;

    if (point_id < npoints) {
        int i, j;
        float min_dist = FLT_MAX;

        // Loop over all clusters
        for (i = 0; i < nclusters; i++) {
            const int cluster_base_index = i * nfeatures;
            float ans = 0.0f;

            // Unroll the inner loop for better ILP and use FMA
#pragma unroll 4
            for (j = 0; j < nfeatures; j++) {
                const int addr = point_id + j * npoints;
                const float t = tex1Dfetch<float>(texObj_features, addr);
                const float c = c_clusters[cluster_base_index + j];
                const float diff = t - c;
                ans = fmaf(diff, diff, ans);
            }

            const float dist = ans;

            if (dist < min_dist) {
                min_dist = dist;
                index = i;
            }
        }
    }


#ifdef GPU_DELTA_REDUCTION
    // count how many points are now closer to a different cluster center
    __shared__ int deltas[THREADS_PER_BLOCK];
    if (threadIdx.x < THREADS_PER_BLOCK) {
        deltas[threadIdx.x] = 0;
    }
#endif
    if (point_id < npoints) {
#ifdef GPU_DELTA_REDUCTION
        /* if membership changes, increase delta by 1 */
        if (membership[point_id] != index) {
            deltas[threadIdx.x] = 1;
        }
#endif
        /* assign the membership to object point_id */
        membership[point_id] = index;
    }

#ifdef GPU_DELTA_REDUCTION
    // make sure all the deltas have finished writing to shared memory
    __syncthreads();

    // Warp-level reduction followed by block-level reduction

    // Step 1: warp-level sum using shuffle
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    int local = deltas[threadIdx.x];

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_down_sync(0xffffffff, local, offset);
    }

    // Step 2: write each warp's sum to shared memory
    __shared__ int warp_sums[THREADS_PER_BLOCK / 32];
    if (lane == 0) {
        warp_sums[warpId] = local;
    }
    __syncthreads();

    // Step 3: first warp reduces warp_sums
    if (warpId == 0) {
        int val = (lane < (THREADS_PER_BLOCK / 32)) ? warp_sums[lane] : 0;

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (lane == 0) {
            block_deltas[blockIdx.y * gridDim.x + blockIdx.x] = val;
        }
    }

#endif


#ifdef GPU_NEW_CENTER_REDUCTION
    int center_id = threadIdx.x / nfeatures;
    int dim_id = threadIdx.x - nfeatures * center_id;

    __shared__ int new_center_ids[THREADS_PER_BLOCK];

    new_center_ids[threadIdx.x] = index;
    __syncthreads();

    /***
    determine which dimension calculte the sum for
    mapping of threads is
    center0[dim0,dim1,dim2,...]center1[dim0,dim1,dim2,...]...
    ***/

    int new_base_index = (point_id - threadIdx.x) * nfeatures + dim_id;
    float accumulator = 0.f;

    if (threadIdx.x < nfeatures * nclusters) {
        // accumulate over all the elements of this threadblock
        for (int i = 0; i < (THREADS_PER_BLOCK); i++) {
            float val =
                tex1Dfetch<float>(texObj_features_flipped, new_base_index + i * nfeatures);
            if (new_center_ids[i] == center_id)
                accumulator += val;
        }

        // now store the sum for this threadblock
        /***
        mapping to global array is
        block0[center0[dim0,dim1,dim2,...]center1[dim0,dim1,dim2,...]...]block1[...]...
        ***/
        block_clusters[(blockIdx.y * gridDim.x + blockIdx.x) * nclusters *
                           nfeatures +
                       threadIdx.x] = accumulator;
    }
#endif
}
#endif // #ifndef _KMEANS_CUDA_KERNEL_H_
