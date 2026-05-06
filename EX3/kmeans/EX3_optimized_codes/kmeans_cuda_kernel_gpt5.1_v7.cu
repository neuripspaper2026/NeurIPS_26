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
    int point_id = blockIdx.x * blockDim.x + threadIdx.x; /* id of thread */

    if (point_id < npoints) {
        // Use a register for base index to avoid repeated multiplication
        int in_base = point_id * nfeatures;
        // Unroll small feature loops to improve ILP where possible
        for (int i = 0; i < nfeatures; ++i) {
            output[point_id + npoints * i] = input[in_base + i];
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
        float min_dist = FLT_MAX;
        float dist; /* distance square between a point to cluster center */

        // Precompute stride to avoid repeated multiplication in inner loop
        const int stride_points = npoints;

        /* find the cluster center id with min distance to pt */
        for (int i = 0; i < nclusters; ++i) {
            int cluster_base_index = i * nfeatures; /* base index of cluster
                                                       centers for inverted
                                                       array */
            float ans = 0.0f; /* Euclidean distance square */

            // Use FMA-friendly pattern; compiler will generate FFMA on A100
            for (int j = 0; j < nfeatures; ++j) {
                int addr = point_id + j * stride_points;
                float fval = tex1Dfetch<float>(texObj_features, addr);
                float cval = c_clusters[cluster_base_index + j];
                float diff = fval - cval;
                ans = fmaf(diff, diff, ans);
            }
            dist = ans;

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

    // warp-level reduction followed by block-level finalization
    int local = deltas[threadIdx.x];

    // Intra-warp reduction using shuffle on A100
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_down_sync(mask, local, offset);
    }

    // Shared memory to collect per-warp partial sums
    __shared__ int warp_sums[32]; // supports up to 1024 threads/block

    if ((threadIdx.x & 31) == 0) {
        warp_sums[threadIdx.x >> 5] = local;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        int sum = (threadIdx.x < (THREADS_PER_BLOCK + 31) / 32)
                      ? warp_sums[threadIdx.x]
                      : 0;
        // Final reduction within first warp
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        if (threadIdx.x == 0) {
            block_deltas[blockIdx.y * gridDim.x + blockIdx.x] = sum;
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
        for (int i = 0; i < (THREADS_PER_BLOCK); ++i) {
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
