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
    // 2D tiling for better memory coalescing on A100
    const int tile_x = blockDim.x;
    const int tile_y = blockDim.y;

    int x = blockIdx.x * tile_x + threadIdx.x; // point index
    int y = blockIdx.y * tile_y + threadIdx.y; // feature index

    if (x < npoints && y < nfeatures) {
        int in_idx  = x * nfeatures + y;
        int out_idx = x + npoints * y;
        output[out_idx] = input[in_idx];
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

        // Loop over all clusters
        for (int i = 0; i < nclusters; i++) {
            const int cluster_base_index = i * nfeatures;
            float ans = 0.0f;

            // Vectorized processing of features in chunks of 4 for higher ILP
            int j = 0;
            const int limit4 = nfeatures & ~3; // largest multiple of 4 <= nfeatures

#pragma unroll 4
            for (; j < limit4; j += 4) {
                const int addr0 = point_id + (j + 0) * npoints;
                const int addr1 = point_id + (j + 1) * npoints;
                const int addr2 = point_id + (j + 2) * npoints;
                const int addr3 = point_id + (j + 3) * npoints;

                const float f0 = tex1Dfetch<float>(texObj_features, addr0);
                const float f1 = tex1Dfetch<float>(texObj_features, addr1);
                const float f2 = tex1Dfetch<float>(texObj_features, addr2);
                const float f3 = tex1Dfetch<float>(texObj_features, addr3);

                const float c0 = c_clusters[cluster_base_index + j + 0];
                const float c1 = c_clusters[cluster_base_index + j + 1];
                const float c2 = c_clusters[cluster_base_index + j + 2];
                const float c3 = c_clusters[cluster_base_index + j + 3];

                const float d0 = f0 - c0;
                const float d1 = f1 - c1;
                const float d2 = f2 - c2;
                const float d3 = f3 - c3;

                ans = fmaf(d0, d0, ans);
                ans = fmaf(d1, d1, ans);
                ans = fmaf(d2, d2, ans);
                ans = fmaf(d3, d3, ans);
            }

            // Handle remaining features (if nfeatures not multiple of 4)
            for (; j < nfeatures; j++) {
                const int addr = point_id + j * npoints;
                const float f = tex1Dfetch<float>(texObj_features, addr);
                const float c = c_clusters[cluster_base_index + j];
                const float d = f - c;
                ans = fmaf(d, d, ans);
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

    // warp-level reduction using shuffle on A100
    int local_delta = deltas[threadIdx.x];

    // Intra-warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_delta += __shfl_down_sync(0xffffffff, local_delta, offset);
    }

    // Shared memory for per-warp sums
    __shared__ int warp_sums[THREADS_PER_BLOCK / warpSize];

    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        warp_sums[warp_id] = local_delta;
    }
    __syncthreads();

    // First warp accumulates the warp_sums
    int block_sum = 0;
    if (warp_id == 0) {
        int val = (threadIdx.x < (THREADS_PER_BLOCK / warpSize))
                      ? warp_sums[lane]
                      : 0;

        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (lane == 0) {
            block_sum = val;
            block_deltas[blockIdx.y * gridDim.x + blockIdx.x] = block_sum;
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
