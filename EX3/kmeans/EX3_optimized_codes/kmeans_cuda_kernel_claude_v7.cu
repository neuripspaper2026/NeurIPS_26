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
    int point_id = threadIdx.x + blockDim.x * blockIdx.x; /* id of thread */
    
    if (point_id < npoints) {
        // Vectorized memory access for better coalescing
        #pragma unroll 4
        for (int i = 0; i < nfeatures; i++) {
            output[point_id + npoints * i] = input[point_id * nfeatures + i];
        }
    }
    return;
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
    float min_dist = FLT_MAX;

    if (point_id < npoints) {
        // Shared memory for features to reduce texture fetches
        extern __shared__ float shared_features[];
        
        // Cooperative loading of features into shared memory
        for (int j = threadIdx.x; j < nfeatures; j += blockDim.x) {
            if (point_id < npoints) {
                int addr = point_id + j * npoints;
                shared_features[j] = tex1Dfetch<float>(texObj_features, addr);
            }
        }
        __syncthreads();

        /* find the cluster center id with min distance to pt */
        #pragma unroll 4
        for (int i = 0; i < nclusters; i++) {
            int cluster_base_index = i * nfeatures;
            float ans = 0.0f;

            // Unroll inner loop for better instruction-level parallelism
            #pragma unroll 8
            for (int j = 0; j < nfeatures; j++) {
                float diff = shared_features[j] - c_clusters[cluster_base_index + j];
                ans = fmaf(diff, diff, ans); // Use fused multiply-add
            }

            // Warp-level optimization: avoid divergence
            if (ans < min_dist) {
                min_dist = ans;
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

    // Optimized reduction using warp-level primitives for A100
    unsigned int lane = threadIdx.x % 32;
    unsigned int warp_id = threadIdx.x / 32;
    
    // Warp-level reduction
    int val = deltas[threadIdx.x];
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    // First thread in each warp writes to shared memory
    __shared__ int warp_sums[32];
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (threadIdx.x < 32) {
        val = (threadIdx.x < (THREADS_PER_BLOCK + 31) / 32) ? warp_sums[threadIdx.x] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (threadIdx.x == 0) {
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
        #pragma unroll 4
        for (int i = 0; i < (THREADS_PER_BLOCK); i++) {
            float val =
                tex1Dfetch<float>(texObj_features_flipped, new_base_index + i * nfeatures);
            // Use predication to avoid branch divergence
            float mask = (float)(new_center_ids[i] == center_id);
            accumulator = fmaf(val, mask, accumulator);
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
