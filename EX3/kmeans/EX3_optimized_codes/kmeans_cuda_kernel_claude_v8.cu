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
        // Vectorized load/store using float4 when possible
        if (nfeatures >= 4 && (nfeatures % 4 == 0)) {
            for (int i = 0; i < nfeatures; i += 4) {
                float4 val = *reinterpret_cast<float4*>(&input[point_id * nfeatures + i]);
                output[point_id + npoints * i] = val.x;
                output[point_id + npoints * (i + 1)] = val.y;
                output[point_id + npoints * (i + 2)] = val.z;
                output[point_id + npoints * (i + 3)] = val.w;
            }
        } else {
            #pragma unroll 8
            for (int i = 0; i < nfeatures; i++)
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

    if (point_id < npoints) {
        float min_dist = FLT_MAX;

        /* find the cluster center id with min distance to pt */
        // Unroll outer loop for better instruction-level parallelism
        #pragma unroll 4
        for (int i = 0; i < nclusters; i++) {
            int cluster_base_index = i * nfeatures; /* base index of cluster
                                                       centers for inverted
                                                       array */
            float ans = 0.0f; /* Euclidean distance square */

            // Unroll inner loop and use register accumulation
            #pragma unroll 8
            for (int j = 0; j < nfeatures; j++) {
                int addr = point_id +
                           j * npoints; /* appropriate index of data point */
                float feature_val = tex1Dfetch<float>(texObj_features, addr);
                float cluster_val = c_clusters[cluster_base_index + j];
                float diff = feature_val - cluster_val;
                ans = __fmaf_rn(diff, diff, ans); /* fused multiply-add for better performance */
            }

            /* see if distance is smaller than previous ones:
            if so, change minimum distance and save index of cluster center */
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

    // Optimized warp-level reduction for A100
    // Use warp shuffle instructions for better performance
    int delta_val = deltas[threadIdx.x];
    
    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        delta_val += __shfl_down_sync(0xffffffff, delta_val, offset);
    }
    
    // First thread in each warp writes to shared memory
    if ((threadIdx.x & 31) == 0) {
        deltas[threadIdx.x / 32] = delta_val;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (threadIdx.x < 32) {
        int final_val = (threadIdx.x < (THREADS_PER_BLOCK + 31) / 32) ? deltas[threadIdx.x] : 0;
        for (int offset = 16; offset > 0; offset /= 2) {
            final_val += __shfl_down_sync(0xffffffff, final_val, offset);
        }
        if (threadIdx.x == 0) {
            block_deltas[blockIdx.y * gridDim.x + blockIdx.x] = final_val;
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
            // Avoid branch divergence using predication
            int match = (new_center_ids[i] == center_id);
            accumulator += val * match;
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
