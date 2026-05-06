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
        // Vectorized loads using float4 when possible
        int feature_base = point_id * nfeatures;
        int vec_features = (nfeatures / 4) * 4;
        
        // Process 4 features at a time
        for (int i = 0; i < vec_features; i += 4) {
            float4 val = *reinterpret_cast<float4*>(&input[feature_base + i]);
            output[point_id + npoints * i] = val.x;
            output[point_id + npoints * (i + 1)] = val.y;
            output[point_id + npoints * (i + 2)] = val.z;
            output[point_id + npoints * (i + 3)] = val.w;
        }
        
        // Handle remaining features
        for (int i = vec_features; i < nfeatures; i++) {
            output[point_id + npoints * i] = input[feature_base + i];
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
        // Shared memory for cluster centers (if not already in constant memory)
        // Pre-load features into registers for reuse across clusters
        float point_features[34]; // Max features based on constant memory size
        int actual_features = min(nfeatures, 34);
        
        // Coalesced loads from texture memory
        #pragma unroll 4
        for (int j = 0; j < actual_features; j++) {
            int addr = point_id + j * npoints;
            point_features[j] = tex1Dfetch<float>(texObj_features, addr);
        }

        /* find the cluster center id with min distance to pt */
        #pragma unroll 4
        for (int i = 0; i < nclusters; i++) {
            int cluster_base_index = i * nfeatures;
            float ans = 0.0f;

            // Unrolled inner loop for better instruction-level parallelism
            #pragma unroll 8
            for (int j = 0; j < actual_features; j++) {
                float diff = point_features[j] - c_clusters[cluster_base_index + j];
                ans = fmaf(diff, diff, ans); // Use FMA for A100
            }
            
            // Warp-level optimization: use __fminf for better performance
            float dist = ans;

            // Branchless minimum update using conditional move
            bool is_closer = dist < min_dist;
            min_dist = is_closer ? dist : min_dist;
            index = is_closer ? i : index;
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
    // Use cooperative groups for better warp synchronization
    if (threadIdx.x < 32) {
        // Warp-level reduction using shuffle operations
        int val = (threadIdx.x < THREADS_PER_BLOCK) ? deltas[threadIdx.x] : 0;
        if (threadIdx.x + 32 < THREADS_PER_BLOCK) {
            val += deltas[threadIdx.x + 32];
        }
        
        // Warp shuffle reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        
        if (threadIdx.x == 0) {
            // Accumulate remaining elements if THREADS_PER_BLOCK > 64
            int total = val;
            for (int i = 64; i < THREADS_PER_BLOCK; i++) {
                total += deltas[i];
            }
            block_deltas[blockIdx.y * gridDim.x + blockIdx.x] = total;
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
        // Unrolled loop for better performance
        #pragma unroll 8
        for (int i = 0; i < (THREADS_PER_BLOCK); i++) {
            float val =
                tex1Dfetch<float>(texObj_features_flipped, new_base_index + i * nfeatures);
            // Branchless accumulation using conditional
            accumulator += (new_center_ids[i] == center_id) ? val : 0.0f;
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
