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
        // Vectorized loads for better memory throughput on A100
        int feature_id = 0;
        
        // Process 4 features at a time using float4 when possible
        for (; feature_id + 3 < nfeatures; feature_id += 4) {
            float4 vals = *reinterpret_cast<float4*>(&input[point_id * nfeatures + feature_id]);
            output[point_id + npoints * feature_id] = vals.x;
            output[point_id + npoints * (feature_id + 1)] = vals.y;
            output[point_id + npoints * (feature_id + 2)] = vals.z;
            output[point_id + npoints * (feature_id + 3)] = vals.w;
        }
        
        // Handle remaining features
        for (; feature_id < nfeatures; feature_id++) {
            output[point_id + npoints * feature_id] = input[point_id * nfeatures + feature_id];
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
        int i, j;
        float min_dist = FLT_MAX;
        float dist; /* distance square between a point to cluster center */

        /* find the cluster center id with min distance to pt */
        for (i = 0; i < nclusters; i++) {
            int cluster_base_index = i * nfeatures; /* base index of cluster
                                                       centers for inverted
                                                       array */
            float ans = 0.0; /* Euclidean distance sqaure */

            // Unroll inner loop for better instruction-level parallelism on A100
            j = 0;
            for (; j + 3 < nfeatures; j += 4) {
                int addr0 = point_id + j * npoints;
                int addr1 = point_id + (j + 1) * npoints;
                int addr2 = point_id + (j + 2) * npoints;
                int addr3 = point_id + (j + 3) * npoints;
                
                float val0 = tex1Dfetch<float>(texObj_features, addr0);
                float val1 = tex1Dfetch<float>(texObj_features, addr1);
                float val2 = tex1Dfetch<float>(texObj_features, addr2);
                float val3 = tex1Dfetch<float>(texObj_features, addr3);
                
                float diff0 = val0 - c_clusters[cluster_base_index + j];
                float diff1 = val1 - c_clusters[cluster_base_index + j + 1];
                float diff2 = val2 - c_clusters[cluster_base_index + j + 2];
                float diff3 = val3 - c_clusters[cluster_base_index + j + 3];
                
                // Use FMA (fused multiply-add) for better performance
                ans = __fmaf_rn(diff0, diff0, ans);
                ans = __fmaf_rn(diff1, diff1, ans);
                ans = __fmaf_rn(diff2, diff2, ans);
                ans = __fmaf_rn(diff3, diff3, ans);
            }
            
            // Handle remaining features
            for (; j < nfeatures; j++) {
                int addr = point_id + j * npoints;
                float diff = tex1Dfetch<float>(texObj_features, addr) -
                             c_clusters[cluster_base_index + j];
                ans = __fmaf_rn(diff, diff, ans);
            }
            
            dist = ans;

            /* see if distance is smaller than previous ones:
            if so, change minimum distance and save index of cluster center */
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

    // Optimized warp-level reduction for A100
    // Use warp shuffle instructions for better performance
    unsigned int lane = threadIdx.x & 31;
    unsigned int warp_id = threadIdx.x >> 5;
    
    // Warp-level reduction using shuffle
    int delta_val = deltas[threadIdx.x];
    for (int offset = 16; offset > 0; offset >>= 1) {
        delta_val += __shfl_down_sync(0xffffffff, delta_val, offset);
    }
    
    // Store warp results to shared memory
    __shared__ int warp_sums[32];
    if (lane == 0) {
        warp_sums[warp_id] = delta_val;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (threadIdx.x < 32) {
        int val = (threadIdx.x < ((THREADS_PER_BLOCK + 31) >> 5)) ? warp_sums[threadIdx.x] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
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
        // Unroll for better performance
        int i = 0;
        for (; i + 3 < THREADS_PER_BLOCK; i += 4) {
            float val0 = tex1Dfetch<float>(texObj_features_flipped, new_base_index + i * nfeatures);
            float val1 = tex1Dfetch<float>(texObj_features_flipped, new_base_index + (i + 1) * nfeatures);
            float val2 = tex1Dfetch<float>(texObj_features_flipped, new_base_index + (i + 2) * nfeatures);
            float val3 = tex1Dfetch<float>(texObj_features_flipped, new_base_index + (i + 3) * nfeatures);
            
            if (new_center_ids[i] == center_id) accumulator += val0;
            if (new_center_ids[i + 1] == center_id) accumulator += val1;
            if (new_center_ids[i + 2] == center_id) accumulator += val2;
            if (new_center_ids[i + 3] == center_id) accumulator += val3;
        }
        
        for (; i < THREADS_PER_BLOCK; i++) {
            float val = tex1Dfetch<float>(texObj_features_flipped, new_base_index + i * nfeatures);
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
