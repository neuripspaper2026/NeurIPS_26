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
    // 2D thread block for better memory coalescing and higher occupancy
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_width = blockDim.x;
    int block_height = blockDim.y;

    int x = blockIdx.x * block_width + tx;  // feature index within tile
    int y = blockIdx.y * block_height + ty; // point index within tile

    if (x < nfeatures && y < npoints) {
        // Coalesced read/write: adjacent threads in x read/write adjacent elements
        int in_idx  = y * nfeatures + x;
        int out_idx = x * npoints  + y;
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
        // cache point features for this thread in registers to avoid repeated tex fetches
        // assuming nfeatures is reasonably small (<= 34 as per constant array comment)
        float feat_reg[34]; // upper bound; safe for typical kmeans.nfeatures
        int j;
        #pragma unroll
        for (j = 0; j < 34; ++j) {
            if (j < nfeatures) {
                int addr = point_id + j * npoints;
                feat_reg[j] = tex1Dfetch<float>(texObj_features, addr);
            }
        }

        int i;
        float min_dist = FLT_MAX;
        float dist; /* distance square between a point to cluster center */

        /* find the cluster center id with min distance to pt */
        for (i = 0; i < nclusters; i++) {
            int cluster_base_index = i * nfeatures; /* base index of cluster
                                                       centers for inverted
                                                       array */
            float ans = 0.0f; /* Euclidean distance square */

            #pragma unroll
            for (j = 0; j < 34; j++) {
                if (j < nfeatures) {
                    float diff = feat_reg[j] -
                                 c_clusters[cluster_base_index + j];
                    ans = fmaf(diff, diff, ans);
                }
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

    // warp-level reduction followed by block-level aggregation to reduce syncs
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    int val = deltas[threadIdx.x];

    // intra-warp reduction (assumes at most 1024 threads/block => 32 warps)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // write warp result to shared memory
    if (lane == 0) {
        deltas[warp] = val;
    }
    __syncthreads();

    // first warp reduces the warp-partial sums
    if (warp == 0) {
        val = (lane < (THREADS_PER_BLOCK + 31) / 32) ? deltas[lane] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            deltas[0] = val;
        }
    }
    __syncthreads();

    // propagate number of changes to global counter
    if (threadIdx.x == 0) {
        block_deltas[blockIdx.y * gridDim.x + blockIdx.x] = deltas[0];
        // printf("original id: %d, modified: %d\n",
        // blockIdx.y*gridDim.x+blockIdx.x, blockIdx.x);
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
