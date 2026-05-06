#ifndef _KMEANS_CUDA_KERNEL_H_
#define _KMEANS_CUDA_KERNEL_H_

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "kmeans.h"

// FIXME: Make this a runtime selectable variable!
#define ASSUMED_NR_CLUSTERS 32

#define SDATA(index) CUT_BANK_CHECKER(sdata, index)

__constant__ __align__(16) float c_clusters[ASSUMED_NR_CLUSTERS *
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
        int base_in  = point_id * nfeatures;
        int base_out = point_id;
        // loop unrolling & restrict assumptions help compiler
        #pragma unroll 4
        for (int i = 0; i < nfeatures; i++) {
            output[base_out + (size_t)npoints * i] = input[base_in + i];
        }
    }
}
/* ----------------- invert_mapping() end --------------------- */

/* to turn on the GPU delta and center reduction */
//#define GPU_DELTA_REDUCTION
//#define GPU_NEW_CENTER_REDUCTION


/* ----------------- kmeansPoint() --------------------- */
/* find the index of nearest cluster centers and change membership*/
__global__ void kmeansPoint(float * __restrict__ features, /* in: [npoints*nfeatures] */
                            int nfeatures, int npoints, int nclusters,
                            int * __restrict__ membership, float * __restrict__ clusters,
                            float * __restrict__ block_clusters, int * __restrict__ block_deltas,
                            cudaTextureObject_t texObj_features,
                            cudaTextureObject_t texObj_features_flipped) {

    // block ID (flattened 2D grid)
    const unsigned int block_id = gridDim.x * blockIdx.y + blockIdx.x;
    // point/thread ID
    const unsigned int point_id =
        block_id * blockDim.x * blockDim.y + threadIdx.x;

    int index = -1;

    if (point_id < npoints) {
        float min_dist = FLT_MAX;

        // cache point features in registers to reduce repeated tex fetches
        // nfeatures is small in typical k-means usage; guard with max bound
        const int MAX_FEATS = 64;
        float feat_reg[MAX_FEATS];
        int nf = nfeatures < MAX_FEATS ? nfeatures : MAX_FEATS;

        #pragma unroll 4
        for (int j = 0; j < nf; ++j) {
            int addr = point_id + (size_t)j * npoints;
            feat_reg[j] = tex1Dfetch<float>(texObj_features, addr);
        }

        // main cluster-distance loop
        for (int i = 0; i < nclusters; i++) {
            int cluster_base_index = i * nfeatures; /* base index of cluster centers */
            float ans = 0.0f; /* Euclidean distance square */

            // unroll inner feature loop for better ILP/FMA usage
            int j = 0;
            for (; j + 3 < nf; j += 4) {
                float f0 = feat_reg[j + 0] - c_clusters[cluster_base_index + j + 0];
                float f1 = feat_reg[j + 1] - c_clusters[cluster_base_index + j + 1];
                float f2 = feat_reg[j + 2] - c_clusters[cluster_base_index + j + 2];
                float f3 = feat_reg[j + 3] - c_clusters[cluster_base_index + j + 3];
                ans = fmaf(f0, f0, ans);
                ans = fmaf(f1, f1, ans);
                ans = fmaf(f2, f2, ans);
                ans = fmaf(f3, f3, ans);
            }
            for (; j < nf; ++j) {
                float diff = feat_reg[j] - c_clusters[cluster_base_index + j];
                ans = fmaf(diff, diff, ans);
            }

            // if nfeatures > MAX_FEATS (rare), fall back to direct loads
            for (; j < nfeatures; ++j) {
                int addr = point_id + (size_t)j * npoints;
                float diff = (tex1Dfetch<float>(texObj_features, addr) -
                              c_clusters[cluster_base_index + j]);
                ans = fmaf(diff, diff, ans);
            }

            float dist = ans;

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

    // warp-level reduction for better efficiency on A100
    int val = deltas[threadIdx.x];

    // intra-warp reduction using shuffle
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // write per-warp partial sums to shared memory
    __shared__ int warp_sums[THREADS_PER_BLOCK / warpSize];
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // first warp reduces the per-warp sums
    if (warp_id == 0) {
        int warp_val = (threadIdx.x < (THREADS_PER_BLOCK / warpSize))
                           ? warp_sums[lane]
                           : 0;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            warp_val += __shfl_down_sync(0xffffffff, warp_val, offset);
        }
        if (lane == 0) {
            block_deltas[blockIdx.y * gridDim.x + blockIdx.x] = warp_val;
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
