#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void kernel(params_common_change *d_common_change,
                       params_common *d_common, params_unique *d_unique) {

    //======================================================================================================================================================
    //	COMMON VARIABLES
    //======================================================================================================================================================

    fp *d_in;
    int rot_row;
    int rot_col;
    int in2_rowlow;
    int in2_collow;
    int ic;
    int jc;
    int jp1;
    int ja1, ja2;
    int ip1;
    int ia1, ia2;
    int ja, jb;
    int ia, ib;
    float s;
    int i;
    int j;
    int row;
    int col;
    int ori_row;
    int ori_col;
    int position;
    float sum;
    int pos_ori;
    float temp;
    float temp2;
    int location;
    int cent;
    int tMask_row;
    int tMask_col;
    float largest_value_current = 0.0f;
    float largest_value = 0.0f;
    int largest_coordinate_current = 0;
    int largest_coordinate = 0;
    float fin_max_val = 0.0f;
    int fin_max_coo = 0;
    int largest_row;
    int largest_col;
    int offset_row;
    int offset_col;
    __shared__ float in_partial_sum[51];     // WATCH THIS !!! HARDCODED VALUE
    __shared__ float in_sqr_partial_sum[51]; // WATCH THIS !!! HARDCODED VALUE
    __shared__ float in_final_sum;
    __shared__ float in_sqr_final_sum;
    float mean;
    float mean_sqr;
    float variance;
    float deviation;
    __shared__ float denomT;
    __shared__ float par_max_val[131]; // WATCH THIS !!! HARDCODED VALUE
    __shared__ int par_max_coo[131];   // WATCH THIS !!! HARDCODED VALUE
    int pointer;
    __shared__ float d_in_mod_temp[2601];
    int ori_pointer;
    int loc_pointer;

    //======================================================================================================================================================
    //	THREAD PARAMETERS
    //======================================================================================================================================================

    const int bx = blockIdx.x;  // get current horizontal block index (0-n)
    const int tx = threadIdx.x; // get current horizontal thread index (0-n)
    const int warp_id = tx >> 5;
    const int lane_id = tx & 31;
    const unsigned full_mask = 0xffffffffu;
    int ei_new;

    //===============================================================================================================================================================================================================
    //===============================================================================================================================================================================================================
    //	GENERATE TEMPLATE
    //===============================================================================================================================================================================================================
    //===============================================================================================================================================================================================================

    // generate templates based on the first frame only
    if (d_common_change->frame_no == 0) {

        //======================================================================================================================================================
        // GET POINTER TO TEMPLATE FOR THE POINT
        //======================================================================================================================================================

        // pointers to: current template for current point
        d_in = &d_unique[bx].d_T[d_unique[bx].in_pointer];

        //======================================================================================================================================================
        //	UPDATE ROW LOC AND COL LOC
        //======================================================================================================================================================

        // uptade temporary endo/epi row/col coordinates (in each block
        // corresponding to point, narrow work to one thread)
        ei_new = tx;
        if (ei_new == 0) {

            // update temporary row/col coordinates
            pointer = d_unique[bx].point_no * d_common->no_frames +
                      d_common_change->frame_no;
            d_unique[bx].d_tRowLoc[pointer] =
                d_unique[bx].d_Row[d_unique[bx].point_no];
            d_unique[bx].d_tColLoc[pointer] =
                d_unique[bx].d_Col[d_unique[bx].point_no];
        }

        //======================================================================================================================================================
        //	CREATE TEMPLATES
        //======================================================================================================================================================

        // work
        ei_new = tx;
        const int in_rows = d_common->in_rows;
        const int frame_rows = d_common->frame_rows;
        const int in_elem = d_common->in_elem;
        while (ei_new < in_elem) {

            int idx1 = ei_new + 1;
            row = idx1 % in_rows - 1;           // (0-n) row
            col = idx1 / in_rows;               // (0-n) column
            if (idx1 % in_rows == 0) {
                row = in_rows - 1;
                col = col - 1;
            }

            ori_row =
                d_unique[bx].d_Row[d_unique[bx].point_no] - 25 + row - 1;
            ori_col =
                d_unique[bx].d_Col[d_unique[bx].point_no] - 25 + col - 1;
            ori_pointer = ori_col * frame_rows + ori_row;

            d_in[col * in_rows + row] = d_common_change->d_frame[ori_pointer];

            ei_new += NUMBER_THREADS;
        }
    }

    //===============================================================================================================================================================================================================
    //===============================================================================================================================================================================================================
    //	PROCESS POINTS
    //===============================================================================================================================================================================================================
    //===============================================================================================================================================================================================================

    // process points in all frames except for the first one
    if (d_common_change->frame_no != 0) {

        //======================================================================================================================================================
        //	SELECTION
        //======================================================================================================================================================

        in2_rowlow = d_unique[bx].d_Row[d_unique[bx].point_no] -
                     d_common->sSize; // (1 to n+1)
        in2_collow =
            d_unique[bx].d_Col[d_unique[bx].point_no] - d_common->sSize;

        const int in2_rows = d_common->in2_rows;
        const int in2_cols = d_common->in2_cols;
        const int in2_elem = d_common->in2_elem;

        ei_new = tx;
        while (ei_new < in2_elem) {

            int idx1 = ei_new + 1;
            row = idx1 % in2_rows - 1;           // (0-n) row
            col = idx1 / in2_rows;               // (0-n) column
            if (idx1 % in2_rows == 0) {
                row = in2_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_rowlow - 1;
            ori_col = col + in2_collow - 1;
            d_unique[bx].d_in2[ei_new] =
                d_common_change
                    ->d_frame[ori_col * d_common->frame_rows + ori_row];

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        //======================================================================================================================================================
        //	CONVOLUTION
        //======================================================================================================================================================

        //====================================================================================================
        //	ROTATION
        //====================================================================================================

        d_in = &d_unique[bx].d_T[d_unique[bx].in_pointer];
        const int in_r = d_common->in_rows;

        ei_new = tx;
        while (ei_new < d_common->in_elem) {

            int idx1 = ei_new + 1;
            row = idx1 % in_r - 1;           // (0-n) row
            col = idx1 / in_r;               // (0-n) column
            if (idx1 % in_r == 0) {
                row = in_r - 1;
                col = col - 1;
            }

            rot_row = (in_r - 1) - row;
            rot_col = (in_r - 1) - col;
            d_in_mod_temp[ei_new] = d_in[rot_col * in_r + rot_row];

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        //====================================================================================================
        //	ACTUAL CONVOLUTION
        //====================================================================================================

        const int conv_elem = d_common->conv_elem;
        const int conv_rows = d_common->conv_rows;
        const int in_cols = d_common->in_cols;
        const int joffset = d_common->joffset;
        const int ioffset = d_common->ioffset;

        ei_new = tx;
        while (ei_new < conv_elem) {

            int idx1 = ei_new + 1;
            ic = idx1 % conv_rows;         // (1-n)
            jc = idx1 / conv_rows + 1;     // (1-n)
            if (idx1 % conv_rows == 0) {
                ic = conv_rows;
                jc = jc - 1;
            }

            j = jc + joffset;
            jp1 = j + 1;
            ja1 = (d_common->in2_cols < jp1) ? (jp1 - d_common->in2_cols) : 1;
            ja2 = (in_cols < j) ? in_cols : j;

            i = ic + ioffset;
            ip1 = i + 1;

            ia1 = (d_common->in2_rows < ip1) ? (ip1 - d_common->in2_rows) : 1;
            ia2 = (in_r < i) ? in_r : i;

            s = 0.0f;

#pragma unroll 2
            for (ja = ja1; ja <= ja2; ja++) {
                jb = jp1 - ja;
#pragma unroll 2
                for (ia = ia1; ia <= ia2; ia++) {
                    ib = ip1 - ia;
                    s += d_in_mod_temp[in_r * (ja - 1) + ia - 1] *
                         d_unique[bx]
                             .d_in2[d_common->in2_rows * (jb - 1) + ib - 1];
                }
            }

            d_unique[bx].d_conv[ei_new] = s;

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        //======================================================================================================================================================
        //	CUMULATIVE SUM
        //======================================================================================================================================================

        //====================================================================================================
        //	PAD ARRAY, VERTICAL CUMULATIVE SUM
        //====================================================================================================

        const int in2_pad_cumv_rows = d_common->in2_pad_cumv_rows;
        const int in2_pad_cumv_cols = d_common->in2_pad_cumv_cols;
        const int in2_pad_cumv_elem = d_common->in2_pad_cumv_elem;
        const int in2_pad_add_rows = d_common->in2_pad_add_rows;
        const int in2_pad_add_cols = d_common->in2_pad_add_cols;

        ei_new = tx;
        while (ei_new < in2_pad_cumv_elem) {

            int idx1 = ei_new + 1;
            row = idx1 % in2_pad_cumv_rows - 1; // (0-n) row
            col = idx1 / in2_pad_cumv_rows;     // (0-n) column
            if (idx1 % in2_pad_cumv_rows == 0) {
                row = in2_pad_cumv_rows - 1;
                col = col - 1;
            }

            if (row > (in2_pad_add_rows - 1) &&
                row < (in2_pad_add_rows + in2_rows) &&
                col > (in2_pad_add_cols - 1) &&
                col < (in2_pad_add_cols + in2_cols)) {
                ori_row = row - in2_pad_add_rows;
                ori_col = col - in2_pad_add_cols;
                d_unique[bx].d_in2_pad_cumv[ei_new] =
                    d_unique[bx].d_in2[ori_col * in2_rows + ori_row];
            } else {
                d_unique[bx].d_in2_pad_cumv[ei_new] = 0.0f;
            }

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        //==================================================
        //	VERTICAL CUMULATIVE SUM
        //==================================================

        ei_new = tx;
        while (ei_new < in2_pad_cumv_cols) {

            pos_ori = ei_new * in2_pad_cumv_rows;
            sum = 0.0f;

#pragma unroll 4
            for (position = pos_ori;
                 position < pos_ori + in2_pad_cumv_rows;
                 position++) {
                float v = d_unique[bx].d_in2_pad_cumv[position] + sum;
                d_unique[bx].d_in2_pad_cumv[position] = v;
                sum = v;
            }

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        //====================================================================================================
        //	SELECTION
        //====================================================================================================

        const int in2_pad_cumv_sel_elem = d_common->in2_pad_cumv_sel_elem;
        const int in2_pad_cumv_sel_rows = d_common->in2_pad_cumv_sel_rows;
        const int in2_pad_cumv_sel_rowlow =
            d_common->in2_pad_cumv_sel_rowlow;
        const int in2_pad_cumv_sel_collow =
            d_common->in2_pad_cumv_sel_collow;

        ei_new = tx;
        while (ei_new < in2_pad_cumv_sel_elem) {

            int idx1 = ei_new + 1;
            row = idx1 % in2_pad_cumv_sel_rows - 1; // (0-n) row
            col = idx1 / in2_pad_cumv_sel_rows;     // (0-n) column
            if (idx1 % in2_pad_cumv_sel_rows == 0) {
                row = in2_pad_cumv_sel_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_pad_cumv_sel_rowlow - 1;
            ori_col = col + in2_pad_cumv_sel_collow - 1;
            d_unique[bx].d_in2_pad_cumv_sel[ei_new] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * in2_pad_cumv_rows + ori_row];

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        //====================================================================================================
        //	SELECTION 2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
        //====================================================================================================

        const int in2_sub_cumh_elem = d_common->in2_sub_cumh_elem;
        const int in2_sub_cumh_rows = d_common->in2_sub_cumh_rows;
        const int in2_pad_cumv_sel2_rowlow =
            d_common->in2_pad_cumv_sel2_rowlow;
        const int in2_pad_cumv_sel2_collow =
            d_common->in2_pad_cumv_sel2_collow;

        ei_new = tx;
        while (ei_new < in2_sub_cumh_elem) {

            int idx1 = ei_new + 1;
            row = idx1 % in2_sub_cumh_rows - 1; // (0-n) row
            col = idx1 / in2_sub_cumh_rows;     // (0-n) column
            if (idx1 % in2_sub_cumh_rows == 0) {
                row = in2_sub_cumh_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_pad_cumv_sel2_rowlow - 1;
            ori_col = col + in2_pad_cumv_sel2_collow - 1;
            d_unique[bx].d_in2_sub_cumh[ei_new] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * in2_pad_cumv_rows + ori_row];

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        ei_new = tx;
        while (ei_new < in2_sub_cumh_elem) {

            d_unique[bx].d_in2_sub_cumh[ei_new] =
                d_unique[bx].d_in2_pad_cumv_sel[ei_new] -
                d_unique[bx].d_in2_sub_cumh[ei_new];

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        ei_new = tx;
        while (ei_new < in2_sub_cumh_rows) {

            pos_ori = ei_new;
            sum = 0.0f;

#pragma unroll 4
            for (position = pos_ori;
                 position < pos_ori + in2_sub_cumh_elem;
                 position += in2_sub_cumh_rows) {
                float v = d_unique[bx].d_in2_sub_cumh[position] + sum;
                d_unique[bx].d_in2_sub_cumh[position] = v;
                sum = v;
            }

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        const int in2_sub_cumh_sel_elem = d_common->in2_sub_cumh_sel_elem;
        const int in2_sub_cumh_sel_rows =
            d_common->in2_sub_cumh_sel_rows;
        const int in2_sub_cumh_sel_rowlow =
            d_common->in2_sub_cumh_sel_rowlow;
        const int in2_sub_cumh_sel_collow =
            d_common->in2_sub_cumh_sel_collow;

        ei_new = tx;
        while (ei_new < in2_sub_cumh_sel_elem) {

            int idx1 = ei_new + 1;
            row = idx1 % in2_sub_cumh_sel_rows - 1; // (0-n) row
            col = idx1 / in2_sub_cumh_sel_rows;     // (0-n) column
            if (idx1 % in2_sub_cumh_sel_rows == 0) {
                row = in2_sub_cumh_sel_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_sub_cumh_sel_rowlow - 1;
            ori_col = col + in2_sub_cumh_sel_collow - 1;
            d_unique[bx].d_in2_sub_cumh_sel[ei_new] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * in2_sub_cumh_rows + ori_row];

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        const int in2_sub2_elem = d_common->in2_sub2_elem;
        const int in2_sub2_rows = d_common->in2_sub2_rows;
        const int in2_sub_cumh_sel2_rowlow =
            d_common->in2_sub_cumh_sel2_rowlow;
        const int in2_sub_cumh_sel2_collow =
            d_common->in2_sub_cumh_sel2_collow;

        ei_new = tx;
        while (ei_new < in2_sub2_elem) {

            int idx1 = ei_new + 1;
            row = idx1 % in2_sub2_rows - 1; // (0-n) row
            col = idx1 / in2_sub2_rows;     // (0-n) column
            if (idx1 % in2_sub2_rows == 0) {
                row = in2_sub2_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_sub_cumh_sel2_rowlow - 1;
            ori_col = col + in2_sub_cumh_sel2_collow - 1;
            d_unique[bx].d_in2_sub2[ei_new] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * in2_sub_cumh_rows + ori_row];

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        ei_new = tx;
        while (ei_new < in2_sub2_elem) {

            d_unique[bx].d_in2_sub2[ei_new] =
                d_unique[bx].d_in2_sub_cumh_sel[ei_new] -
                d_unique[bx].d_in2_sub2[ei_new];

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        //======================================================================================================================================================
        //	CUMULATIVE SUM 2
        //======================================================================================================================================================

        //====================================================================================================
        //	MULTIPLICATION
        //====================================================================================================

        const int in2_sqr_elem = d_common->in2_sqr_elem;
        const int in2_sqr_rows = d_common->in2_sqr_rows;
        const int in2_sqr_cols = d_common->in2_sqr_cols;

        ei_new = tx;
        while (ei_new < in2_sqr_elem) {

            temp = d_unique[bx].d_in2[ei_new];
            d_unique[bx].d_in2_sqr[ei_new] = temp * temp;

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        //====================================================================================================
        //	PAD ARRAY, VERTICAL CUMULATIVE SUM
        //====================================================================================================

        ei_new = tx;
        while (ei_new < in2_pad_cumv_elem) {

            int idx1 = ei_new + 1;
            row = idx1 % in2_pad_cumv_rows - 1; // (0-n) row
            col = idx1 / in2_pad_cumv_rows;     // (0-n) column
            if (idx1 % in2_pad_cumv_rows == 0) {
                row = in2_pad_cumv_rows - 1;
                col = col - 1;
            }

            if (row > (in2_pad_add_rows - 1) &&
                row < (in2_pad_add_rows + in2_sqr_rows) &&
                col > (in2_pad_add_cols - 1) &&
                col < (in2_pad_add_cols + in2_sqr_cols)) {
                ori_row = row - in2_pad_add_rows;
                ori_col = col - in2_pad_add_cols;
                d_unique[bx].d_in2_pad_cumv[ei_new] =
                    d_unique[bx]
                        .d_in2_sqr[ori_col * in2_sqr_rows + ori_row];
            } else {
                d_unique[bx].d_in2_pad_cumv[ei_new] = 0.0f;
            }

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        ei_new = tx;
        while (ei_new < in2_pad_cumv_cols) {

            pos_ori = ei_new * in2_pad_cumv_rows;
            sum = 0.0f;

#pragma unroll 4
            for (position = pos_ori;
                 position < pos_ori + in2_pad_cumv_rows;
                 position++) {
                float v = d_unique[bx].d_in2_pad_cumv[position] + sum;
                d_unique[bx].d_in2_pad_cumv[position] = v;
                sum = v;
            }

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        ei_new = tx;
        while (ei_new < in2_pad_cumv_sel_elem) {

            int idx1 = ei_new + 1;
            row = idx1 % in2_pad_cumv_sel_rows - 1; // (0-n) row
            col = idx1 / in2_pad_cumv_sel_rows;     // (0-n) column
            if (idx1 % in2_pad_cumv_sel_rows == 0) {
                row = in2_pad_cumv_sel_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_pad_cumv_sel_rowlow - 1;
            ori_col = col + in2_pad_cumv_sel_collow - 1;
            d_unique[bx].d_in2_pad_cumv_sel[ei_new] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * in2_pad_cumv_rows + ori_row];

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        ei_new = tx;
        while (ei_new < in2_sub_cumh_elem) {

            int idx1 = ei_new + 1;
            row = idx1 % in2_sub_cumh_rows - 1; // (0-n) row
            col = idx1 / in2_sub_cumh_rows;     // (0-n) column
            if (idx1 % in2_sub_cumh_rows == 0) {
                row = in2_sub_cumh_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_pad_cumv_sel2_rowlow - 1;
            ori_col = col + in2_pad_cumv_sel2_collow - 1;
            d_unique[bx].d_in2_sub_cumh[ei_new] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * in2_pad_cumv_rows + ori_row];

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        ei_new = tx;
        while (ei_new < in2_sub_cumh_elem) {

            d_unique[bx].d_in2_sub_cumh[ei_new] =
                d_unique[bx].d_in2_pad_cumv_sel[ei_new] -
                d_unique[bx].d_in2_sub_cumh[ei_new];

            ei_new += NUMBER_THREADS;
        }

        ei_new = tx;
        while (ei_new < in2_sub_cumh_rows) {

            pos_ori = ei_new;
            sum = 0.0f;

#pragma unroll 4
            for (position = pos_ori;
                 position < pos_ori + in2_sub_cumh_elem;
                 position += in2_sub_cumh_rows) {
                float v = d_unique[bx].d_in2_sub_cumh[position] + sum;
                d_unique[bx].d_in2_sub_cumh[position] = v;
                sum = v;
            }

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        ei_new = tx;
        while (ei_new < in2_sub_cumh_sel_elem) {

            int idx1 = ei_new + 1;
            row = idx1 % in2_sub_cumh_sel_rows - 1; // (0-n) row
            col = idx1 / in2_sub_cumh_sel_rows;     // (0-n) column
            if (idx1 % in2_sub_cumh_sel_rows == 0) {
                row = in2_sub_cumh_sel_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_sub_cumh_sel_rowlow - 1;
            ori_col = col + in2_sub_cumh_sel_collow - 1;
            d_unique[bx].d_in2_sub_cumh_sel[ei_new] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * in2_sub_cumh_rows + ori_row];

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        ei_new = tx;
        while (ei_new < in2_sub2_elem) {

            int idx1 = ei_new + 1;
            row = idx1 % in2_sub2_rows - 1; // (0-n) row
            col = idx1 / in2_sub2_rows;     // (0-n) column
            if (idx1 % in2_sub2_rows == 0) {
                row = in2_sub2_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_sub_cumh_sel2_rowlow - 1;
            ori_col = col + in2_sub_cumh_sel2_collow - 1;
            d_unique[bx].d_in2_sqr_sub2[ei_new] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * in2_sub_cumh_rows + ori_row];

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        ei_new = tx;
        while (ei_new < in2_sub2_elem) {

            d_unique[bx].d_in2_sqr_sub2[ei_new] =
                d_unique[bx].d_in2_sub_cumh_sel[ei_new] -
                d_unique[bx].d_in2_sqr_sub2[ei_new];

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        //======================================================================================================================================================
        //	FINAL
        //======================================================================================================================================================

        //====================================================================================================
        //	DENOMINATOR A		SAVE RESULT IN CUMULATIVE SUM A2
        //====================================================================================================

        const float in_elem_f = float(d_common->in_elem);

        ei_new = tx;
        while (ei_new < in2_sub2_elem) {

            temp = d_unique[bx].d_in2_sub2[ei_new];
            temp2 = d_unique[bx].d_in2_sqr_sub2[ei_new] -
                    (temp * temp / in_elem_f);
            if (temp2 < 0.0f) {
                temp2 = 0.0f;
            }
            d_unique[bx].d_in2_sqr_sub2[ei_new] = sqrtf(temp2);

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        //====================================================================================================
        //	MULTIPLICATION
        //====================================================================================================

        const int in_sqr_elem = d_common->in_sqr_elem;
        const int in_sqr_rows = d_common->in_sqr_rows;
        const int in_sqr_cols = d_common->in_sqr_cols;

        ei_new = tx;
        while (ei_new < in_sqr_elem) {

            temp = d_in[ei_new];
            d_unique[bx].d_in_sqr[ei_new] = temp * temp;

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        //====================================================================================================
        //	IN SUM
        //====================================================================================================

        ei_new = tx;
        while (ei_new < in_cols) {

            sum = 0.0f;
#pragma unroll 4
            for (i = 0; i < in_r; i++) {
                sum += d_in[ei_new * in_r + i];
            }
            in_partial_sum[ei_new] = sum;

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        //====================================================================================================
        //	IN_SQR SUM
        //====================================================================================================

        ei_new = tx;
        while (ei_new < in_sqr_rows) {

            sum = 0.0f;
#pragma unroll 4
            for (i = 0; i < in_sqr_cols; i++) {
                sum += d_unique[bx].d_in_sqr[ei_new + in_sqr_rows * i];
            }
            in_sqr_partial_sum[ei_new] = sum;

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        //====================================================================================================
        //	FINAL SUMMATION (warp-level reduction instead of scalar loops)
        //====================================================================================================

        // Use two warps (0 and 1) to compute in_final_sum and in_sqr_final_sum
        float local_sum0 = 0.0f;
        float local_sum1 = 0.0f;

        if (warp_id == 0) {
            for (int idx = lane_id; idx < in_cols; idx += 32) {
                local_sum0 += in_partial_sum[idx];
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                local_sum0 += __shfl_down_sync(full_mask, local_sum0, offset);
            }
            if (lane_id == 0) {
                in_final_sum = local_sum0;
            }
        } else if (warp_id == 1) {
            for (int idx = lane_id; idx < in_sqr_cols; idx += 32) {
                local_sum1 += in_sqr_partial_sum[idx];
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                local_sum1 += __shfl_down_sync(full_mask, local_sum1, offset);
            }
            if (lane_id == 0) {
                in_sqr_final_sum = local_sum1;
            }
        }

        __syncthreads();

        //====================================================================================================
        //	DENOMINATOR T
        //====================================================================================================

        if (tx == 0) {

            mean = in_final_sum / in_elem_f;
            mean_sqr = mean * mean;
            variance = (in_sqr_final_sum / in_elem_f) - mean_sqr;
            if (variance < 0.0f) variance = 0.0f;
            deviation = sqrtf(variance);

            denomT = sqrtf(float(d_common->in_elem - 1)) * deviation;
        }

        __syncthreads();

        //====================================================================================================
        //	DENOMINATOR		SAVE RESULT IN CUMULATIVE SUM A2
        //====================================================================================================

        ei_new = tx;
        while (ei_new < in2_sub2_elem) {

            d_unique[bx].d_in2_sqr_sub2[ei_new] =
                d_unique[bx].d_in2_sqr_sub2[ei_new] * denomT;

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        //====================================================================================================
        //	NUMERATOR	SAVE RESULT IN CONVOLUTION
        //====================================================================================================

        ei_new = tx;
        while (ei_new < conv_elem) {

            d_unique[bx].d_conv[ei_new] =
                d_unique[bx].d_conv[ei_new] -
                d_unique[bx].d_in2_sub2[ei_new] * in_final_sum / in_elem_f;

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        //====================================================================================================
        //	CORRELATION	SAVE RESULT IN CUMULATIVE SUM A2
        //====================================================================================================

        ei_new = tx;
        while (ei_new < in2_sub2_elem) {

            d_unique[bx].d_in2_sqr_sub2[ei_new] =
                d_unique[bx].d_conv[ei_new] /
                d_unique[bx].d_in2_sqr_sub2[ei_new];

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        //======================================================================================================================================================
        //	TEMPLATE MASK CREATE
        //======================================================================================================================================================

        cent = d_common->sSize + d_common->tSize + 1;
        if (d_common_change->frame_no == 0) {
            tMask_row = cent + d_unique[bx].d_Row[d_unique[bx].point_no] -
                        d_unique[bx].d_Row[d_unique[bx].point_no] - 1;
            tMask_col = cent + d_unique[bx].d_Col[d_unique[bx].point_no] -
                        d_unique[bx].d_Col[d_unique[bx].point_no] - 1;
        } else {
            pointer = d_common_change->frame_no - 1 +
                      d_unique[bx].point_no * d_common->no_frames;
            tMask_row = cent + d_unique[bx].d_tRowLoc[pointer] -
                        d_unique[bx].d_Row[d_unique[bx].point_no] - 1;
            tMask_col = cent + d_unique[bx].d_tColLoc[pointer] -
                        d_unique[bx].d_Col[d_unique[bx].point_no] - 1;
        }

        const int tMask_elem = d_common->tMask_elem;
        const int tMask_rows = d_common->tMask_rows;

        ei_new = tx;
        while (ei_new < tMask_elem) {

            location = tMask_col * tMask_rows + tMask_row;
            d_unique[bx].d_tMask[ei_new] = (ei_new == location) ? 1 : 0;

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        //======================================================================================================================================================
        //	MASK CONVOLUTION
        //======================================================================================================================================================

        const int mask_conv_elem = d_common->mask_conv_elem;
        const int mask_conv_rows = d_common->mask_conv_rows;
        const int mask_conv_cols = d_common->mask_conv_cols;
        const int mask_conv_joffset = d_common->mask_conv_joffset;
        const int mask_conv_ioffset = d_common->mask_conv_ioffset;
        const int mask_rows = d_common->mask_rows;
        const int mask_cols = d_common->mask_cols;
        const int tMask_cols = d_common->tMask_cols;

        ei_new = tx;
        while (ei_new < mask_conv_elem) {

            int idx1 = ei_new + 1;
            ic = idx1 % mask_conv_rows;           // (1-n)
            jc = idx1 / mask_conv_rows + 1;       // (1-n)
            if (idx1 % mask_conv_rows == 0) {
                ic = mask_conv_rows;
                jc = jc - 1;
            }

            j = jc + mask_conv_joffset;
            jp1 = j + 1;
            ja1 = (mask_cols < jp1) ? (jp1 - mask_cols) : 1;
            ja2 = (tMask_cols < j) ? tMask_cols : j;

            i = ic + mask_conv_ioffset;
            ip1 = i + 1;

            ia1 = (mask_rows < ip1) ? (ip1 - mask_rows) : 1;
            ia2 = (tMask_rows < i) ? tMask_rows : i;

            s = 0.0f;

#pragma unroll 2
            for (ja = ja1; ja <= ja2; ja++) {
                jb = jp1 - ja;
#pragma unroll 2
                for (ia = ia1; ia <= ia2; ia++) {
                    ib = ip1 - ia;
                    s += d_unique[bx]
                             .d_tMask[tMask_rows * (ja - 1) + ia - 1];
                }
            }

            d_unique[bx].d_mask_conv[ei_new] =
                d_unique[bx].d_in2_sqr_sub2[ei_new] * s;

            ei_new += NUMBER_THREADS;
        }

        __syncthreads();

        //======================================================================================================================================================
        //	MAXIMUM VALUE
        //======================================================================================================================================================

        //====================================================================================================
        //	INITIAL SEARCH (parallelized across threads)
        //====================================================================================================

        largest_value = 0.0f;
        largest_coordinate = 0;

        const int total_mask_elems = mask_conv_rows * mask_conv_cols;

        ei_new = tx;
        while (ei_new < total_mask_elems) {
            largest_coordinate_current = ei_new;
            largest_value_current =
                fabsf(d_unique[bx].d_mask_conv[largest_coordinate_current]);
            if (largest_value_current > largest_value) {
                largest_value = largest_value_current;
                largest_coordinate = largest_coordinate_current;
            }
            ei_new += NUMBER_THREADS;
        }

        par_max_coo[tx] = largest_coordinate;
        par_max_val[tx] = largest_value;

        __syncthreads();

        // warp-level reduction among threads in the block
        float val = par_max_val[tx];
        int coo = par_max_coo[tx];

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float val_other = __shfl_down_sync(full_mask, val, offset);
            int coo_other = __shfl_down_sync(full_mask, coo, offset);
            if (val_other > val) {
                val = val_other;
                coo = coo_other;
            }
        }

        if (lane_id == 0) {
            par_max_val[warp_id] = val;
            par_max_coo[warp_id] = coo;
        }

        __syncthreads();

        if (tx == 0) {
            fin_max_val = par_max_val[0];
            fin_max_coo = par_max_coo[0];
            int max_warps = (NUMBER_THREADS + 31) / 32;
            if (max_warps > 131) max_warps = 131;
            for (i = 1; i < max_warps; i++) {
                if (par_max_val[i] > fin_max_val) {
                    fin_max_val = par_max_val[i];
                    fin_max_coo = par_max_coo[i];
                }
            }

            largest_row =
                (fin_max_coo + 1) % mask_conv_rows - 1; // (0-n) row
            largest_col =
                (fin_max_coo + 1) / mask_conv_rows; // (0-n) column
            if ((fin_max_coo + 1) % mask_conv_rows == 0) {
                largest_row = mask_conv_rows - 1;
                largest_col = largest_col - 1;
            }

            largest_row = largest_row + 1;
            largest_col = largest_col + 1;
            offset_row = largest_row - in_r -
                         (d_common->sSize - d_common->tSize);
            offset_col = largest_col - in_cols -
                         (d_common->sSize - d_common->tSize);
            pointer = d_common_change->frame_no +
                      d_unique[bx].point_no * d_common->no_frames;
            d_unique[bx].d_tRowLoc[pointer] =
                d_unique[bx].d_Row[d_unique[bx].point_no] + offset_row;
            d_unique[bx].d_tColLoc[pointer] =
                d_unique[bx].d_Col[d_unique[bx].point_no] + offset_col;
        }

        __syncthreads();
    }

    //===============================================================================================================================================================================================================
    //===============================================================================================================================================================================================================
    //	COORDINATE AND TEMPLATE UPDATE
    //===============================================================================================================================================================================================================
    //===============================================================================================================================================================================================================

    if (d_common_change->frame_no != 0 &&
        (d_common_change->frame_no) % 10 == 0) {

        loc_pointer = d_unique[bx].point_no * d_common->no_frames +
                      d_common_change->frame_no;
        d_unique[bx].d_Row[d_unique[bx].point_no] =
            d_unique[bx].d_tRowLoc[loc_pointer];
        d_unique[bx].d_Col[d_unique[bx].point_no] =
            d_unique[bx].d_tColLoc[loc_pointer];

        d_in = &d_unique[bx].d_T[d_unique[bx].in_pointer];

        const int in_rows2 = d_common->in_rows;
        const int frame_rows2 = d_common->frame_rows;
        const int in_elem2 = d_common->in_elem;
        const float alpha = d_common->alpha;

        ei_new = tx;
        while (ei_new < in_elem2) {

            int idx1 = ei_new + 1;
            row = idx1 % in_rows2 - 1;       // (0-n) row
            col = idx1 / in_rows2;           // (0-n) column
            if (idx1 % in_rows2 == 0) {
                row = in_rows2 - 1;
                col = col - 1;
            }

            ori_row =
                d_unique[bx].d_Row[d_unique[bx].point_no] - 25 + row - 1;
            ori_col =
                d_unique[bx].d_Col[d_unique[bx].point_no] - 25 + col - 1;
            ori_pointer = ori_col * frame_rows2 + ori_row;

            float prev = d_in[ei_new];
            float newv = d_common_change->d_frame[ori_pointer];
            d_in[ei_new] = alpha * prev + (1.0f - alpha) * newv;

            ei_new += NUMBER_THREADS;
        }
    }
}
