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
        for (ei_new = tx; ei_new < d_common->in_elem;
             ei_new += blockDim.x) {

            // figure out row/col location in new matrix
            int idxp1 = ei_new + 1;
            row = idxp1 % d_common->in_rows - 1;     // (0-n) row
            col = idxp1 / d_common->in_rows;         // (0-n) column
            if (idxp1 % d_common->in_rows == 0) {
                row = d_common->in_rows - 1;
                col = col - 1;
            }

            // figure out row/col location in corresponding new template area in
            // image and give to every thread (get top left corner and progress
            // down and right)
            ori_row = d_unique[bx].d_Row[d_unique[bx].point_no] - 25 + row - 1;
            ori_col = d_unique[bx].d_Col[d_unique[bx].point_no] - 25 + col - 1;
            ori_pointer = __mul24(ori_col, d_common->frame_rows) + ori_row;

            // update template
            d_in[col * d_common->in_rows + row] =
                d_common_change->d_frame[ori_pointer];
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

        // work
        for (ei_new = tx; ei_new < d_common->in2_elem;
             ei_new += blockDim.x) {

            // figure out row/col location in new matrix
            int idxp1 = ei_new + 1;
            row = idxp1 % d_common->in2_rows - 1;     // (0-n) row
            col = idxp1 / d_common->in2_rows;         // (0-n) column
            if (idxp1 % d_common->in2_rows == 0) {
                row = d_common->in2_rows - 1;
                col = col - 1;
            }

            // figure out corresponding location in old matrix and copy values
            // to new matrix
            ori_row = row + in2_rowlow - 1;
            ori_col = col + in2_collow - 1;
            d_unique[bx].d_in2[ei_new] =
                d_common_change
                    ->d_frame[__mul24(ori_col, d_common->frame_rows) +
                              ori_row];
        }

        //======================================================================================================================================================
        //	SYNCHRONIZE THREADS
        //======================================================================================================================================================

        __syncthreads();

        //======================================================================================================================================================
        //	CONVOLUTION
        //======================================================================================================================================================

        //====================================================================================================
        //	ROTATION
        //====================================================================================================

        // variables
        d_in = &d_unique[bx].d_T[d_unique[bx].in_pointer];

        // work
        for (ei_new = tx; ei_new < d_common->in_elem;
             ei_new += blockDim.x) {

            // figure out row/col location in padded array
            int idxp1 = ei_new + 1;
            row = idxp1 % d_common->in_rows - 1;     // (0-n) row
            col = idxp1 / d_common->in_rows;         // (0-n) column
            if (idxp1 % d_common->in_rows == 0) {
                row = d_common->in_rows - 1;
                col = col - 1;
            }

            // execution
            rot_row = (d_common->in_rows - 1) - row;
            rot_col = (d_common->in_rows - 1) - col;
            d_in_mod_temp[ei_new] =
                d_in[rot_col * d_common->in_rows + rot_row];
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	ACTUAL CONVOLUTION
        //====================================================================================================

        for (ei_new = tx; ei_new < d_common->conv_elem;
             ei_new += blockDim.x) {

            // figure out row/col location in array
            int idxp1 = ei_new + 1;
            ic = idxp1 % d_common->conv_rows;     // (1-n)
            jc = idxp1 / d_common->conv_rows + 1; // (1-n)
            if (idxp1 % d_common->conv_rows == 0) {
                ic = d_common->conv_rows;
                jc = jc - 1;
            }

            j = jc + d_common->joffset;
            jp1 = j + 1;
            if (d_common->in2_cols < jp1) {
                ja1 = jp1 - d_common->in2_cols;
            } else {
                ja1 = 1;
            }
            if (d_common->in_cols < j) {
                ja2 = d_common->in_cols;
            } else {
                ja2 = j;
            }

            i = ic + d_common->ioffset;
            ip1 = i + 1;

            if (d_common->in2_rows < ip1) {
                ia1 = ip1 - d_common->in2_rows;
            } else {
                ia1 = 1;
            }
            if (d_common->in_rows < i) {
                ia2 = d_common->in_rows;
            } else {
                ia2 = i;
            }

            s = 0.0f;

            for (ja = ja1; ja <= ja2; ja++) {
                jb = jp1 - ja;
#pragma unroll 4
                for (ia = ia1; ia <= ia2; ia++) {
                    ib = ip1 - ia;
                    s = fmaf(
                        d_in_mod_temp[d_common->in_rows * (ja - 1) + ia - 1],
                        d_unique[bx]
                            .d_in2[d_common->in2_rows * (jb - 1) + ib - 1],
                        s);
                }
            }

            d_unique[bx].d_conv[ei_new] = s;
        }

        //======================================================================================================================================================
        //	SYNCHRONIZE THREADS
        //======================================================================================================================================================

        __syncthreads();

        //======================================================================================================================================================
        //	CUMULATIVE SUM
        //======================================================================================================================================================

        //====================================================================================================
        //	PAD ARRAY, VERTICAL CUMULATIVE SUM
        //====================================================================================================

        //==================================================
        //	PADD ARRAY
        //==================================================

        for (ei_new = tx; ei_new < d_common->in2_pad_cumv_elem;
             ei_new += blockDim.x) {

            // figure out row/col location in padded array
            int idxp1 = ei_new + 1;
            row = idxp1 % d_common->in2_pad_cumv_rows - 1; // (0-n) row
            col = idxp1 / d_common->in2_pad_cumv_rows;     // (0-n) column
            if (idxp1 % d_common->in2_pad_cumv_rows == 0) {
                row = d_common->in2_pad_cumv_rows - 1;
                col = col - 1;
            }

            // execution
            if (row > (d_common->in2_pad_add_rows - 1) &&
                row < (d_common->in2_pad_add_rows + d_common->in2_rows) &&
                col > (d_common->in2_pad_add_cols - 1) &&
                col < (d_common->in2_pad_add_cols + d_common->in2_cols)) {
                ori_row = row - d_common->in2_pad_add_rows;
                ori_col = col - d_common->in2_pad_add_cols;
                d_unique[bx].d_in2_pad_cumv[ei_new] =
                    d_unique[bx].d_in2[ori_col * d_common->in2_rows + ori_row];
            } else {
                d_unique[bx].d_in2_pad_cumv[ei_new] = 0.0f;
            }
        }

        //==================================================
        //	SYNCHRONIZE THREADS
        //==================================================

        __syncthreads();

        //==================================================
        //	VERTICAL CUMULATIVE SUM
        //==================================================

        for (ei_new = tx; ei_new < d_common->in2_pad_cumv_cols;
             ei_new += blockDim.x) {

            // figure out column position
            pos_ori = ei_new * d_common->in2_pad_cumv_rows;

            sum = 0.0f;

#pragma unroll 4
            for (position = pos_ori;
                 position < pos_ori + d_common->in2_pad_cumv_rows;
                 position++) {
                float v = d_unique[bx].d_in2_pad_cumv[position] + sum;
                d_unique[bx].d_in2_pad_cumv[position] = v;
                sum = v;
            }
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	SELECTION
        //====================================================================================================

        for (ei_new = tx; ei_new < d_common->in2_pad_cumv_sel_elem;
             ei_new += blockDim.x) {

            int idxp1 = ei_new + 1;
            row = idxp1 % d_common->in2_pad_cumv_sel_rows - 1; // (0-n) row
            col = idxp1 / d_common->in2_pad_cumv_sel_rows;     // (0-n) column
            if (idxp1 % d_common->in2_pad_cumv_sel_rows == 0) {
                row = d_common->in2_pad_cumv_sel_rows - 1;
                col = col - 1;
            }

            ori_row = row + d_common->in2_pad_cumv_sel_rowlow - 1;
            ori_col = col + d_common->in2_pad_cumv_sel_collow - 1;
            d_unique[bx].d_in2_pad_cumv_sel[ei_new] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * d_common->in2_pad_cumv_rows +
                                    ori_row];
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	SELECTION 2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
        //====================================================================================================

        //==================================================
        //	SELECTION 2
        //==================================================

        for (ei_new = tx; ei_new < d_common->in2_sub_cumh_elem;
             ei_new += blockDim.x) {

            int idxp1 = ei_new + 1;
            row = idxp1 % d_common->in2_sub_cumh_rows - 1; // (0-n) row
            col = idxp1 / d_common->in2_sub_cumh_rows;     // (0-n) column
            if (idxp1 % d_common->in2_sub_cumh_rows == 0) {
                row = d_common->in2_sub_cumh_rows - 1;
                col = col - 1;
            }

            ori_row = row + d_common->in2_pad_cumv_sel2_rowlow - 1;
            ori_col = col + d_common->in2_pad_cumv_sel2_collow - 1;
            d_unique[bx].d_in2_sub_cumh[ei_new] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * d_common->in2_pad_cumv_rows +
                                    ori_row];
        }

        //==================================================
        //	SYNCHRONIZE THREADS
        //==================================================

        __syncthreads();

        //==================================================
        //	SUBTRACTION
        //==================================================

        for (ei_new = tx; ei_new < d_common->in2_sub_cumh_elem;
             ei_new += blockDim.x) {

            d_unique[bx].d_in2_sub_cumh[ei_new] =
                d_unique[bx].d_in2_pad_cumv_sel[ei_new] -
                d_unique[bx].d_in2_sub_cumh[ei_new];
        }

        //==================================================
        //	SYNCHRONIZE THREADS
        //==================================================

        __syncthreads();

        //==================================================
        //	HORIZONTAL CUMULATIVE SUM
        //==================================================

        for (ei_new = tx; ei_new < d_common->in2_sub_cumh_rows;
             ei_new += blockDim.x) {

            pos_ori = ei_new;

            sum = 0.0f;

#pragma unroll 4
            for (position = pos_ori;
                 position < pos_ori + d_common->in2_sub_cumh_elem;
                 position += d_common->in2_sub_cumh_rows) {
                float v = d_unique[bx].d_in2_sub_cumh[position] + sum;
                d_unique[bx].d_in2_sub_cumh[position] = v;
                sum = v;
            }
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	SELECTION
        //====================================================================================================

        for (ei_new = tx; ei_new < d_common->in2_sub_cumh_sel_elem;
             ei_new += blockDim.x) {

            int idxp1 = ei_new + 1;
            row = idxp1 % d_common->in2_sub_cumh_sel_rows - 1; // (0-n) row
            col = idxp1 / d_common->in2_sub_cumh_sel_rows;     // (0-n) column
            if (idxp1 % d_common->in2_sub_cumh_sel_rows == 0) {
                row = d_common->in2_sub_cumh_sel_rows - 1;
                col = col - 1;
            }

            ori_row = row + d_common->in2_sub_cumh_sel_rowlow - 1;
            ori_col = col + d_common->in2_sub_cumh_sel_collow - 1;
            d_unique[bx].d_in2_sub_cumh_sel[ei_new] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * d_common->in2_sub_cumh_rows +
                                    ori_row];
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	SELECTION 2, SUBTRACTION
        //====================================================================================================

        //==================================================
        //	SELECTION 2
        //==================================================

        for (ei_new = tx; ei_new < d_common->in2_sub2_elem;
             ei_new += blockDim.x) {

            int idxp1 = ei_new + 1;
            row = idxp1 % d_common->in2_sub2_rows - 1; // (0-n) row
            col = idxp1 / d_common->in2_sub2_rows;     // (0-n) column
            if (idxp1 % d_common->in2_sub2_rows == 0) {
                row = d_common->in2_sub2_rows - 1;
                col = col - 1;
            }

            ori_row = row + d_common->in2_sub_cumh_sel2_rowlow - 1;
            ori_col = col + d_common->in2_sub_cumh_sel2_collow - 1;
            d_unique[bx].d_in2_sub2[ei_new] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * d_common->in2_sub_cumh_rows +
                                    ori_row];
        }

        //==================================================
        //	SYNCHRONIZE THREADS
        //==================================================

        __syncthreads();

        //==================================================
        //	SUBTRACTION
        //==================================================

        for (ei_new = tx; ei_new < d_common->in2_sub2_elem;
             ei_new += blockDim.x) {

            d_unique[bx].d_in2_sub2[ei_new] =
                d_unique[bx].d_in2_sub_cumh_sel[ei_new] -
                d_unique[bx].d_in2_sub2[ei_new];
        }

        //======================================================================================================================================================
        //	SYNCHRONIZE THREADS
        //======================================================================================================================================================

        __syncthreads();

        //======================================================================================================================================================
        //	CUMULATIVE SUM 2
        //======================================================================================================================================================

        //====================================================================================================
        //	MULTIPLICATION
        //====================================================================================================

        for (ei_new = tx; ei_new < d_common->in2_sqr_elem;
             ei_new += blockDim.x) {

            temp = d_unique[bx].d_in2[ei_new];
            d_unique[bx].d_in2_sqr[ei_new] = temp * temp;
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	PAD ARRAY, VERTICAL CUMULATIVE SUM
        //====================================================================================================

        //==================================================
        //	PAD ARRAY
        //==================================================

        for (ei_new = tx; ei_new < d_common->in2_pad_cumv_elem;
             ei_new += blockDim.x) {

            int idxp1 = ei_new + 1;
            row = idxp1 % d_common->in2_pad_cumv_rows - 1; // (0-n) row
            col = idxp1 / d_common->in2_pad_cumv_rows;     // (0-n) column
            if (idxp1 % d_common->in2_pad_cumv_rows == 0) {
                row = d_common->in2_pad_cumv_rows - 1;
                col = col - 1;
            }

            if (row > (d_common->in2_pad_add_rows - 1) &&
                row <
                    (d_common->in2_pad_add_rows + d_common->in2_sqr_rows) &&
                col > (d_common->in2_pad_add_cols - 1) &&
                col < (d_common->in2_pad_add_cols + d_common->in2_sqr_cols)) {
                ori_row = row - d_common->in2_pad_add_rows;
                ori_col = col - d_common->in2_pad_add_cols;
                d_unique[bx].d_in2_pad_cumv[ei_new] =
                    d_unique[bx].d_in2_sqr[ori_col * d_common->in2_sqr_rows +
                                           ori_row];
            } else {
                d_unique[bx].d_in2_pad_cumv[ei_new] = 0.0f;
            }
        }

        //==================================================
        //	SYNCHRONIZE THREADS
        //==================================================

        __syncthreads();

        //==================================================
        //	VERTICAL CUMULATIVE SUM
        //==================================================

        for (ei_new = tx; ei_new < d_common->in2_pad_cumv_cols;
             ei_new += blockDim.x) {

            pos_ori = ei_new * d_common->in2_pad_cumv_rows;

            sum = 0.0f;

#pragma unroll 4
            for (position = pos_ori;
                 position < pos_ori + d_common->in2_pad_cumv_rows;
                 position++) {
                float v = d_unique[bx].d_in2_pad_cumv[position] + sum;
                d_unique[bx].d_in2_pad_cumv[position] = v;
                sum = v;
            }
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	SELECTION
        //====================================================================================================

        for (ei_new = tx; ei_new < d_common->in2_pad_cumv_sel_elem;
             ei_new += blockDim.x) {

            int idxp1 = ei_new + 1;
            row = idxp1 % d_common->in2_pad_cumv_sel_rows - 1; // (0-n) row
            col = idxp1 / d_common->in2_pad_cumv_sel_rows;     // (0-n) column
            if (idxp1 % d_common->in2_pad_cumv_sel_rows == 0) {
                row = d_common->in2_pad_cumv_sel_rows - 1;
                col = col - 1;
            }

            ori_row = row + d_common->in2_pad_cumv_sel_rowlow - 1;
            ori_col = col + d_common->in2_pad_cumv_sel_collow - 1;
            d_unique[bx].d_in2_pad_cumv_sel[ei_new] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * d_common->in2_pad_cumv_rows +
                                    ori_row];
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	SELECTION 2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
        //====================================================================================================

        //==================================================
        //	SELECTION 2
        //==================================================

        for (ei_new = tx; ei_new < d_common->in2_sub_cumh_elem;
             ei_new += blockDim.x) {

            int idxp1 = ei_new + 1;
            row = idxp1 % d_common->in2_sub_cumh_rows - 1; // (0-n) row
            col = idxp1 / d_common->in2_sub_cumh_rows;     // (0-n) column
            if (idxp1 % d_common->in2_sub_cumh_rows == 0) {
                row = d_common->in2_sub_cumh_rows - 1;
                col = col - 1;
            }

            ori_row = row + d_common->in2_pad_cumv_sel2_rowlow - 1;
            ori_col = col + d_common->in2_pad_cumv_sel2_collow - 1;
            d_unique[bx].d_in2_sub_cumh[ei_new] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * d_common->in2_pad_cumv_rows +
                                    ori_row];
        }

        //==================================================
        //	SYNCHRONIZE THREADS
        //==================================================

        __syncthreads();

        //==================================================
        //	SUBTRACTION
        //==================================================

        for (ei_new = tx; ei_new < d_common->in2_sub_cumh_elem;
             ei_new += blockDim.x) {

            d_unique[bx].d_in2_sub_cumh[ei_new] =
                d_unique[bx].d_in2_pad_cumv_sel[ei_new] -
                d_unique[bx].d_in2_sub_cumh[ei_new];
        }

        //==================================================
        //	HORIZONTAL CUMULATIVE SUM
        //==================================================

        for (ei_new = tx; ei_new < d_common->in2_sub_cumh_rows;
             ei_new += blockDim.x) {

            pos_ori = ei_new;

            sum = 0.0f;

#pragma unroll 4
            for (position = pos_ori;
                 position < pos_ori + d_common->in2_sub_cumh_elem;
                 position += d_common->in2_sub_cumh_rows) {
                float v = d_unique[bx].d_in2_sub_cumh[position] + sum;
                d_unique[bx].d_in2_sub_cumh[position] = v;
                sum = v;
            }
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	SELECTION
        //====================================================================================================

        for (ei_new = tx; ei_new < d_common->in2_sub_cumh_sel_elem;
             ei_new += blockDim.x) {

            int idxp1 = ei_new + 1;
            row = idxp1 % d_common->in2_sub_cumh_sel_rows - 1; // (0-n) row
            col = idxp1 / d_common->in2_sub_cumh_sel_rows;     // (0-n) column
            if (idxp1 % d_common->in2_sub_cumh_sel_rows == 0) {
                row = d_common->in2_sub_cumh_sel_rows - 1;
                col = col - 1;
            }

            ori_row = row + d_common->in2_sub_cumh_sel_rowlow - 1;
            ori_col = col + d_common->in2_sub_cumh_sel_collow - 1;
            d_unique[bx].d_in2_sub_cumh_sel[ei_new] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * d_common->in2_sub_cumh_rows +
                                    ori_row];
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	SELECTION 2, SUBTRACTION
        //====================================================================================================

        //==================================================
        //	SELECTION 2
        //==================================================

        for (ei_new = tx; ei_new < d_common->in2_sub2_elem;
             ei_new += blockDim.x) {

            int idxp1 = ei_new + 1;
            row = idxp1 % d_common->in2_sub2_rows - 1; // (0-n) row
            col = idxp1 / d_common->in2_sub2_rows;     // (0-n) column
            if (idxp1 % d_common->in2_sub2_rows == 0) {
                row = d_common->in2_sub2_rows - 1;
                col = col - 1;
            }

            ori_row = row + d_common->in2_sub_cumh_sel2_rowlow - 1;
            ori_col = col + d_common->in2_sub_cumh_sel2_collow - 1;
            d_unique[bx].d_in2_sqr_sub2[ei_new] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * d_common->in2_sub_cumh_rows +
                                    ori_row];
        }

        //==================================================
        //	SYNCHRONIZE THREADS
        //==================================================

        __syncthreads();

        //==================================================
        //	SUBTRACTION
        //==================================================

        for (ei_new = tx; ei_new < d_common->in2_sub2_elem;
             ei_new += blockDim.x) {

            d_unique[bx].d_in2_sqr_sub2[ei_new] =
                d_unique[bx].d_in2_sub_cumh_sel[ei_new] -
                d_unique[bx].d_in2_sqr_sub2[ei_new];
        }

        //======================================================================================================================================================
        //	SYNCHRONIZE THREADS
        //======================================================================================================================================================

        __syncthreads();

        //======================================================================================================================================================
        //	FINAL
        //======================================================================================================================================================

        //====================================================================================================
        //	DENOMINATOR A		SAVE RESULT IN CUMULATIVE SUM A2
        //====================================================================================================

        for (ei_new = tx; ei_new < d_common->in2_sub2_elem;
             ei_new += blockDim.x) {

            temp = d_unique[bx].d_in2_sub2[ei_new];
            temp2 = d_unique[bx].d_in2_sqr_sub2[ei_new] -
                    (temp * temp / d_common->in_elem);
            if (temp2 < 0.0f) {
                temp2 = 0.0f;
            }
            d_unique[bx].d_in2_sqr_sub2[ei_new] = sqrtf(temp2);
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	MULTIPLICATION
        //====================================================================================================

        for (ei_new = tx; ei_new < d_common->in_sqr_elem;
             ei_new += blockDim.x) {

            temp = d_in[ei_new];
            d_unique[bx].d_in_sqr[ei_new] = temp * temp;
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	IN SUM
        //====================================================================================================

        for (ei_new = tx; ei_new < d_common->in_cols;
             ei_new += blockDim.x) {

            sum = 0.0f;
#pragma unroll 4
            for (i = 0; i < d_common->in_rows; i++) {
                sum = sum + d_in[ei_new * d_common->in_rows + i];
            }
            in_partial_sum[ei_new] = sum;
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	IN_SQR SUM
        //====================================================================================================

        for (ei_new = tx; ei_new < d_common->in_sqr_rows;
             ei_new += blockDim.x) {

            sum = 0.0f;
#pragma unroll 4
            for (i = 0; i < d_common->in_sqr_cols; i++) {
                sum += d_unique[bx].d_in_sqr[ei_new +
                                             d_common->in_sqr_rows * i];
            }
            in_sqr_partial_sum[ei_new] = sum;
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	FINAL SUMMATION
        //====================================================================================================

        if (tx == 0) {

            in_final_sum = 0.0f;
            for (i = 0; i < d_common->in_cols; i++) {
                in_final_sum += in_partial_sum[i];
            }

        } else if (tx == 1) {

            in_sqr_final_sum = 0.0f;
            for (i = 0; i < d_common->in_sqr_cols; i++) {
                in_sqr_final_sum += in_sqr_partial_sum[i];
            }
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	DENOMINATOR T
        //====================================================================================================

        if (tx == 0) {

            mean = in_final_sum /
                   d_common
                       ->in_elem; // gets mean (average) value of element in ROI
            mean_sqr = mean * mean;
            variance = (in_sqr_final_sum / d_common->in_elem) -
                       mean_sqr;        // gets variance of ROI
            deviation = sqrtf(variance); // gets standard deviation of ROI

            denomT = sqrtf(float(d_common->in_elem - 1)) * deviation;
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	DENOMINATOR		SAVE RESULT IN CUMULATIVE SUM A2
        //====================================================================================================

        for (ei_new = tx; ei_new < d_common->in2_sub2_elem;
             ei_new += blockDim.x) {

            d_unique[bx].d_in2_sqr_sub2[ei_new] =
                d_unique[bx].d_in2_sqr_sub2[ei_new] * denomT;
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	NUMERATOR	SAVE RESULT IN CONVOLUTION
        //====================================================================================================

        for (ei_new = tx; ei_new < d_common->conv_elem;
             ei_new += blockDim.x) {

            d_unique[bx].d_conv[ei_new] =
                d_unique[bx].d_conv[ei_new] -
                d_unique[bx].d_in2_sub2[ei_new] * in_final_sum /
                    d_common->in_elem;
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	CORRELATION	SAVE RESULT IN CUMULATIVE SUM A2
        //====================================================================================================

        for (ei_new = tx; ei_new < d_common->in2_sub2_elem;
             ei_new += blockDim.x) {

            d_unique[bx].d_in2_sqr_sub2[ei_new] =
                d_unique[bx].d_conv[ei_new] /
                d_unique[bx].d_in2_sqr_sub2[ei_new];
        }

        //======================================================================================================================================================
        //	SYNCHRONIZE THREADS
        //======================================================================================================================================================

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

        for (ei_new = tx; ei_new < d_common->tMask_elem;
             ei_new += blockDim.x) {

            location = tMask_col * d_common->tMask_rows + tMask_row;

            if (ei_new == location) {
                d_unique[bx].d_tMask[ei_new] = 1;
            } else {
                d_unique[bx].d_tMask[ei_new] = 0;
            }
        }

        //======================================================================================================================================================
        //	SYNCHRONIZE THREADS
        //======================================================================================================================================================

        __syncthreads();

        //======================================================================================================================================================
        //	MASK CONVOLUTION
        //======================================================================================================================================================

        for (ei_new = tx; ei_new < d_common->mask_conv_elem;
             ei_new += blockDim.x) {

            int idxp1 = ei_new + 1;
            ic = idxp1 % d_common->mask_conv_rows;     // (1-n)
            jc = idxp1 / d_common->mask_conv_rows + 1; // (1-n)
            if (idxp1 % d_common->mask_conv_rows == 0) {
                ic = d_common->mask_conv_rows;
                jc = jc - 1;
            }

            j = jc + d_common->mask_conv_joffset;
            jp1 = j + 1;
            if (d_common->mask_cols < jp1) {
                ja1 = jp1 - d_common->mask_cols;
            } else {
                ja1 = 1;
            }
            if (d_common->tMask_cols < j) {
                ja2 = d_common->tMask_cols;
            } else {
                ja2 = j;
            }

            i = ic + d_common->mask_conv_ioffset;
            ip1 = i + 1;

            if (d_common->mask_rows < ip1) {
                ia1 = ip1 - d_common->mask_rows;
            } else {
                ia1 = 1;
            }
            if (d_common->tMask_rows < i) {
                ia2 = d_common->tMask_rows;
            } else {
                ia2 = i;
            }

            s = 0.0f;

            for (ja = ja1; ja <= ja2; ja++) {
                jb = jp1 - ja;
#pragma unroll 4
                for (ia = ia1; ia <= ia2; ia++) {
                    ib = ip1 - ia;
                    s = s + d_unique[bx].d_tMask[d_common->tMask_rows *
                                                 (ja - 1) +
                                                 ia - 1];
                }
            }

            d_unique[bx].d_mask_conv[ei_new] =
                d_unique[bx].d_in2_sqr_sub2[ei_new] * s;
        }

        //======================================================================================================================================================
        //	SYNCHRONIZE THREADS
        //======================================================================================================================================================

        __syncthreads();

        //======================================================================================================================================================
        //	MAXIMUM VALUE
        //======================================================================================================================================================

        //====================================================================================================
        //	INITIAL SEARCH
        //====================================================================================================

        for (ei_new = tx; ei_new < d_common->mask_conv_rows;
             ei_new += blockDim.x) {

            largest_value = 0.0f;
            largest_coordinate = 0;

            int base = ei_new * d_common->mask_conv_rows;
#pragma unroll 4
            for (i = 0; i < d_common->mask_conv_cols; i++) {
                largest_coordinate_current = base + i;
                largest_value_current =
                    fabsf(d_unique[bx].d_mask_conv[largest_coordinate_current]);
                if (largest_value_current > largest_value) {
                    largest_coordinate = largest_coordinate_current;
                    largest_value = largest_value_current;
                }
            }
            par_max_coo[ei_new] = largest_coordinate;
            par_max_val[ei_new] = largest_value;
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	FINAL SEARCH
        //====================================================================================================

        if (tx == 0) {

            fin_max_val = 0.0f;
            fin_max_coo = 0;
            for (i = 0; i < d_common->mask_conv_rows; i++) {
                if (par_max_val[i] > fin_max_val) {
                    fin_max_val = par_max_val[i];
                    fin_max_coo = par_max_coo[i];
                }
            }

            largest_row =
                (fin_max_coo + 1) % d_common->mask_conv_rows - 1; // (0-n) row
            largest_col =
                (fin_max_coo + 1) / d_common->mask_conv_rows; // (0-n) column
            if ((fin_max_coo + 1) % d_common->mask_conv_rows == 0) {
                largest_row = d_common->mask_conv_rows - 1;
                largest_col = largest_col - 1;
            }

            largest_row =
                largest_row + 1; // compensate to match MATLAB format (1-n)
            largest_col =
                largest_col + 1; // compensate to match MATLAB format (1-n)
            offset_row = largest_row - d_common->in_rows -
                         (d_common->sSize - d_common->tSize);
            offset_col = largest_col - d_common->in_cols -
                         (d_common->sSize - d_common->tSize);
            pointer = d_common_change->frame_no +
                      d_unique[bx].point_no * d_common->no_frames;
            d_unique[bx].d_tRowLoc[pointer] =
                d_unique[bx].d_Row[d_unique[bx].point_no] + offset_row;
            d_unique[bx].d_tColLoc[pointer] =
                d_unique[bx].d_Col[d_unique[bx].point_no] + offset_col;
        }

        //======================================================================================================================================================
        //	SYNCHRONIZE THREADS
        //======================================================================================================================================================

        __syncthreads();
    }

    //===============================================================================================================================================================================================================
    //===============================================================================================================================================================================================================
    //	COORDINATE AND TEMPLATE UPDATE
    //===============================================================================================================================================================================================================
    //===============================================================================================================================================================================================================

    if (d_common_change->frame_no != 0 &&
        (d_common_change->frame_no) % 10 == 0) {

        // update coordinate
        loc_pointer = d_unique[bx].point_no * d_common->no_frames +
                      d_common_change->frame_no;
        d_unique[bx].d_Row[d_unique[bx].point_no] =
            d_unique[bx].d_tRowLoc[loc_pointer];
        d_unique[bx].d_Col[d_unique[bx].point_no] =
            d_unique[bx].d_tColLoc[loc_pointer];

        // pointer to template (same as earlier)
        d_in = &d_unique[bx].d_T[d_unique[bx].in_pointer];

        for (ei_new = tx; ei_new < d_common->in_elem;
             ei_new += blockDim.x) {

            int idxp1 = ei_new + 1;
            row = idxp1 % d_common->in_rows - 1;     // (0-n) row
            col = idxp1 / d_common->in_rows;         // (0-n) column
            if (idxp1 % d_common->in_rows == 0) {
                row = d_common->in_rows - 1;
                col = col - 1;
            }

            ori_row = d_unique[bx].d_Row[d_unique[bx].point_no] - 25 + row - 1;
            ori_col = d_unique[bx].d_Col[d_unique[bx].point_no] - 25 + col - 1;
            ori_pointer = __mul24(ori_col, d_common->frame_rows) + ori_row;

            float oldv = d_in[ei_new];
            float newv = d_common_change->d_frame[ori_pointer];
            d_in[ei_new] = d_common->alpha * oldv +
                           (1.0f - d_common->alpha) * newv;
        }
    }
}
