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
    __shared__ float in_partial_sum[51];     // WATCH THIS !!! HARDCODEED VALUE
    __shared__ float in_sqr_partial_sum[51]; // WATCH THIS !!! HARDCODEED VALUE
    __shared__ float in_final_sum;
    __shared__ float in_sqr_final_sum;
    float mean;
    float mean_sqr;
    float variance;
    float deviation;
    __shared__ float denomT;
    __shared__ float par_max_val[131]; // WATCH THIS !!! HARDCODEED VALUE
    __shared__ int par_max_coo[131];   // WATCH THIS !!! HARDCODEED VALUE
    int pointer;
    __shared__ float d_in_mod_temp[2601];
    int ori_pointer;
    int loc_pointer;

    //======================================================================================================================================================
    //	THREAD PARAMETERS
    //======================================================================================================================================================

    const int bx = blockIdx.x;  // get current horizontal block index (0-n)
    const int tx = threadIdx.x; // get current horizontal thread index (0-n)
    const int stride = blockDim.x;
    int ei_new;

    // cache some invariants in registers
    const int frame_rows = d_common->frame_rows;
    const int in_rows = d_common->in_rows;
    const int in_cols = d_common->in_cols;
    const int in_elem = d_common->in_elem;
    const int in2_rows = d_common->in2_rows;
    const int in2_cols = d_common->in2_cols;
    const int conv_rows = d_common->conv_rows;
    const int conv_elem = d_common->conv_elem;
    const int in2_pad_cumv_rows = d_common->in2_pad_cumv_rows;
    const int in2_pad_cumv_cols = d_common->in2_pad_cumv_cols;
    const int in2_sub_cumh_rows = d_common->in2_sub_cumh_rows;
    const int in2_sub_cumh_cols = d_common->in2_sub_cumh_cols;
    const int in2_sub2_rows = d_common->in2_sub2_rows;
    const int in2_sub2_cols = d_common->in2_sub2_cols;
    const int mask_conv_rows = d_common->mask_conv_rows;
    const int mask_conv_cols = d_common->mask_conv_cols;
    const int tMask_rows = d_common->tMask_rows;
    const int tMask_cols = d_common->tMask_cols;

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
        for (ei_new = tx; ei_new < in_elem; ei_new += stride) {

            // figure out row/col location in new matrix
            row = (ei_new + 1) % in_rows - 1;     // (0-n) row
            col = (ei_new + 1) / in_rows + 1 - 1; // (0-n) column
            if ((ei_new + 1) % in_rows == 0) {
                row = in_rows - 1;
                col = col - 1;
            }

            // figure out row/col location in corresponding new template area in
            // image and give to every thread (get top left corner and progress
            // down and right)
            ori_row = d_unique[bx].d_Row[d_unique[bx].point_no] - 25 + row - 1;
            ori_col = d_unique[bx].d_Col[d_unique[bx].point_no] - 25 + col - 1;
            ori_pointer = ori_col * frame_rows + ori_row;

            // update template
            d_in[col * in_rows + row] =
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

        const int in2_elem = d_common->in2_elem;

        // work
        for (ei_new = tx; ei_new < in2_elem; ei_new += stride) {

            // figure out row/col location in new matrix
            row = (ei_new + 1) % in2_rows - 1;     // (0-n) row
            col = (ei_new + 1) / in2_rows + 1 - 1; // (0-n) column
            if ((ei_new + 1) % in2_rows == 0) {
                row = in2_rows - 1;
                col = col - 1;
            }

            // figure out corresponding location in old matrix and copy values
            // to new matrix
            ori_row = row + in2_rowlow - 1;
            ori_col = col + in2_collow - 1;
            d_unique[bx].d_in2[ei_new] =
                d_common_change
                    ->d_frame[ori_col * frame_rows + ori_row];
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
        for (ei_new = tx; ei_new < in_elem; ei_new += stride) {

            // figure out row/col location in padded array
            row = (ei_new + 1) % in_rows - 1;     // (0-n) row
            col = (ei_new + 1) / in_rows + 1 - 1; // (0-n) column
            if ((ei_new + 1) % in_rows == 0) {
                row = in_rows - 1;
                col = col - 1;
            }

            // execution
            rot_row = (in_rows - 1) - row;
            rot_col = (in_rows - 1) - col;
            d_in_mod_temp[ei_new] = d_in[rot_col * in_rows + rot_row];
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	ACTUAL CONVOLUTION
        //====================================================================================================

        const int joffset = d_common->joffset;
        const int ioffset = d_common->ioffset;

        for (ei_new = tx; ei_new < conv_elem; ei_new += stride) {

            // figure out row/col location in array
            ic = (ei_new + 1) % conv_rows;     // (1-n)
            jc = (ei_new + 1) / conv_rows + 1; // (1-n)
            if ((ei_new + 1) % conv_rows == 0) {
                ic = conv_rows;
                jc = jc - 1;
            }

            j = jc + joffset;
            jp1 = j + 1;
            if (in2_cols < jp1) {
                ja1 = jp1 - in2_cols;
            } else {
                ja1 = 1;
            }
            if (in_cols < j) {
                ja2 = in_cols;
            } else {
                ja2 = j;
            }

            i = ic + ioffset;
            ip1 = i + 1;

            if (in2_rows < ip1) {
                ia1 = ip1 - in2_rows;
            } else {
                ia1 = 1;
            }
            if (in_rows < i) {
                ia2 = in_rows;
            } else {
                ia2 = i;
            }

            s = 0.0f;

            for (ja = ja1; ja <= ja2; ja++) {
                jb = jp1 - ja;
#pragma unroll
                for (ia = ia1; ia <= ia2; ia++) {
                    ib = ip1 - ia;
                    s = fmaf(d_in_mod_temp[in_rows * (ja - 1) + ia - 1],
                             d_unique[bx].d_in2[in2_rows * (jb - 1) + ib - 1],
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

        const int in2_pad_cumv_elem = d_common->in2_pad_cumv_elem;
        const int in2_pad_add_rows = d_common->in2_pad_add_rows;
        const int in2_pad_add_cols = d_common->in2_pad_add_cols;

        //==================================================
        //	PADD ARRAY
        //==================================================

        for (ei_new = tx; ei_new < in2_pad_cumv_elem; ei_new += stride) {

            // figure out row/col location in padded array
            row = (ei_new + 1) % in2_pad_cumv_rows - 1; // (0-n) row
            col = (ei_new + 1) / in2_pad_cumv_rows + 1 -
                  1; // (0-n) column
            if ((ei_new + 1) % in2_pad_cumv_rows == 0) {
                row = in2_pad_cumv_rows - 1;
                col = col - 1;
            }

            // execution
            if (row > (in2_pad_add_rows -
                       1) && // do if has numbers in original array
                row < (in2_pad_add_rows + in2_rows) &&
                col > (in2_pad_add_cols - 1) &&
                col < (in2_pad_add_cols + in2_cols)) {
                ori_row = row - in2_pad_add_rows;
                ori_col = col - in2_pad_add_cols;
                d_unique[bx].d_in2_pad_cumv[ei_new] =
                    d_unique[bx].d_in2[ori_col * in2_rows + ori_row];
            } else { // do if otherwise
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

        for (ei_new = tx; ei_new < in2_pad_cumv_cols; ei_new += stride) {

            // figure out column position
            pos_ori = ei_new * in2_pad_cumv_rows;

            // variables
            sum = 0.0f;

            // loop through all rows
#pragma unroll 4
            for (position = pos_ori;
                 position < pos_ori + in2_pad_cumv_rows;
                 position++) {
                sum += d_unique[bx].d_in2_pad_cumv[position];
                d_unique[bx].d_in2_pad_cumv[position] = sum;
            }
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	SELECTION
        //====================================================================================================

        const int in2_pad_cumv_sel_elem = d_common->in2_pad_cumv_sel_elem;
        const int in2_pad_cumv_sel_rows = d_common->in2_pad_cumv_sel_rows;
        const int in2_pad_cumv_sel_rowlow = d_common->in2_pad_cumv_sel_rowlow;
        const int in2_pad_cumv_sel_collow = d_common->in2_pad_cumv_sel_collow;

        for (ei_new = tx; ei_new < in2_pad_cumv_sel_elem; ei_new += stride) {

            // figure out row/col location in new matrix
            row =
                (ei_new + 1) % in2_pad_cumv_sel_rows - 1; // (0-n) row
            col = (ei_new + 1) / in2_pad_cumv_sel_rows + 1 -
                  1; // (0-n) column
            if ((ei_new + 1) % in2_pad_cumv_sel_rows == 0) {
                row = in2_pad_cumv_sel_rows - 1;
                col = col - 1;
            }

            // figure out corresponding location in old matrix and copy values
            // to new matrix
            ori_row = row + in2_pad_cumv_sel_rowlow - 1;
            ori_col = col + in2_pad_cumv_sel_collow - 1;
            d_unique[bx].d_in2_pad_cumv_sel[ei_new] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * in2_pad_cumv_rows +
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

        const int in2_sub_cumh_elem = d_common->in2_sub_cumh_elem;
        const int in2_sub_cumh_rows_loc = in2_sub_cumh_rows;
        const int in2_sub_cumh_sel2_rowlow = d_common->in2_pad_cumv_sel2_rowlow;
        const int in2_sub_cumh_sel2_collow = d_common->in2_pad_cumv_sel2_collow;

        for (ei_new = tx; ei_new < in2_sub_cumh_elem; ei_new += stride) {

            // figure out row/col location in new matrix
            row = (ei_new + 1) % in2_sub_cumh_rows_loc - 1; // (0-n) row
            col = (ei_new + 1) / in2_sub_cumh_rows_loc + 1 -
                  1; // (0-n) column
            if ((ei_new + 1) % in2_sub_cumh_rows_loc == 0) {
                row = in2_sub_cumh_rows_loc - 1;
                col = col - 1;
            }

            // figure out corresponding location in old matrix and copy values
            // to new matrix
            ori_row = row + in2_sub_cumh_sel2_rowlow - 1;
            ori_col = col + in2_sub_cumh_sel2_collow - 1;
            d_unique[bx].d_in2_sub_cumh[ei_new] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * in2_pad_cumv_rows +
                                    ori_row];
        }

        //==================================================
        //	SYNCHRONIZE THREADS
        //==================================================

        __syncthreads();

        //==================================================
        //	SUBTRACTION
        //==================================================

        for (ei_new = tx; ei_new < in2_sub_cumh_elem; ei_new += stride) {

            // subtract
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

        for (ei_new = tx; ei_new < in2_sub_cumh_rows_loc; ei_new += stride) {

            // figure out row position
            pos_ori = ei_new;

            // variables
            sum = 0.0f;

            // loop through all rows
#pragma unroll 4
            for (position = pos_ori;
                 position < pos_ori + in2_sub_cumh_elem;
                 position += in2_sub_cumh_rows_loc) {
                sum += d_unique[bx].d_in2_sub_cumh[position];
                d_unique[bx].d_in2_sub_cumh[position] = sum;
            }
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	SELECTION
        //====================================================================================================

        const int in2_sub_cumh_sel_elem = d_common->in2_sub_cumh_sel_elem;
        const int in2_sub_cumh_sel_rows = d_common->in2_sub_cumh_sel_rows;
        const int in2_sub_cumh_sel_rowlow = d_common->in2_sub_cumh_sel_rowlow;
        const int in2_sub_cumh_sel_collow = d_common->in2_sub_cumh_sel_collow;

        for (ei_new = tx; ei_new < in2_sub_cumh_sel_elem; ei_new += stride) {

            // figure out row/col location in new matrix
            row =
                (ei_new + 1) % in2_sub_cumh_sel_rows - 1; // (0-n) row
            col = (ei_new + 1) / in2_sub_cumh_sel_rows + 1 -
                  1; // (0-n) column
            if ((ei_new + 1) % in2_sub_cumh_sel_rows == 0) {
                row = in2_sub_cumh_sel_rows - 1;
                col = col - 1;
            }

            // figure out corresponding location in old matrix and copy values
            // to new matrix
            ori_row = row + in2_sub_cumh_sel_rowlow - 1;
            ori_col = col + in2_sub_cumh_sel_collow - 1;
            d_unique[bx].d_in2_sub_cumh_sel[ei_new] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * in2_sub_cumh_rows_loc +
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

        const int in2_sub2_elem = d_common->in2_sub2_elem;
        const int in2_sub2_rows_loc = in2_sub2_rows;
        const int in2_sub_cumh_sel2_rowlow2 = d_common->in2_sub_cumh_sel2_rowlow;
        const int in2_sub_cumh_sel2_collow2 = d_common->in2_sub_cumh_sel2_collow;

        for (ei_new = tx; ei_new < in2_sub2_elem; ei_new += stride) {

            // figure out row/col location in new matrix
            row = (ei_new + 1) % in2_sub2_rows_loc - 1; // (0-n) row
            col =
                (ei_new + 1) / in2_sub2_rows_loc + 1 - 1; // (0-n) column
            if ((ei_new + 1) % in2_sub2_rows_loc == 0) {
                row = in2_sub2_rows_loc - 1;
                col = col - 1;
            }

            // figure out corresponding location in old matrix and copy values
            // to new matrix
            ori_row = row + in2_sub_cumh_sel2_rowlow2 - 1;
            ori_col = col + in2_sub_cumh_sel2_collow2 - 1;
            d_unique[bx].d_in2_sub2[ei_new] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * in2_sub_cumh_rows_loc +
                                    ori_row];
        }

        //==================================================
        //	SYNCHRONIZE THREADS
        //==================================================

        __syncthreads();

        //==================================================
        //	SUBTRACTION
        //==================================================

        for (ei_new = tx; ei_new < in2_sub2_elem; ei_new += stride) {

            // subtract
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

        const int in2_sqr_elem = d_common->in2_sqr_elem;

        for (ei_new = tx; ei_new < in2_sqr_elem; ei_new += stride) {

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

        const int in2_sqr_rows = d_common->in2_sqr_rows;
        const int in2_sqr_cols = d_common->in2_sqr_cols;

        //==================================================
        //	PAD ARRAY
        //==================================================

        for (ei_new = tx; ei_new < in2_pad_cumv_elem; ei_new += stride) {

            // figure out row/col location in padded array
            row = (ei_new + 1) % in2_pad_cumv_rows - 1; // (0-n) row
            col = (ei_new + 1) / in2_pad_cumv_rows + 1 -
                  1; // (0-n) column
            if ((ei_new + 1) % in2_pad_cumv_rows == 0) {
                row = in2_pad_cumv_rows - 1;
                col = col - 1;
            }

            // execution
            if (row > (in2_pad_add_rows -
                       1) && // do if has numbers in original array
                row < (in2_pad_add_rows + in2_sqr_rows) &&
                col > (in2_pad_add_cols - 1) &&
                col < (in2_pad_add_cols + in2_sqr_cols)) {
                ori_row = row - in2_pad_add_rows;
                ori_col = col - in2_pad_add_cols;
                d_unique[bx].d_in2_pad_cumv[ei_new] =
                    d_unique[bx]
                        .d_in2_sqr[ori_col * in2_sqr_rows + ori_row];
            } else { // do if otherwise
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

        for (ei_new = tx; ei_new < in2_pad_cumv_cols; ei_new += stride) {

            // figure out column position
            pos_ori = ei_new * in2_pad_cumv_rows;

            // variables
            sum = 0.0f;

            // loop through all rows
#pragma unroll 4
            for (position = pos_ori;
                 position < pos_ori + in2_pad_cumv_rows;
                 position++) {
                sum += d_unique[bx].d_in2_pad_cumv[position];
                d_unique[bx].d_in2_pad_cumv[position] = sum;
            }
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	SELECTION
        //====================================================================================================

        for (ei_new = tx; ei_new < in2_pad_cumv_sel_elem; ei_new += stride) {

            // figure out row/col location in new matrix
            row =
                (ei_new + 1) % in2_pad_cumv_sel_rows - 1; // (0-n) row
            col = (ei_new + 1) / in2_pad_cumv_sel_rows + 1 -
                  1; // (0-n) column
            if ((ei_new + 1) % in2_pad_cumv_sel_rows == 0) {
                row = in2_pad_cumv_sel_rows - 1;
                col = col - 1;
            }

            // figure out corresponding location in old matrix and copy values
            // to new matrix
            ori_row = row + in2_pad_cumv_sel_rowlow - 1;
            ori_col = col + in2_pad_cumv_sel_collow - 1;
            d_unique[bx].d_in2_pad_cumv_sel[ei_new] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * in2_pad_cumv_rows +
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

        for (ei_new = tx; ei_new < in2_sub_cumh_elem; ei_new += stride) {

            // figure out row/col location in new matrix
            row = (ei_new + 1) % in2_sub_cumh_rows_loc - 1; // (0-n) row
            col = (ei_new + 1) / in2_sub_cumh_rows_loc + 1 -
                  1; // (0-n) column
            if ((ei_new + 1) % in2_sub_cumh_rows_loc == 0) {
                row = in2_sub_cumh_rows_loc - 1;
                col = col - 1;
            }

            // figure out corresponding location in old matrix and copy values
            // to new matrix
            ori_row = row + in2_sub_cumh_sel2_rowlow - 1;
            ori_col = col + in2_sub_cumh_sel2_collow - 1;
            d_unique[bx].d_in2_sub_cumh[ei_new] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * in2_pad_cumv_rows +
                                    ori_row];
        }

        //==================================================
        //	SYNCHRONIZE THREADS
        //==================================================

        __syncthreads();

        //==================================================
        //	SUBTRACTION
        //==================================================

        for (ei_new = tx; ei_new < in2_sub_cumh_elem; ei_new += stride) {

            // subtract
            d_unique[bx].d_in2_sub_cumh[ei_new] =
                d_unique[bx].d_in2_pad_cumv_sel[ei_new] -
                d_unique[bx].d_in2_sub_cumh[ei_new];
        }

        //==================================================
        //	HORIZONTAL CUMULATIVE SUM
        //==================================================

        for (ei_new = tx; ei_new < in2_sub_cumh_rows_loc; ei_new += stride) {

            // figure out row position
            pos_ori = ei_new;

            // variables
            sum = 0.0f;

            // loop through all rows
#pragma unroll 4
            for (position = pos_ori;
                 position < pos_ori + in2_sub_cumh_elem;
                 position += in2_sub_cumh_rows_loc) {
                sum += d_unique[bx].d_in2_sub_cumh[position];
                d_unique[bx].d_in2_sub_cumh[position] = sum;
            }
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	SELECTION
        //====================================================================================================

        for (ei_new = tx; ei_new < in2_sub_cumh_sel_elem; ei_new += stride) {

            // figure out row/col location in new matrix
            row =
                (ei_new + 1) % in2_sub_cumh_sel_rows - 1; // (0-n) row
            col = (ei_new + 1) / in2_sub_cumh_sel_rows + 1 -
                  1; // (0-n) column
            if ((ei_new + 1) % in2_sub_cumh_sel_rows == 0) {
                row = in2_sub_cumh_sel_rows - 1;
                col = col - 1;
            }

            // figure out corresponding location in old matrix and copy values
            // to new matrix
            ori_row = row + in2_sub_cumh_sel_rowlow - 1;
            ori_col = col + in2_sub_cumh_sel_collow - 1;
            d_unique[bx].d_in2_sub_cumh_sel[ei_new] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * in2_sub_cumh_rows_loc +
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

        for (ei_new = tx; ei_new < in2_sub2_elem; ei_new += stride) {

            // figure out row/col location in new matrix
            row = (ei_new + 1) % in2_sub2_rows_loc - 1; // (0-n) row
            col =
                (ei_new + 1) / in2_sub2_rows_loc + 1 - 1; // (0-n) column
            if ((ei_new + 1) % in2_sub2_rows_loc == 0) {
                row = in2_sub2_rows_loc - 1;
                col = col - 1;
            }

            // figure out corresponding location in old matrix and copy values
            // to new matrix
            ori_row = row + in2_sub_cumh_sel2_rowlow2 - 1;
            ori_col = col + in2_sub_cumh_sel2_collow2 - 1;
            d_unique[bx].d_in2_sqr_sub2[ei_new] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * in2_sub_cumh_rows_loc +
                                    ori_row];
        }

        //==================================================
        //	SYNCHRONIZE THREADS
        //==================================================

        __syncthreads();

        //==================================================
        //	SUBTRACTION
        //==================================================

        for (ei_new = tx; ei_new < in2_sub2_elem; ei_new += stride) {

            // subtract
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

        const float inv_in_elem = 1.0f / (float)in_elem;

        for (ei_new = tx; ei_new < in2_sub2_elem; ei_new += stride) {

            temp = d_unique[bx].d_in2_sub2[ei_new];
            temp2 = d_unique[bx].d_in2_sqr_sub2[ei_new] -
                    (temp * temp * inv_in_elem);
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

        const int in_sqr_elem = d_common->in_sqr_elem;
        const int in_sqr_rows = d_common->in_sqr_rows;
        const int in_sqr_cols = d_common->in_sqr_cols;

        for (ei_new = tx; ei_new < in_sqr_elem; ei_new += stride) {

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

        for (ei_new = tx; ei_new < in_cols; ei_new += stride) {

            sum = 0.0f;
#pragma unroll 4
            for (i = 0; i < in_rows; i++) {

                sum += d_in[ei_new * in_rows + i];
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

        for (ei_new = tx; ei_new < in_sqr_rows; ei_new += stride) {

            sum = 0.0f;
#pragma unroll 4
            for (i = 0; i < in_sqr_cols; i++) {

                sum += d_unique[bx].d_in_sqr[ei_new + in_sqr_rows * i];
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

            float local_sum = 0.0f;
#pragma unroll
            for (i = 0; i < 51; i++) {
                if (i < in_cols) {
                    local_sum += in_partial_sum[i];
                }
            }
            in_final_sum = local_sum;

        } else if (tx == 1) {

            float local_sum_sqr = 0.0f;
#pragma unroll
            for (i = 0; i < 51; i++) {
                if (i < in_sqr_cols) {
                    local_sum_sqr += in_sqr_partial_sum[i];
                }
            }
            in_sqr_final_sum = local_sum_sqr;
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	DENOMINATOR T
        //====================================================================================================

        if (tx == 0) {

            mean = in_final_sum * inv_in_elem; // gets mean (average) value of element in ROI
            mean_sqr = mean * mean;
            variance = (in_sqr_final_sum * inv_in_elem) -
                       mean_sqr;        // gets variance of ROI
            variance = variance < 0.0f ? 0.0f : variance;
            deviation = sqrtf(variance); // gets standard deviation of ROI

            denomT = sqrtf((float)(in_elem - 1)) * deviation;
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	DENOMINATOR		SAVE RESULT IN CUMULATIVE SUM A2
        //====================================================================================================

        for (ei_new = tx; ei_new < in2_sub2_elem; ei_new += stride) {

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

        for (ei_new = tx; ei_new < conv_elem; ei_new += stride) {

            d_unique[bx].d_conv[ei_new] = d_unique[bx].d_conv[ei_new] -
                                          d_unique[bx].d_in2_sub2[ei_new] *
                                              in_final_sum * inv_in_elem;
        }

        //====================================================================================================
        //	SYNCHRONIZE THREADS
        //====================================================================================================

        __syncthreads();

        //====================================================================================================
        //	CORRELATION	SAVE RESULT IN CUMULATIVE SUM A2
        //====================================================================================================

        for (ei_new = tx; ei_new < in2_sub2_elem; ei_new += stride) {

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

        const int tMask_elem = d_common->tMask_elem;

        // work
        for (ei_new = tx; ei_new < tMask_elem; ei_new += stride) {

            location = tMask_col * tMask_rows + tMask_row;

            d_unique[bx].d_tMask[ei_new] = (ei_new == location) ? 1 : 0;
        }

        //======================================================================================================================================================
        //	SYNCHRONIZE THREADS
        //======================================================================================================================================================

        __syncthreads();

        //======================================================================================================================================================
        //	MASK CONVOLUTION
        //======================================================================================================================================================

        const int mask_conv_elem = d_common->mask_conv_elem;
        const int mask_rows = d_common->mask_rows;
        const int mask_cols = d_common->mask_cols;
        const int mask_conv_joffset = d_common->mask_conv_joffset;
        const int mask_conv_ioffset = d_common->mask_conv_ioffset;

        for (ei_new = tx; ei_new < mask_conv_elem; ei_new += stride) {

            // figure out row/col location in array
            ic = (ei_new + 1) % mask_conv_rows;     // (1-n)
            jc = (ei_new + 1) / mask_conv_rows + 1; // (1-n)
            if ((ei_new + 1) % mask_conv_rows == 0) {
                ic = mask_conv_rows;
                jc = jc - 1;
            }

            j = jc + mask_conv_joffset;
            jp1 = j + 1;
            if (mask_cols < jp1) {
                ja1 = jp1 - mask_cols;
            } else {
                ja1 = 1;
            }
            if (tMask_cols < j) {
                ja2 = tMask_cols;
            } else {
                ja2 = j;
            }

            i = ic + mask_conv_ioffset;
            ip1 = i + 1;

            if (mask_rows < ip1) {
                ia1 = ip1 - mask_rows;
            } else {
                ia1 = 1;
            }
            if (tMask_rows < i) {
                ia2 = tMask_rows;
            } else {
                ia2 = i;
            }

            s = 0.0f;

            for (ja = ja1; ja <= ja2; ja++) {
                jb = jp1 - ja;
#pragma unroll
                for (ia = ia1; ia <= ia2; ia++) {
                    (void)jb;
                    ib = ip1 - ia;
                    (void)ib;
                    s += d_unique[bx].d_tMask[tMask_rows * (ja - 1) + ia - 1];
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

        for (ei_new = tx; ei_new < mask_conv_rows; ei_new += stride) {

            largest_value = 0.0f;
            largest_coordinate = 0;

#pragma unroll 4
            for (i = 0; i < mask_conv_cols; i++) {
                largest_coordinate_current =
                    ei_new * mask_conv_rows + i;
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

#pragma unroll 4
            for (i = 0; i < mask_conv_rows; i++) {
                if (par_max_val[i] > fin_max_val) {
                    fin_max_val = par_max_val[i];
                    fin_max_coo = par_max_coo[i];
                }
            }

            // convert coordinate to row/col form
            largest_row =
                (fin_max_coo + 1) % mask_conv_rows - 1; // (0-n) row
            largest_col =
                (fin_max_coo + 1) / mask_conv_rows; // (0-n) column
            if ((fin_max_coo + 1) % mask_conv_rows == 0) {
                largest_row = mask_conv_rows - 1;
                largest_col = largest_col - 1;
            }

            // calculate offset
            largest_row =
                largest_row + 1; // compensate to match MATLAB format (1-n)
            largest_col =
                largest_col + 1; // compensate to match MATLAB format (1-n)
            offset_row = largest_row - in_rows -
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

    // if the last frame in the bath, update template
    if (d_common_change->frame_no != 0 &&
        (d_common_change->frame_no) % 10 == 0) {

        // update coordinate
        loc_pointer = d_unique[bx].point_no * d_common->no_frames +
                      d_common_change->frame_no;
        d_unique[bx].d_Row[d_unique[bx].point_no] =
            d_unique[bx].d_tRowLoc[loc_pointer];
        d_unique[bx].d_Col[d_unique[bx].point_no] =
            d_unique[bx].d_tColLoc[loc_pointer];

        d_in = &d_unique[bx].d_T[d_unique[bx].in_pointer];

        // work
        for (ei_new = tx; ei_new < in_elem; ei_new += stride) {

            // figure out row/col location in new matrix
            row = (ei_new + 1) % in_rows - 1;     // (0-n) row
            col = (ei_new + 1) / in_rows + 1 - 1; // (0-n) column
            if ((ei_new + 1) % in_rows == 0) {
                row = in_rows - 1;
                col = col - 1;
            }

            // figure out row/col location in corresponding new template area in
            // image and give to every thread (get top left corner and progress
            // down and right)
            ori_row = d_unique[bx].d_Row[d_unique[bx].point_no] - 25 + row - 1;
            ori_col = d_unique[bx].d_Col[d_unique[bx].point_no] - 25 + col - 1;
            ori_pointer = ori_col * frame_rows + ori_row;

            // update template
            d_in[ei_new] = d_common->alpha * d_in[ei_new] +
                           (1.0f - d_common->alpha) *
                               d_common_change->d_frame[ori_pointer];
        }
    }
}
