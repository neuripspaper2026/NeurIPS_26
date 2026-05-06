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
    const int nthreads = blockDim.x;
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
        const int base_row = d_unique[bx].d_Row[d_unique[bx].point_no] - 25 - 1;
        const int base_col = d_unique[bx].d_Col[d_unique[bx].point_no] - 25 - 1;

        while (ei_new < in_elem) {

            // figure out row/col location in new matrix
            int t = ei_new + 1;
            row = t % in_rows - 1;     // (0-n) row
            col = t / in_rows + 1 - 1; // (0-n) column
            if (t % in_rows == 0) {
                row = in_rows - 1;
                col = col - 1;
            }

            // figure out row/col location in corresponding new template area in
            // image and give to every thread (get top left corner and progress
            // down and right)
            ori_row = base_row + row;
            ori_col = base_col + col;
            ori_pointer = ori_col * frame_rows + ori_row;

            // update template
            d_in[col * in_rows + row] =
                d_common_change->d_frame[ori_pointer];

            // go for second round
            ei_new += nthreads;
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

        const int sSize = d_common->sSize;
        in2_rowlow = d_unique[bx].d_Row[d_unique[bx].point_no] - sSize;
        in2_collow = d_unique[bx].d_Col[d_unique[bx].point_no] - sSize;

        const int in2_rows = d_common->in2_rows;
        const int in2_elem = d_common->in2_elem;

        // work
        ei_new = tx;
        while (ei_new < in2_elem) {

            int t = ei_new + 1;
            row = t % in2_rows - 1;     // (0-n) row
            col = t / in2_rows + 1 - 1; // (0-n) column
            if (t % in2_rows == 0) {
                row = in2_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_rowlow - 1;
            ori_col = col + in2_collow - 1;
            d_unique[bx].d_in2[ei_new] =
                d_common_change
                    ->d_frame[ori_col * d_common->frame_rows + ori_row];

            ei_new += nthreads;
        }

        __syncthreads();

        //======================================================================================================================================================
        //	CONVOLUTION
        //======================================================================================================================================================

        //====================================================================================================
        //	ROTATION
        //====================================================================================================

        d_in = &d_unique[bx].d_T[d_unique[bx].in_pointer];

        const int in_cols = d_common->in_cols;
        const int conv_rows = d_common->conv_rows;
        const int conv_elem = d_common->conv_elem;
        const int in_rows = d_common->in_rows;

        ei_new = tx;
        while (ei_new < in_elem) {

            int t = ei_new + 1;
            row = t % in_rows - 1;     // (0-n) row
            col = t / in_rows + 1 - 1; // (0-n) column
            if (t % in_rows == 0) {
                row = in_rows - 1;
                col = col - 1;
            }

            rot_row = (in_rows - 1) - row;
            rot_col = (in_rows - 1) - col;
            d_in_mod_temp[ei_new] = d_in[rot_col * in_rows + rot_row];

            ei_new += nthreads;
        }

        __syncthreads();

        //====================================================================================================
        //	ACTUAL CONVOLUTION
        //====================================================================================================

        const int in2_cols = d_common->in2_cols;
        const int joffset = d_common->joffset;
        const int ioffset = d_common->ioffset;

        ei_new = tx;
        while (ei_new < conv_elem) {

            int t = ei_new + 1;
            ic = t % conv_rows;     // (1-n)
            jc = t / conv_rows + 1; // (1-n)
            if (t % conv_rows == 0) {
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
#pragma unroll 4
                for (ia = ia1; ia <= ia2; ia++) {
                    ib = ip1 - ia;
                    s = fmaf(
                        d_in_mod_temp[in_rows * (ja - 1) + ia - 1],
                        d_unique[bx].d_in2[in2_rows * (jb - 1) + ib - 1], s);
                }
            }

            d_unique[bx].d_conv[ei_new] = s;

            ei_new += nthreads;
        }

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

        const int in2_pad_cumv_rows = d_common->in2_pad_cumv_rows;
        const int in2_pad_cumv_elem = d_common->in2_pad_cumv_elem;
        const int in2_pad_add_rows = d_common->in2_pad_add_rows;
        const int in2_pad_add_cols = d_common->in2_pad_add_cols;
        const int in2_pad_cumv_cols = d_common->in2_pad_cumv_cols;

        ei_new = tx;
        while (ei_new < in2_pad_cumv_elem) {

            int t = ei_new + 1;
            row = t % in2_pad_cumv_rows - 1; // (0-n) row
            col = t / in2_pad_cumv_rows + 1 - 1; // (0-n) column
            if (t % in2_pad_cumv_rows == 0) {
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

            ei_new += nthreads;
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

            ei_new += nthreads;
        }

        __syncthreads();

        //====================================================================================================
        //	SELECTION
        //====================================================================================================

        const int in2_pad_cumv_sel_rows = d_common->in2_pad_cumv_sel_rows;
        const int in2_pad_cumv_sel_elem = d_common->in2_pad_cumv_sel_elem;
        const int in2_pad_cumv_sel_rowlow = d_common->in2_pad_cumv_sel_rowlow;
        const int in2_pad_cumv_sel_collow = d_common->in2_pad_cumv_sel_collow;

        ei_new = tx;
        while (ei_new < in2_pad_cumv_sel_elem) {

            int t = ei_new + 1;
            row = t % in2_pad_cumv_sel_rows - 1; // (0-n) row
            col = t / in2_pad_cumv_sel_rows + 1 - 1; // (0-n) column
            if (t % in2_pad_cumv_sel_rows == 0) {
                row = in2_pad_cumv_sel_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_pad_cumv_sel_rowlow - 1;
            ori_col = col + in2_pad_cumv_sel_collow - 1;
            d_unique[bx].d_in2_pad_cumv_sel[ei_new] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * in2_pad_cumv_rows + ori_row];

            ei_new += nthreads;
        }

        __syncthreads();

        //====================================================================================================
        //	SELECTION 2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
        //====================================================================================================

        //==================================================
        //	SELECTION 2
        //==================================================

        const int in2_sub_cumh_rows = d_common->in2_sub_cumh_rows;
        const int in2_sub_cumh_elem = d_common->in2_sub_cumh_elem;
        const int in2_pad_cumv_sel2_rowlow = d_common->in2_pad_cumv_sel2_rowlow;
        const int in2_pad_cumv_sel2_collow = d_common->in2_pad_cumv_sel2_collow;

        ei_new = tx;
        while (ei_new < in2_sub_cumh_elem) {

            int t = ei_new + 1;
            row = t % in2_sub_cumh_rows - 1; // (0-n) row
            col = t / in2_sub_cumh_rows + 1 - 1; // (0-n) column
            if (t % in2_sub_cumh_rows == 0) {
                row = in2_sub_cumh_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_pad_cumv_sel2_rowlow - 1;
            ori_col = col + in2_pad_cumv_sel2_collow - 1;
            d_unique[bx].d_in2_sub_cumh[ei_new] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * in2_pad_cumv_rows + ori_row];

            ei_new += nthreads;
        }

        __syncthreads();

        //==================================================
        //	SUBTRACTION
        //==================================================

        ei_new = tx;
        while (ei_new < in2_sub_cumh_elem) {

            d_unique[bx].d_in2_sub_cumh[ei_new] =
                d_unique[bx].d_in2_pad_cumv_sel[ei_new] -
                d_unique[bx].d_in2_sub_cumh[ei_new];

            ei_new += nthreads;
        }

        __syncthreads();

        //==================================================
        //	HORIZONTAL CUMULATIVE SUM
        //==================================================

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

            ei_new += nthreads;
        }

        __syncthreads();

        //====================================================================================================
        //	SELECTION
        //====================================================================================================

        const int in2_sub_cumh_sel_rows = d_common->in2_sub_cumh_sel_rows;
        const int in2_sub_cumh_sel_elem = d_common->in2_sub_cumh_sel_elem;
        const int in2_sub_cumh_sel_rowlow = d_common->in2_sub_cumh_sel_rowlow;
        const int in2_sub_cumh_sel_collow = d_common->in2_sub_cumh_sel_collow;

        ei_new = tx;
        while (ei_new < in2_sub_cumh_sel_elem) {

            int t = ei_new + 1;
            row = t % in2_sub_cumh_sel_rows - 1; // (0-n) row
            col = t / in2_sub_cumh_sel_rows + 1 - 1; // (0-n) column
            if (t % in2_sub_cumh_sel_rows == 0) {
                row = in2_sub_cumh_sel_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_sub_cumh_sel_rowlow - 1;
            ori_col = col + in2_sub_cumh_sel_collow - 1;
            d_unique[bx].d_in2_sub_cumh_sel[ei_new] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * in2_sub_cumh_rows + ori_row];

            ei_new += nthreads;
        }

        __syncthreads();

        //====================================================================================================
        //	SELECTION 2, SUBTRACTION
        //====================================================================================================

        //==================================================
        //	SELECTION 2
        //==================================================

        const int in2_sub2_rows = d_common->in2_sub2_rows;
        const int in2_sub2_elem = d_common->in2_sub2_elem;
        const int in2_sub_cumh_sel2_rowlow = d_common->in2_sub_cumh_sel2_rowlow;
        const int in2_sub_cumh_sel2_collow = d_common->in2_sub_cumh_sel2_collow;

        ei_new = tx;
        while (ei_new < in2_sub2_elem) {

            int t = ei_new + 1;
            row = t % in2_sub2_rows - 1; // (0-n) row
            col = t / in2_sub2_rows + 1 - 1; // (0-n) column
            if (t % in2_sub2_rows == 0) {
                row = in2_sub2_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_sub_cumh_sel2_rowlow - 1;
            ori_col = col + in2_sub_cumh_sel2_collow - 1;
            d_unique[bx].d_in2_sub2[ei_new] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * in2_sub_cumh_rows + ori_row];

            ei_new += nthreads;
        }

        __syncthreads();

        //==================================================
        //	SUBTRACTION
        //==================================================

        ei_new = tx;
        while (ei_new < in2_sub2_elem) {

            d_unique[bx].d_in2_sub2[ei_new] =
                d_unique[bx].d_in2_sub_cumh_sel[ei_new] -
                d_unique[bx].d_in2_sub2[ei_new];

            ei_new += nthreads;
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

            ei_new += nthreads;
        }

        __syncthreads();

        //====================================================================================================
        //	PAD ARRAY, VERTICAL CUMULATIVE SUM
        //====================================================================================================

        ei_new = tx;
        while (ei_new < in2_pad_cumv_elem) {

            int t = ei_new + 1;
            row = t % in2_pad_cumv_rows - 1; // (0-n) row
            col = t / in2_pad_cumv_rows + 1 - 1; // (0-n) column
            if (t % in2_pad_cumv_rows == 0) {
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

            ei_new += nthreads;
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

            ei_new += nthreads;
        }

        __syncthreads();

        //====================================================================================================
        //	SELECTION
        //====================================================================================================

        ei_new = tx;
        while (ei_new < in2_pad_cumv_sel_elem) {

            int t = ei_new + 1;
            row = t % in2_pad_cumv_sel_rows - 1; // (0-n) row
            col = t / in2_pad_cumv_sel_rows + 1 - 1; // (0-n) column
            if (t % in2_pad_cumv_sel_rows == 0) {
                row = in2_pad_cumv_sel_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_pad_cumv_sel_rowlow - 1;
            ori_col = col + in2_pad_cumv_sel_collow - 1;
            d_unique[bx].d_in2_pad_cumv_sel[ei_new] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * in2_pad_cumv_rows + ori_row];

            ei_new += nthreads;
        }

        __syncthreads();

        //====================================================================================================
        //	SELECTION 2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
        //====================================================================================================

        ei_new = tx;
        while (ei_new < in2_sub_cumh_elem) {

            int t = ei_new + 1;
            row = t % in2_sub_cumh_rows - 1; // (0-n) row
            col = t / in2_sub_cumh_rows + 1 - 1; // (0-n) column
            if (t % in2_sub_cumh_rows == 0) {
                row = in2_sub_cumh_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_pad_cumv_sel2_rowlow - 1;
            ori_col = col + in2_pad_cumv_sel2_collow - 1;
            d_unique[bx].d_in2_sub_cumh[ei_new] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * in2_pad_cumv_rows + ori_row];

            ei_new += nthreads;
        }

        __syncthreads();

        ei_new = tx;
        while (ei_new < in2_sub_cumh_elem) {

            d_unique[bx].d_in2_sub_cumh[ei_new] =
                d_unique[bx].d_in2_pad_cumv_sel[ei_new] -
                d_unique[bx].d_in2_sub_cumh[ei_new];

            ei_new += nthreads;
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

            ei_new += nthreads;
        }

        __syncthreads();

        ei_new = tx;
        while (ei_new < in2_sub_cumh_sel_elem) {

            int t = ei_new + 1;
            row = t % in2_sub_cumh_sel_rows - 1; // (0-n) row
            col = t / in2_sub_cumh_sel_rows + 1 - 1; // (0-n) column
            if (t % in2_sub_cumh_sel_rows == 0) {
                row = in2_sub_cumh_sel_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_sub_cumh_sel_rowlow - 1;
            ori_col = col + in2_sub_cumh_sel_collow - 1;
            d_unique[bx].d_in2_sub_cumh_sel[ei_new] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * in2_sub_cumh_rows + ori_row];

            ei_new += nthreads;
        }

        __syncthreads();

        //====================================================================================================
        //	SELECTION 2, SUBTRACTION
        //====================================================================================================

        ei_new = tx;
        while (ei_new < in2_sub2_elem) {

            int t = ei_new + 1;
            row = t % in2_sub2_rows - 1; // (0-n) row
            col = t / in2_sub2_rows + 1 - 1; // (0-n) column
            if (t % in2_sub2_rows == 0) {
                row = in2_sub2_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_sub_cumh_sel2_rowlow - 1;
            ori_col = col + in2_sub_cumh_sel2_collow - 1;
            d_unique[bx].d_in2_sqr_sub2[ei_new] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * in2_sub_cumh_rows + ori_row];

            ei_new += nthreads;
        }

        __syncthreads();

        ei_new = tx;
        while (ei_new < in2_sub2_elem) {

            d_unique[bx].d_in2_sqr_sub2[ei_new] =
                d_unique[bx].d_in2_sub_cumh_sel[ei_new] -
                d_unique[bx].d_in2_sqr_sub2[ei_new];

            ei_new += nthreads;
        }

        __syncthreads();

        //======================================================================================================================================================
        //	FINAL
        //======================================================================================================================================================

        //====================================================================================================
        //	DENOMINATOR A		SAVE RESULT IN CUMULATIVE SUM A2
        //====================================================================================================

        ei_new = tx;
        while (ei_new < in2_sub2_elem) {

            temp = d_unique[bx].d_in2_sub2[ei_new];
            temp2 = d_unique[bx].d_in2_sqr_sub2[ei_new] -
                    (temp * temp / in_elem);
            if (temp2 < 0.0f) {
                temp2 = 0.0f;
            }
            d_unique[bx].d_in2_sqr_sub2[ei_new] = sqrtf(temp2);

            ei_new += nthreads;
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

            ei_new += nthreads;
        }

        __syncthreads();

        //====================================================================================================
        //	IN SUM
        //====================================================================================================

        ei_new = tx;
        while (ei_new < in_cols) {

            sum = 0.0f;
#pragma unroll 4
            for (i = 0; i < in_rows; i++) {
                sum += d_in[ei_new * in_rows + i];
            }
            in_partial_sum[ei_new] = sum;

            ei_new += nthreads;
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
                sum +=
                    d_unique[bx].d_in_sqr[ei_new + in_sqr_rows * i];
            }
            in_sqr_partial_sum[ei_new] = sum;

            ei_new += nthreads;
        }

        __syncthreads();

        //====================================================================================================
        //	FINAL SUMMATION
        //====================================================================================================

        if (tx == 0) {

            float local_sum = 0.0f;
#pragma unroll
            for (i = 0; i < in_cols; i++) {
                local_sum += in_partial_sum[i];
            }
            in_final_sum = local_sum;

        } else if (tx == 1) {

            float local_sum2 = 0.0f;
#pragma unroll
            for (i = 0; i < in_sqr_cols; i++) {
                local_sum2 += in_sqr_partial_sum[i];
            }
            in_sqr_final_sum = local_sum2;
        }

        __syncthreads();

        //====================================================================================================
        //	DENOMINATOR T
        //====================================================================================================

        if (tx == 0) {

            mean = in_final_sum /
                   in_elem; // gets mean (average) value of element in ROI
            mean_sqr = mean * mean;
            variance = (in_sqr_final_sum / in_elem) -
                       mean_sqr;        // gets variance of ROI
            deviation = sqrtf(variance); // gets standard deviation of ROI

            denomT = sqrtf(float(in_elem - 1)) * deviation;
        }

        __syncthreads();

        //====================================================================================================
        //	DENOMINATOR		SAVE RESULT IN CUMULATIVE SUM A2
        //====================================================================================================

        ei_new = tx;
        while (ei_new < in2_sub2_elem) {

            d_unique[bx].d_in2_sqr_sub2[ei_new] =
                d_unique[bx].d_in2_sqr_sub2[ei_new] * denomT;

            ei_new += nthreads;
        }

        __syncthreads();

        //====================================================================================================
        //	NUMERATOR	SAVE RESULT IN CONVOLUTION
        //====================================================================================================

        const float inv_in_elem = 1.0f / float(in_elem);

        ei_new = tx;
        while (ei_new < conv_elem) {

            d_unique[bx].d_conv[ei_new] =
                d_unique[bx].d_conv[ei_new] -
                d_unique[bx].d_in2_sub2[ei_new] *
                    in_final_sum * inv_in_elem;

            ei_new += nthreads;
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

            ei_new += nthreads;
        }

        __syncthreads();

        //======================================================================================================================================================
        //	TEMPLATE MASK CREATE
        //======================================================================================================================================================

        cent = sSize + d_common->tSize + 1;
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
        const int mask_location = tMask_col * tMask_rows + tMask_row;
        while (ei_new < tMask_elem) {

            d_unique[bx].d_tMask[ei_new] =
                (ei_new == mask_location) ? 1 : 0;

            ei_new += nthreads;
        }

        __syncthreads();

        //======================================================================================================================================================
        //	MASK CONVOLUTION
        //======================================================================================================================================================

        const int mask_conv_rows = d_common->mask_conv_rows;
        const int mask_conv_elem = d_common->mask_conv_elem;
        const int mask_cols = d_common->mask_cols;
        const int mask_rows = d_common->mask_rows;
        const int tMask_cols = d_common->tMask_cols;
        const int mask_conv_joffset = d_common->mask_conv_joffset;
        const int mask_conv_ioffset = d_common->mask_conv_ioffset;

        ei_new = tx;
        while (ei_new < mask_conv_elem) {

            int t = ei_new + 1;
            ic = t % mask_conv_rows;     // (1-n)
            jc = t / mask_conv_rows + 1; // (1-n)
            if (t % mask_conv_rows == 0) {
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
#pragma unroll 4
                for (ia = ia1; ia <= ia2; ia++) {
                    s += d_unique[bx]
                             .d_tMask[tMask_rows * (ja - 1) + ia - 1];
                }
            }

            d_unique[bx].d_mask_conv[ei_new] =
                d_unique[bx].d_in2_sqr_sub2[ei_new] * s;

            ei_new += nthreads;
        }

        __syncthreads();

        //======================================================================================================================================================
        //	MAXIMUM VALUE
        //======================================================================================================================================================

        //====================================================================================================
        //	INITIAL SEARCH
        //====================================================================================================

        const int mask_conv_cols = d_common->mask_conv_cols;

        ei_new = tx;
        while (ei_new < mask_conv_rows) {

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

            ei_new += nthreads;
        }

        __syncthreads();

        //====================================================================================================
        //	FINAL SEARCH
        //====================================================================================================

        if (tx == 0) {

            fin_max_val = 0.0f;
            fin_max_coo = 0;

#pragma unroll
            for (i = 0; i < mask_conv_rows; i++) {
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
            offset_row = largest_row - in_rows -
                         (sSize - d_common->tSize);
            offset_col = largest_col - in_cols -
                         (sSize - d_common->tSize);
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

        ei_new = tx;
        const int in_rows2 = d_common->in_rows;
        const int frame_rows2 = d_common->frame_rows;
        const int in_elem2 = d_common->in_elem;
        const int base_row2 = d_unique[bx].d_Row[d_unique[bx].point_no] - 25 - 1;
        const int base_col2 = d_unique[bx].d_Col[d_unique[bx].point_no] - 25 - 1;
        const float alpha = d_common->alpha;
        const float one_minus_alpha = 1.0f - alpha;

        while (ei_new < in_elem2) {

            int t = ei_new + 1;
            row = t % in_rows2 - 1;     // (0-n) row
            col = t / in_rows2 + 1 - 1; // (0-n) column
            if (t % in_rows2 == 0) {
                row = in_rows2 - 1;
                col = col - 1;
            }

            ori_row = base_row2 + row;
            ori_col = base_col2 + col;
            ori_pointer = ori_col * frame_rows2 + ori_row;

            d_in[ei_new] = alpha * d_in[ei_new] +
                           one_minus_alpha *
                               d_common_change->d_frame[ori_pointer];

            ei_new += nthreads;
        }
    }
}
