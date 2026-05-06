<<<CODE>>>
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
    float largest_value_current = 0;
    float largest_value = 0;
    int largest_coordinate_current = 0;
    int largest_coordinate = 0;
    float fin_max_val = 0;
    int fin_max_coo = 0;
    int largest_row;
    int largest_col;
    int offset_row;
    int offset_col;
    __shared__ float in_partial_sum[51];
    __shared__ float in_sqr_partial_sum[51];
    __shared__ float in_final_sum;
    __shared__ float in_sqr_final_sum;
    float mean;
    float mean_sqr;
    float variance;
    float deviation;
    __shared__ float denomT;
    __shared__ float par_max_val[131];
    __shared__ int par_max_coo[131];
    int pointer;
    __shared__ float d_in_mod_temp[2601];
    int ori_pointer;
    int loc_pointer;

    //======================================================================================================================================================
    //	THREAD PARAMETERS
    //======================================================================================================================================================

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ei_new;

    // Cache frequently accessed values
    const int frame_no = d_common_change->frame_no;
    const int point_no = d_unique[bx].point_no;
    const int in_elem = d_common->in_elem;
    const int in_rows = d_common->in_rows;
    const int in_cols = d_common->in_cols;
    const int frame_rows = d_common->frame_rows;
    const int no_frames = d_common->no_frames;

    //===============================================================================================================================================================================================================
    //===============================================================================================================================================================================================================
    //	GENERATE TEMPLATE
    //===============================================================================================================================================================================================================
    //===============================================================================================================================================================================================================

    if (frame_no == 0) {

        //======================================================================================================================================================
        // GET POINTER TO TEMPLATE FOR THE POINT
        //======================================================================================================================================================

        d_in = &d_unique[bx].d_T[d_unique[bx].in_pointer];

        //======================================================================================================================================================
        //	UPDATE ROW LOC AND COL LOC
        //======================================================================================================================================================

        if (tx == 0) {
            pointer = point_no * no_frames + frame_no;
            d_unique[bx].d_tRowLoc[pointer] = d_unique[bx].d_Row[point_no];
            d_unique[bx].d_tColLoc[pointer] = d_unique[bx].d_Col[point_no];
        }

        //======================================================================================================================================================
        //	CREATE TEMPLATES
        //======================================================================================================================================================

        const int row_base = d_unique[bx].d_Row[point_no] - 26;
        const int col_base = d_unique[bx].d_Col[point_no] - 26;

        for (ei_new = tx; ei_new < in_elem; ei_new += NUMBER_THREADS) {
            row = (ei_new + 1) % in_rows - 1;
            col = (ei_new + 1) / in_rows + 1 - 1;
            if ((ei_new + 1) % in_rows == 0) {
                row = in_rows - 1;
                col = col - 1;
            }

            ori_row = row_base + row;
            ori_col = col_base + col;
            ori_pointer = ori_col * frame_rows + ori_row;

            d_in[col * in_rows + row] = d_common_change->d_frame[ori_pointer];
        }
    }

    //===============================================================================================================================================================================================================
    //===============================================================================================================================================================================================================
    //	PROCESS POINTS
    //===============================================================================================================================================================================================================
    //===============================================================================================================================================================================================================

    if (frame_no != 0) {

        //======================================================================================================================================================
        //	SELECTION
        //======================================================================================================================================================

        const int sSize = d_common->sSize;
        in2_rowlow = d_unique[bx].d_Row[point_no] - sSize;
        in2_collow = d_unique[bx].d_Col[point_no] - sSize;

        const int in2_elem = d_common->in2_elem;
        const int in2_rows = d_common->in2_rows;
        const int in2_cols = d_common->in2_cols;

        for (ei_new = tx; ei_new < in2_elem; ei_new += NUMBER_THREADS) {
            row = (ei_new + 1) % in2_rows - 1;
            col = (ei_new + 1) / in2_rows + 1 - 1;
            if ((ei_new + 1) % in2_rows == 0) {
                row = in2_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_rowlow - 1;
            ori_col = col + in2_collow - 1;
            d_unique[bx].d_in2[ei_new] =
                d_common_change->d_frame[ori_col * frame_rows + ori_row];
        }

        __syncthreads();

        //======================================================================================================================================================
        //	CONVOLUTION
        //======================================================================================================================================================

        //====================================================================================================
        //	ROTATION
        //====================================================================================================

        d_in = &d_unique[bx].d_T[d_unique[bx].in_pointer];

        const int in_rows_m1 = in_rows - 1;
        for (ei_new = tx; ei_new < in_elem; ei_new += NUMBER_THREADS) {
            row = (ei_new + 1) % in_rows - 1;
            col = (ei_new + 1) / in_rows + 1 - 1;
            if ((ei_new + 1) % in_rows == 0) {
                row = in_rows - 1;
                col = col - 1;
            }

            rot_row = in_rows_m1 - row;
            rot_col = in_rows_m1 - col;
            d_in_mod_temp[ei_new] = d_in[rot_col * in_rows + rot_row];
        }

        __syncthreads();

        //====================================================================================================
        //	ACTUAL CONVOLUTION
        //====================================================================================================

        const int conv_elem = d_common->conv_elem;
        const int conv_rows = d_common->conv_rows;
        const int joffset = d_common->joffset;
        const int ioffset = d_common->ioffset;

        for (ei_new = tx; ei_new < conv_elem; ei_new += NUMBER_THREADS) {
            ic = (ei_new + 1) % conv_rows;
            jc = (ei_new + 1) / conv_rows + 1;
            if ((ei_new + 1) % conv_rows == 0) {
                ic = conv_rows;
                jc = jc - 1;
            }

            j = jc + joffset;
            jp1 = j + 1;
            ja1 = (in2_cols < jp1) ? (jp1 - in2_cols) : 1;
            ja2 = (in_cols < j) ? in_cols : j;

            i = ic + ioffset;
            ip1 = i + 1;
            ia1 = (in2_rows < ip1) ? (ip1 - in2_rows) : 1;
            ia2 = (in_rows < i) ? in_rows : i;

            s = 0;
            for (ja = ja1; ja <= ja2; ja++) {
                jb = jp1 - ja;
                for (ia = ia1; ia <= ia2; ia++) {
                    ib = ip1 - ia;
                    s += d_in_mod_temp[in_rows * (ja - 1) + ia - 1] *
                         d_unique[bx].d_in2[in2_rows * (jb - 1) + ib - 1];
                }
            }

            d_unique[bx].d_conv[ei_new] = s;
        }

        __syncthreads();

        //======================================================================================================================================================
        //	CUMULATIVE SUM
        //======================================================================================================================================================

        //====================================================================================================
        //	PAD ARRAY, VERTICAL CUMULATIVE SUM
        //====================================================================================================

        const int in2_pad_cumv_elem = d_common->in2_pad_cumv_elem;
        const int in2_pad_cumv_rows = d_common->in2_pad_cumv_rows;
        const int in2_pad_add_rows = d_common->in2_pad_add_rows;
        const int in2_pad_add_cols = d_common->in2_pad_add_cols;

        for (ei_new = tx; ei_new < in2_pad_cumv_elem; ei_new += NUMBER_THREADS) {
            row = (ei_new + 1) % in2_pad_cumv_rows - 1;
            col = (ei_new + 1) / in2_pad_cumv_rows + 1 - 1;
            if ((ei_new + 1) % in2_pad_cumv_rows == 0) {
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
                d_unique[bx].d_in2_pad_cumv[ei_new] = 0;
            }
        }

        __syncthreads();

        const int in2_pad_cumv_cols = d_common->in2_pad_cumv_cols;
        for (ei_new = tx; ei_new < in2_pad_cumv_cols; ei_new += NUMBER_THREADS) {
            pos_ori = ei_new * in2_pad_cumv_rows;
            sum = 0;
            for (position = pos_ori; position < pos_ori + in2_pad_cumv_rows; position++) {
                d_unique[bx].d_in2_pad_cumv[position] += sum;
                sum = d_unique[bx].d_in2_pad_cumv[position];
            }
        }

        __syncthreads();

        //====================================================================================================
        //	SELECTION
        //====================================================================================================

        const int in2_pad_cumv_sel_elem = d_common->in2_pad_cumv_sel_elem;
        const int in2_pad_cumv_sel_rows = d_common->in2_pad_cumv_sel_rows;
        const int in2_pad_cumv_sel_rowlow = d_common->in2_pad_cumv_sel_rowlow;
        const int in2_pad_cumv_sel_collow = d_common->in2_pad_cumv_sel_collow;

        for (ei_new = tx; ei_new < in2_pad_cumv_sel_elem; ei_new += NUMBER_THREADS) {
            row = (ei_new + 1) % in2_pad_cumv_sel_rows - 1;
            col = (ei_new + 1) / in2_pad_cumv_sel_rows + 1 - 1;
            if ((ei_new + 1) % in2_pad_cumv_sel_rows == 0) {
                row = in2_pad_cumv_sel_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_pad_cumv_sel_rowlow - 1;
            ori_col = col + in2_pad_cumv_sel_collow - 1;
            d_unique[bx].d_in2_pad_cumv_sel[ei_new] =
                d_unique[bx].d_in2_pad_cumv[ori_col * in2_pad_cumv_rows + ori_row];
        }

        __syncthreads();

        //====================================================================================================
        //	SELECTION 2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
        //====================================================================================================

        const int in2_sub_cumh_elem = d_common->in2_sub_cumh_elem;
        const int in2_sub_cumh_rows = d_common->in2_sub_cumh_rows;
        const int in2_pad_cumv_sel2_rowlow = d_common->in2_pad_cumv_sel2_rowlow;
        const int in2_pad_cumv_sel2_collow = d_common->in2_pad_cumv_sel2_collow;

        for (ei_new = tx; ei_new < in2_sub_cumh_elem; ei_new += NUMBER_THREADS) {
            row = (ei_new + 1) % in2_sub_cumh_rows - 1;
            col = (ei_new + 1) / in2_sub_cumh_rows + 1 - 1;
            if ((ei_new + 1) % in2_sub_cumh_rows == 0) {
                row = in2_sub_cumh_rows - 1;
                col = col - 1;
            }

            ori_row = row + in2_pad_cumv_sel2_rowlow - 1;
            ori_col = col + in2_pad_cumv_sel2_collow - 1;
            d_unique[bx].d_in2_sub_cumh[ei_new] =
                d_unique[bx].d_in2_pad_cumv[ori_col * in2_pad_cumv_rows + ori_row];
        }

        __syncthreads();

        for (ei_new = tx; ei_new < in2_sub_cumh_elem; ei_new += NUMBER_THREADS) {
            d_unique[bx].d_in2_sub_cumh[ei_new] =
                d_unique[bx].d_in2_pad_cumv_sel[ei_new] -
                d_unique[bx].d_in2_sub_cumh[ei_new];
        }

        __syncthreads();

        for (ei_new = tx; ei_new < in2_sub_cumh_rows; ei_new += NUMBER_THREADS) {
            pos_ori = ei_new;
            sum = 0;
            for (position = pos_ori; position < pos_ori + in2_sub_cumh_elem;
