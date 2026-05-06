__global__ void kernel(params_common_change *d_common_change,
                       params_common *d_common, params_unique *d_unique) {

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

    // hardcoded sizes kept, but access assumed within bounds of in_cols/in_rows
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

    // keep in shared to reuse for convolution, rotation, etc.
    __shared__ float d_in_mod_temp[2601];

    int ori_pointer;
    int loc_pointer;

    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int stride = blockDim.x;

    int ei_new;

    //====================================================================================================================================
    // GENERATE TEMPLATE
    //====================================================================================================================================

    if (d_common_change->frame_no == 0) {

        d_in = &d_unique[bx].d_T[d_unique[bx].in_pointer];

        ei_new = tx;
        if (ei_new == 0) {
            pointer = d_unique[bx].point_no * d_common->no_frames +
                      d_common_change->frame_no;
            d_unique[bx].d_tRowLoc[pointer] =
                d_unique[bx].d_Row[d_unique[bx].point_no];
            d_unique[bx].d_tColLoc[pointer] =
                d_unique[bx].d_Col[d_unique[bx].point_no];
        }

        __syncthreads();

        ei_new = tx;
        while (ei_new < d_common->in_elem) {

            int idx = ei_new;
            int r = idx % d_common->in_rows;
            int c = idx / d_common->in_rows;

            row = r;
            col = c;

            ori_row = d_unique[bx].d_Row[d_unique[bx].point_no] - 25 + row - 1;
            ori_col = d_unique[bx].d_Col[d_unique[bx].point_no] - 25 + col - 1;
            ori_pointer = ori_col * d_common->frame_rows + ori_row;

            d_in[col * d_common->in_rows + row] =
                d_common_change->d_frame[ori_pointer];

            ei_new += stride;
        }
    }

    //====================================================================================================================================
    // PROCESS POINTS
    //====================================================================================================================================

    if (d_common_change->frame_no != 0) {

        in2_rowlow = d_unique[bx].d_Row[d_unique[bx].point_no] -
                     d_common->sSize;
        in2_collow =
            d_unique[bx].d_Col[d_unique[bx].point_no] - d_common->sSize;

        // SELECTION
        ei_new = tx;
        while (ei_new < d_common->in2_elem) {

            int idx = ei_new;
            int r = idx % d_common->in2_rows;
            int c = idx / d_common->in2_rows;

            row = r;
            col = c;

            ori_row = row + in2_rowlow - 1;
            ori_col = col + in2_collow - 1;
            d_unique[bx].d_in2[idx] =
                d_common_change
                    ->d_frame[ori_col * d_common->frame_rows + ori_row];

            ei_new += stride;
        }

        __syncthreads();

        // ROTATION: load template into shared memory (coalesced where possible)
        d_in = &d_unique[bx].d_T[d_unique[bx].in_pointer];

        ei_new = tx;
        while (ei_new < d_common->in_elem) {

            int idx = ei_new;
            int r = idx % d_common->in_rows;
            int c = idx / d_common->in_rows;

            row = r;
            col = c;

            rot_row = (d_common->in_rows - 1) - row;
            rot_col = (d_common->in_rows - 1) - col;
            d_in_mod_temp[idx] = d_in[rot_col * d_common->in_rows + rot_row];

            ei_new += stride;
        }

        __syncthreads();

        // ACTUAL CONVOLUTION
        ei_new = tx;
        while (ei_new < d_common->conv_elem) {

            int idx = ei_new;

            ic = (idx + 1) % d_common->conv_rows;
            jc = (idx + 1) / d_common->conv_rows + 1;
            if ((idx + 1) % d_common->conv_rows == 0) {
                ic = d_common->conv_rows;
                jc = jc - 1;
            }

            j = jc + d_common->joffset;
            jp1 = j + 1;
            ja1 = (d_common->in2_cols < jp1) ? (jp1 - d_common->in2_cols) : 1;
            ja2 = (d_common->in_cols < j) ? d_common->in_cols : j;

            i = ic + d_common->ioffset;
            ip1 = i + 1;

            ia1 = (d_common->in2_rows < ip1) ? (ip1 - d_common->in2_rows) : 1;
            ia2 = (d_common->in_rows < i) ? d_common->in_rows : i;

            float s_local = 0.0f;

#pragma unroll
            for (ja = ja1; ja <= ja2; ja++) {
                jb = jp1 - ja;
#pragma unroll
                for (ia = ia1; ia <= ia2; ia++) {
                    ib = ip1 - ia;
                    s_local += d_in_mod_temp[d_common->in_rows * (ja - 1) +
                                             ia - 1] *
                               d_unique[bx]
                                   .d_in2[d_common->in2_rows * (jb - 1) +
                                          ib - 1];
                }
            }

            d_unique[bx].d_conv[idx] = s_local;

            ei_new += stride;
        }

        __syncthreads();

        //====================================================================
        // CUMULATIVE SUM
        //====================================================================

        // PAD ARRAY, VERTICAL CUMULATIVE SUM (for in2)
        ei_new = tx;
        while (ei_new < d_common->in2_pad_cumv_elem) {

            int idx = ei_new;
            int r = idx % d_common->in2_pad_cumv_rows;
            int c = idx / d_common->in2_pad_cumv_rows;

            row = r;
            col = c;

            if (row > (d_common->in2_pad_add_rows - 1) &&
                row < (d_common->in2_pad_add_rows + d_common->in2_rows) &&
                col > (d_common->in2_pad_add_cols - 1) &&
                col < (d_common->in2_pad_add_cols + d_common->in2_cols)) {
                ori_row = row - d_common->in2_pad_add_rows;
                ori_col = col - d_common->in2_pad_add_cols;
                d_unique[bx].d_in2_pad_cumv[idx] =
                    d_unique[bx].d_in2[ori_col * d_common->in2_rows + ori_row];
            } else {
                d_unique[bx].d_in2_pad_cumv[idx] = 0.0f;
            }

            ei_new += stride;
        }

        __syncthreads();

        // VERTICAL CUMULATIVE SUM
        ei_new = tx;
        while (ei_new < d_common->in2_pad_cumv_cols) {

            pos_ori = ei_new * d_common->in2_pad_cumv_rows;
            sum = 0.0f;

            for (position = pos_ori;
                 position < pos_ori + d_common->in2_pad_cumv_rows;
                 position++) {
                float v = d_unique[bx].d_in2_pad_cumv[position] + sum;
                d_unique[bx].d_in2_pad_cumv[position] = v;
                sum = v;
            }

            ei_new += stride;
        }

        __syncthreads();

        // SELECTION
        ei_new = tx;
        while (ei_new < d_common->in2_pad_cumv_sel_elem) {

            int idx = ei_new;
            int r = idx % d_common->in2_pad_cumv_sel_rows;
            int c = idx / d_common->in2_pad_cumv_sel_rows;

            row = r;
            col = c;

            ori_row = row + d_common->in2_pad_cumv_sel_rowlow - 1;
            ori_col = col + d_common->in2_pad_cumv_sel_collow - 1;
            d_unique[bx].d_in2_pad_cumv_sel[idx] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * d_common->in2_pad_cumv_rows +
                                    ori_row];

            ei_new += stride;
        }

        __syncthreads();

        // SELECTION 2
        ei_new = tx;
        while (ei_new < d_common->in2_sub_cumh_elem) {

            int idx = ei_new;
            int r = idx % d_common->in2_sub_cumh_rows;
            int c = idx / d_common->in2_sub_cumh_rows;

            row = r;
            col = c;

            ori_row = row + d_common->in2_pad_cumv_sel2_rowlow - 1;
            ori_col = col + d_common->in2_pad_cumv_sel2_collow - 1;
            d_unique[bx].d_in2_sub_cumh[idx] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * d_common->in2_pad_cumv_rows +
                                    ori_row];

            ei_new += stride;
        }

        __syncthreads();

        // SUBTRACTION
        ei_new = tx;
        while (ei_new < d_common->in2_sub_cumh_elem) {

            d_unique[bx].d_in2_sub_cumh[ei_new] =
                d_unique[bx].d_in2_pad_cumv_sel[ei_new] -
                d_unique[bx].d_in2_sub_cumh[ei_new];

            ei_new += stride;
        }

        __syncthreads();

        // HORIZONTAL CUMULATIVE SUM
        ei_new = tx;
        while (ei_new < d_common->in2_sub_cumh_rows) {

            pos_ori = ei_new;
            sum = 0.0f;

            for (position = pos_ori;
                 position < pos_ori + d_common->in2_sub_cumh_elem;
                 position += d_common->in2_sub_cumh_rows) {
                float v = d_unique[bx].d_in2_sub_cumh[position] + sum;
                d_unique[bx].d_in2_sub_cumh[position] = v;
                sum = v;
            }

            ei_new += stride;
        }

        __syncthreads();

        // SELECTION
        ei_new = tx;
        while (ei_new < d_common->in2_sub_cumh_sel_elem) {

            int idx = ei_new;
            int r = idx % d_common->in2_sub_cumh_sel_rows;
            int c = idx / d_common->in2_sub_cumh_sel_rows;

            row = r;
            col = c;

            ori_row = row + d_common->in2_sub_cumh_sel_rowlow - 1;
            ori_col = col + d_common->in2_sub_cumh_sel_collow - 1;
            d_unique[bx].d_in2_sub_cumh_sel[idx] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * d_common->in2_sub_cumh_rows +
                                    ori_row];

            ei_new += stride;
        }

        __syncthreads();

        // SELECTION 2
        ei_new = tx;
        while (ei_new < d_common->in2_sub2_elem) {

            int idx = ei_new;
            int r = idx % d_common->in2_sub2_rows;
            int c = idx / d_common->in2_sub2_rows;

            row = r;
            col = c;

            ori_row = row + d_common->in2_sub_cumh_sel2_rowlow - 1;
            ori_col = col + d_common->in2_sub_cumh_sel2_collow - 1;
            d_unique[bx].d_in2_sub2[idx] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * d_common->in2_sub_cumh_rows +
                                    ori_row];

            ei_new += stride;
        }

        __syncthreads();

        // SUBTRACTION
        ei_new = tx;
        while (ei_new < d_common->in2_sub2_elem) {

            d_unique[bx].d_in2_sub2[ei_new] =
                d_unique[bx].d_in2_sub_cumh_sel[ei_new] -
                d_unique[bx].d_in2_sub2[ei_new];

            ei_new += stride;
        }

        __syncthreads();

        //====================================================================
        // CUMULATIVE SUM 2 (squared)
        //====================================================================

        // MULTIPLICATION (in2^2)
        ei_new = tx;
        while (ei_new < d_common->in2_sqr_elem) {

            temp = d_unique[bx].d_in2[ei_new];
            d_unique[bx].d_in2_sqr[ei_new] = temp * temp;

            ei_new += stride;
        }

        __syncthreads();

        // PAD ARRAY (squared), VERTICAL CUMULATIVE SUM
        ei_new = tx;
        while (ei_new < d_common->in2_pad_cumv_elem) {

            int idx = ei_new;
            int r = idx % d_common->in2_pad_cumv_rows;
            int c = idx / d_common->in2_pad_cumv_rows;

            row = r;
            col = c;

            if (row > (d_common->in2_pad_add_rows - 1) &&
                row <
                    (d_common->in2_pad_add_rows + d_common->in2_sqr_rows) &&
                col > (d_common->in2_pad_add_cols - 1) &&
                col < (d_common->in2_pad_add_cols + d_common->in2_sqr_cols)) {
                ori_row = row - d_common->in2_pad_add_rows;
                ori_col = col - d_common->in2_pad_add_cols;
                d_unique[bx].d_in2_pad_cumv[idx] =
                    d_unique[bx]
                        .d_in2_sqr[ori_col * d_common->in2_sqr_rows + ori_row];
            } else {
                d_unique[bx].d_in2_pad_cumv[idx] = 0.0f;
            }

            ei_new += stride;
        }

        __syncthreads();

        // VERTICAL CUMULATIVE SUM (squared)
        ei_new = tx;
        while (ei_new < d_common->in2_pad_cumv_cols) {

            pos_ori = ei_new * d_common->in2_pad_cumv_rows;
            sum = 0.0f;

            for (position = pos_ori;
                 position < pos_ori + d_common->in2_pad_cumv_rows;
                 position++) {
                float v = d_unique[bx].d_in2_pad_cumv[position] + sum;
                d_unique[bx].d_in2_pad_cumv[position] = v;
                sum = v;
            }

            ei_new += stride;
        }

        __syncthreads();

        // SELECTION
        ei_new = tx;
        while (ei_new < d_common->in2_pad_cumv_sel_elem) {

            int idx = ei_new;
            int r = idx % d_common->in2_pad_cumv_sel_rows;
            int c = idx / d_common->in2_pad_cumv_sel_rows;

            row = r;
            col = c;

            ori_row = row + d_common->in2_pad_cumv_sel_rowlow - 1;
            ori_col = col + d_common->in2_pad_cumv_sel_collow - 1;
            d_unique[bx].d_in2_pad_cumv_sel[idx] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * d_common->in2_pad_cumv_rows +
                                    ori_row];

            ei_new += stride;
        }

        __syncthreads();

        // SELECTION 2
        ei_new = tx;
        while (ei_new < d_common->in2_sub_cumh_elem) {

            int idx = ei_new;
            int r = idx % d_common->in2_sub_cumh_rows;
            int c = idx / d_common->in2_sub_cumh_rows;

            row = r;
            col = c;

            ori_row = row + d_common->in2_pad_cumv_sel2_rowlow - 1;
            ori_col = col + d_common->in2_pad_cumv_sel2_collow - 1;
            d_unique[bx].d_in2_sub_cumh[idx] =
                d_unique[bx]
                    .d_in2_pad_cumv[ori_col * d_common->in2_pad_cumv_rows +
                                    ori_row];

            ei_new += stride;
        }

        __syncthreads();

        // SUBTRACTION
        ei_new = tx;
        while (ei_new < d_common->in2_sub_cumh_elem) {

            d_unique[bx].d_in2_sub_cumh[ei_new] =
                d_unique[bx].d_in2_pad_cumv_sel[ei_new] -
                d_unique[bx].d_in2_sub_cumh[ei_new];

            ei_new += stride;
        }

        // HORIZONTAL CUMULATIVE SUM (squared)
        ei_new = tx;
        while (ei_new < d_common->in2_sub_cumh_rows) {

            pos_ori = ei_new;
            sum = 0.0f;

            for (position = pos_ori;
                 position < pos_ori + d_common->in2_sub_cumh_elem;
                 position += d_common->in2_sub_cumh_rows) {
                float v = d_unique[bx].d_in2_sub_cumh[position] + sum;
                d_unique[bx].d_in2_sub_cumh[position] = v;
                sum = v;
            }

            ei_new += stride;
        }

        __syncthreads();

        // SELECTION
        ei_new = tx;
        while (ei_new < d_common->in2_sub_cumh_sel_elem) {

            int idx = ei_new;
            int r = idx % d_common->in2_sub_cumh_sel_rows;
            int c = idx / d_common->in2_sub_cumh_sel_rows;

            row = r;
            col = c;

            ori_row = row + d_common->in2_sub_cumh_sel_rowlow - 1;
            ori_col = col + d_common->in2_sub_cumh_sel_collow - 1;
            d_unique[bx].d_in2_sub_cumh_sel[idx] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * d_common->in2_sub_cumh_rows +
                                    ori_row];

            ei_new += stride;
        }

        __syncthreads();

        // SELECTION 2
        ei_new = tx;
        while (ei_new < d_common->in2_sub2_elem) {

            int idx = ei_new;
            int r = idx % d_common->in2_sub2_rows;
            int c = idx / d_common->in2_sub2_rows;

            row = r;
            col = c;

            ori_row = row + d_common->in2_sub_cumh_sel2_rowlow - 1;
            ori_col = col + d_common->in2_sub_cumh_sel2_collow - 1;
            d_unique[bx].d_in2_sqr_sub2[idx] =
                d_unique[bx]
                    .d_in2_sub_cumh[ori_col * d_common->in2_sub_cumh_rows +
                                    ori_row];

            ei_new += stride;
        }

        __syncthreads();

        // SUBTRACTION
        ei_new = tx;
        while (ei_new < d_common->in2_sub2_elem) {

            d_unique[bx].d_in2_sqr_sub2[ei_new] =
                d_unique[bx].d_in2_sub_cumh_sel[ei_new] -
                d_unique[bx].d_in2_sqr_sub2[ei_new];

            ei_new += stride;
        }

        __syncthreads();

        //====================================================================
        // FINAL: denominator A
        //====================================================================

        ei_new = tx;
        while (ei_new < d_common->in2_sub2_elem) {

            temp = d_unique[bx].d_in2_sub2[ei_new];
            temp2 = d_unique[bx].d_in2_sqr_sub2[ei_new] -
                    (temp * temp / d_common->in_elem);
            if (temp2 < 0.0f) {
                temp2 = 0.0f;
            }
            d_unique[bx].d_in2_sqr_sub2[ei_new] = sqrtf(temp2);

            ei_new += stride;
        }

        __syncthreads();

        // MULTIPLICATION (template^2)
        ei_new = tx;
        while (ei_new < d_common->in_sqr_elem) {

            temp = d_in[ei_new];
            d_unique[bx].d_in_sqr[ei_new] = temp * temp;

            ei_new += stride;
        }

        __syncthreads();

        // IN SUM (template)
        ei_new = tx;
        while (ei_new < d_common->in_cols) {

            sum = 0.0f;
#pragma unroll
            for (i = 0; i < d_common->in_rows; i++) {
                sum += d_in[ei_new * d_common->in_rows + i];
            }
            in_partial_sum[ei_new] = sum;

            ei_new += stride;
        }

        __syncthreads();

        // IN_SQR SUM
        ei_new = tx;
        while (ei_new < d_common->in_sqr_rows) {

            sum = 0.0f;
#pragma unroll
            for (i = 0; i < d_common->in_sqr_cols; i++) {
                sum += d_unique[bx].d_in_sqr[ei_new +
                                             d_common->in_sqr_rows * i];
            }
            in_sqr_partial_sum[ei_new] = sum;

            ei_new += stride;
        }

        __syncthreads();

        // FINAL SUMMATION (use two threads, then broadcast)
        if (tx == 0) {
            float acc = 0.0f;
            for (i = 0; i < d_common->in_cols; i++) {
                acc += in_partial_sum[i];
            }
            in_final_sum = acc;
        } else if (tx == 1) {
            float acc2 = 0.0f;
            for (i = 0; i < d_common->in_sqr_cols; i++) {
                acc2 += in_sqr_partial_sum[i];
            }
            in_sqr_final_sum = acc2;
        }

        __syncthreads();

        // DENOMINATOR T
        if (tx == 0) {

            mean = in_final_sum / d_common->in_elem;
            mean_sqr = mean * mean;
            variance = (in_sqr_final_sum / d_common->in_elem) - mean_sqr;
            variance = variance > 0.0f ? variance : 0.0f;
            deviation = sqrtf(variance);

            denomT = sqrtf(float(d_common->in_elem - 1)) * deviation;
        }

        __syncthreads();

        // DENOMINATOR multiply
        ei_new = tx;
        while (ei_new < d_common->in2_sub2_elem) {

            d_unique[bx].d_in2_sqr_sub2[ei_new] =
                d_unique[bx].d_in2_sqr_sub2[ei_new] * denomT;

            ei_new += stride;
        }

        __syncthreads();

        // NUMERATOR
        ei_new = tx;
        while (ei_new < d_common->conv_elem) {

            d_unique[bx].d_conv[ei_new] =
                d_unique[bx].d_conv[ei_new] -
                d_unique[bx].d_in2_sub2[ei_new] * in_final_sum /
                    d_common->in_elem;

            ei_new += stride;
        }

        __syncthreads();

        // CORRELATION
        ei_new = tx;
        while (ei_new < d_common->in2_sub2_elem) {

            d_unique[bx].d_in2_sqr_sub2[ei_new] =
                d_unique[bx].d_conv[ei_new] /
                d_unique[bx].d_in2_sqr_sub2[ei_new];

            ei_new += stride;
        }

        __syncthreads();

        // TEMPLATE MASK CREATE
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

        ei_new = tx;
        while (ei_new < d_common->tMask_elem) {

            location = tMask_col * d_common->tMask_rows + tMask_row;

            d_unique[bx].d_tMask[ei_new] =
                (ei_new == location) ? 1.0f : 0.0f;

            ei_new += stride;
        }

        __syncthreads();

        // MASK CONVOLUTION
        ei_new = tx;
        while (ei_new < d_common->mask_conv_elem) {

            int idx = ei_new;

            ic = (idx + 1) % d_common->mask_conv_rows;
            jc = (idx + 1) / d_common->mask_conv_rows + 1;
            if ((idx + 1) % d_common->mask_conv_rows == 0) {
                ic = d_common->mask_conv_rows;
                jc = jc - 1;
            }

            j = jc + d_common->mask_conv_joffset;
            jp1 = j + 1;
            ja1 = (d_common->mask_cols < jp1) ? (jp1 - d_common->mask_cols) : 1;
            ja2 = (d_common->tMask_cols < j) ? d_common->tMask_cols : j;

            i = ic + d_common->mask_conv_ioffset;
            ip1 = i + 1;

            ia1 = (d_common->mask_rows < ip1) ? (ip1 - d_common->mask_rows) : 1;
            ia2 = (d_common->tMask_rows < i) ? d_common->tMask_rows : i;

            s = 0.0f;

#pragma unroll
            for (ja = ja1; ja <= ja2; ja++) {
                // jb is unused in mask conv multiplication by 1 but kept
                jb = jp1 - ja;
#pragma unroll
                for (ia = ia1; ia <= ia2; ia++) {
                    ib = ip1 - ia;
                    (void)jb;
                    (void)ib;
                    s += d_unique[bx]
                             .d_tMask[d_common->tMask_rows * (ja - 1) +
                                      ia - 1];
                }
            }

            d_unique[bx].d_mask_conv[idx] =
                d_unique[bx].d_in2_sqr_sub2[idx] * s;

            ei_new += stride;
        }

        __syncthreads();

        // MAXIMUM VALUE
        // INITIAL SEARCH
        ei_new = tx;
        while (ei_new < d_common->mask_conv_rows) {

            largest_value = 0.0f;
            largest_coordinate = 0;

#pragma unroll
            for (i = 0; i < d_common->mask_conv_cols; i++) {
                largest_coordinate_current =
                    ei_new * d_common->mask_conv_rows + i;
                largest_value_current =
                    fabsf(d_unique[bx]
                              .d_mask_conv[largest_coordinate_current]);
                if (largest_value_current > largest_value) {
                    largest_coordinate = largest_coordinate_current;
                    largest_value = largest_value_current;
                }
            }
            par_max_coo[ei_new] = largest_coordinate;
            par_max_val[ei_new] = largest_value;

            ei_new += stride;
        }

        __syncthreads();

        // FINAL SEARCH
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
                (fin_max_coo + 1) % d_common->mask_conv_rows - 1;
            largest_col =
                (fin_max_coo + 1) / d_common->mask_conv_rows;
            if ((fin_max_coo + 1) % d_common->mask_conv_rows == 0) {
                largest_row = d_common->mask_conv_rows - 1;
                largest_col = largest_col - 1;
            }

            largest_row = largest_row + 1;
            largest_col = largest_col + 1;
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

        __syncthreads();
    }

    //====================================================================================================================================
    // COORDINATE AND TEMPLATE UPDATE
    //====================================================================================================================================

    if (d_common_change->frame_no != 0 &&
        (d_common_change->frame_no) % 10 == 0) {

        d_in = &d_unique[bx].d_T[d_unique[bx].in_pointer];

        loc_pointer = d_unique[bx].point_no * d_common->no_frames +
                      d_common_change->frame_no;
        d_unique[bx].d_Row[d_unique[bx].point_no] =
            d_unique[bx].d_tRowLoc[loc_pointer];
        d_unique[bx].d_Col[d_unique[bx].point_no] =
            d_unique[bx].d_tColLoc[loc_pointer];

        __syncthreads();

        ei_new = tx;
        while (ei_new < d_common->in_elem) {

            int idx = ei_new;
            int r = idx % d_common->in_rows;
            int c = idx / d_common->in_rows;

            row = r;
            col = c;

            ori_row = d_unique[bx].d_Row[d_unique[bx].point_no] - 25 +
                      row - 1;
            ori_col = d_unique[bx].d_Col[d_unique[bx].point_no] - 25 +
                      col - 1;
            ori_pointer = ori_col * d_common->frame_rows + ori_row;

            float old_val = d_in[idx];
            float new_val = d_common_change->d_frame[ori_pointer];
            d_in[idx] = d_common->alpha * old_val +
                        (1.0f - d_common->alpha) * new_val;

            ei_new += stride;
        }
    }
}
