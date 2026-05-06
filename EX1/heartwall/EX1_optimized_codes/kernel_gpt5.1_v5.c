#include <time.h>
#include <math.h>

static double heartwall_kernel_time = 0.0;

void reset_heartwall_kernel_time(void) { heartwall_kernel_time = 0.0; }

double get_heartwall_kernel_time(void) { return heartwall_kernel_time; }

void kernel(public_struct public, private_struct private) {

    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    int ei_new;
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
    fp s;
    int i;
    int j;
    int row;
    int col;
    int ori_row;
    int ori_col;
    int position;
    fp sum;
    int pos_ori;
    fp temp;
    fp temp2;
    int location;
    int cent;
    int tMask_row;
    int tMask_col;
    fp fin_max_val = 0;
    int fin_max_coo = 0;
    int largest_row;
    int largest_col;
    int offset_row;
    int offset_col;
    fp in_final_sum;
    fp in_sqr_final_sum;
    fp mean;
    fp mean_sqr;
    fp variance;
    fp deviation;
    fp denomT;
    int pointer;
    int ori_pointer;
    int loc_pointer;
    int ei_mod;

    //======================================================================================================================================================
    //	GENERATE TEMPLATE
    //======================================================================================================================================================

    // generate templates based on the first frame only
    if (public.frame_no == 0) {

        // update temporary row/col coordinates
        pointer = private.point_no * public.frames + public.frame_no;
        private.d_tRowLoc[pointer] = private.d_Row[private.point_no];
        private.d_tColLoc[pointer] = private.d_Col[private.point_no];

        // pointers to: current frame, template for current point
        d_in = &private.d_T[private.in_pointer];

        // update template, limit the number of working threads to the size of
        // template
        const int frame_rows = public.frame_rows;
        const int in_mod_rows = public.in_mod_rows;
        const int in_mod_cols = public.in_mod_cols;
        const int base_row = private.d_Row[private.point_no] - 26;
        const int base_col = private.d_Col[private.point_no] - 26;

        for (col = 0; col < in_mod_cols; col++) {
            const int ori_col_base = (base_col + col) * frame_rows;
            const int dst_col_offset = col * in_mod_rows;
            for (row = 0; row < in_mod_rows; row++) {
                ori_row = base_row + row;
                ori_pointer = ori_col_base + ori_row;
                d_in[dst_col_offset + row] = public.d_frame[ori_pointer];
            }
        }
    }

    //======================================================================================================================================================
    //	PROCESS POINTS
    //======================================================================================================================================================

    // process points in all frames except for the first one
    if (public.frame_no != 0) {

        const int frame_rows = public.frame_rows;
        const int in2_rows = public.in2_rows;
        const int in2_cols = public.in2_cols;
        const int in2_rows_stride = public.in2_rows;
        const int in2_pad_rows = public.in2_pad_rows;
        const int in2_pad_cols = public.in2_pad_cols;
        const int in2_pad_rows_stride = public.in2_pad_rows;
        const int in2_sub_rows = public.in2_sub_rows;
        const int in2_sub_cols = public.in2_sub_cols;
        const int in2_sub_rows_stride = public.in2_sub_rows;
        const int conv_rows = public.conv_rows;
        const int conv_cols = public.conv_cols;
        const int in_mod_rows = public.in_mod_rows;
        const int in_mod_cols = public.in_mod_cols;
        const int in_mod_elem = public.in_mod_elem;

        in2_rowlow = private.d_Row[private.point_no] - public.sSize;
        in2_collow = private.d_Col[private.point_no] - public.sSize;

        // work: copy region to d_in2 and d_in2_sqr
        for (col = 0; col < in2_cols; col++) {
            const int ori_col_base = (in2_collow + col - 1) * frame_rows;
            const int dst_col_offset = col * in2_rows_stride;
            for (row = 0; row < in2_rows; row++) {
                ori_row = row + in2_rowlow - 1;
                temp = public.d_frame[ori_col_base + ori_row];
                private.d_in2[dst_col_offset + row] = temp;
                private.d_in2_sqr[dst_col_offset + row] = temp * temp;
            }
        }

        // variables
        d_in = &private.d_T[private.in_pointer];

        // work: rotate template into d_in_mod and compute squared template
        {
            const int in_mod_rows_m1 = in_mod_rows - 1;
            const int in_mod_cols_m1 = in_mod_cols - 1;
            for (col = 0; col < in_mod_cols; col++) {
                const int dst_col_offset = col * in_mod_rows;
                for (row = 0; row < in_mod_rows; row++) {
                    rot_row = in_mod_rows_m1 - row;
                    rot_col = in_mod_cols_m1 - col;
                    pointer = rot_col * in_mod_rows + rot_row;
                    temp = d_in[pointer];
                    private.d_in_mod[dst_col_offset + row] = temp;
                    private.d_in_sqr[pointer] = temp * temp;
                }
            }
        }

        in_final_sum = 0;
        for (i = 0; i < in_mod_elem; i++) {
            in_final_sum += d_in[i];
        }

        in_sqr_final_sum = 0;
        for (i = 0; i < in_mod_elem; i++) {
            in_sqr_final_sum += private.d_in_sqr[i];
        }

        mean = in_final_sum / (fp)in_mod_elem;
        mean_sqr = mean * mean;
        variance = (in_sqr_final_sum / (fp)in_mod_elem) - mean_sqr;
        deviation = sqrt(variance);
        denomT = sqrt((fp)(in_mod_elem - 1)) * deviation;

        // work: convolution
        for (col = 1; col <= conv_cols; col++) {

            j = col + public.joffset;
            jp1 = j + 1;
            ja1 = (public.in2_cols < jp1) ? (jp1 - public.in2_cols) : 1;
            ja2 = (in_mod_cols < j) ? in_mod_cols : j;

            const int conv_col_offset = (col - 1) * conv_rows;

            for (row = 1; row <= conv_rows; row++) {

                i = row + public.ioffset;
                ip1 = i + 1;

                ia1 = (in2_rows < ip1) ? (ip1 - in2_rows) : 1;
                ia2 = (in_mod_rows < i) ? in_mod_rows : i;

                s = 0;

                for (ja = ja1; ja <= ja2; ja++) {
                    jb = jp1 - ja;
                    const int in_mod_col_base =
                        (ja - 1) * in_mod_rows - 1; // used with ia (1-based)
                    const int in2_col_base =
                        (jb - 1) * in2_rows - 1; // used with ib (1-based)
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        s += private.d_in_mod[in_mod_col_base + ia] *
                             private.d_in2[in2_col_base + ib];
                    }
                }

                private.d_conv[conv_col_offset + (row - 1)] = s;
            }
        }

        // work: build padded in2 (non-squared)
        {
            const int add_rows = public.in2_pad_add_rows;
            const int add_cols = public.in2_pad_add_cols;
            for (col = 0; col < in2_pad_cols; col++) {
                const int pad_col_offset = col * in2_pad_rows_stride;
                const int col_gt = (col > (add_cols - 1));
                const int col_lt =
                    (col < (add_cols + in2_cols)); // both used in condition
                for (row = 0; row < in2_pad_rows; row++) {
                    if (row > (add_rows - 1) &&
                        row < (add_rows + in2_rows) && col_gt && col_lt) {
                        ori_row = row - add_rows;
                        ori_col = col - add_cols;
                        private.d_in2_pad[pad_col_offset + row] =
                            private.d_in2[ori_col * in2_rows_stride + ori_row];
                    } else {
                        private.d_in2_pad[pad_col_offset + row] = 0;
                    }
                }
            }
        }

        // cumulative sum over rows (column-wise)
        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {
            pos_ori = ei_new * in2_pad_rows_stride;
            sum = 0;
            const int end_pos = pos_ori + in2_pad_rows;
            for (position = pos_ori; position < end_pos; position++) {
                sum = private.d_in2_pad[position] += sum;
            }
        }

        // work: build in2_sub
        {
            const int sel_rowlow = public.in2_pad_cumv_sel_rowlow - 1;
            const int sel_collow = public.in2_pad_cumv_sel_collow - 1;
            const int sel2_rowlow = public.in2_pad_cumv_sel2_rowlow - 1;
            const int sel2_collow = public.in2_pad_cumv_sel2_collow - 1;

            for (col = 0; col < in2_sub_cols; col++) {
                const int col_sel = (col + sel_collow);
                const int col_sel2 = (col + sel2_collow);
                const int pad_col_sel_base =
                    col_sel * in2_pad_rows_stride + sel_rowlow;
                const int pad_col_sel2_base =
                    col_sel2 * in2_pad_rows_stride + sel2_rowlow;
                const int dst_col_offset = col * in2_sub_rows_stride;
                for (row = 0; row < in2_sub_rows; row++) {
                    temp = private.d_in2_pad[pad_col_sel_base + row];
                    temp2 = private.d_in2_pad[pad_col_sel2_base + row];
                    private.d_in2_sub[dst_col_offset + row] = temp - temp2;
                }
            }
        }

        // cumulative sum over columns (row-wise) for in2_sub
        {
            const int in2_sub_elem = public.in2_sub_elem;
            const int step = in2_sub_rows;
            const int limit = in2_sub_rows + in2_sub_elem;
            for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {
                pos_ori = ei_new;
                sum = 0;
                for (position = pos_ori; position < limit; position += step) {
                    sum = private.d_in2_sub[position] += sum;
                }
            }
        }

        // work: build in2_sub2_sqr and adjust numerator (d_conv)
        {
            const int sub2_rows = public.in2_sub2_sqr_rows;
            const int sub2_cols = public.in2_sub2_sqr_cols;
            const int sub2_rows_stride = public.in2_sub2_sqr_rows;
            const int sel_rowlow = public.in2_sub_cumh_sel_rowlow - 1;
            const int sel_collow = public.in2_sub_cumh_sel_collow - 1;
            const int sel2_rowlow = public.in2_sub_cumh_sel2_rowlow - 1;
            const int sel2_collow = public.in2_sub_cumh_sel2_collow - 1;

            for (col = 0; col < sub2_cols; col++) {
                const int col_sel = col + sel_collow;
                const int col_sel2 = col + sel2_collow;
                const int sub_col_sel_base =
                    col_sel * in2_sub_rows_stride + sel_rowlow;
                const int sub_col_sel2_base =
                    col_sel2 * in2_sub_rows_stride + sel2_rowlow;
                const int dst_col_offset = col * sub2_rows_stride;
                const int conv_col_offset = col * sub2_rows_stride;
                for (row = 0; row < sub2_rows; row++) {
                    temp = private.d_in2_sub[sub_col_sel_base + row];
                    temp2 = private.d_in2_sub[sub_col_sel2_base + row];
                    temp2 = temp - temp2;
                    const fp sq = temp2 * temp2;
                    private.d_in2_sub2_sqr[dst_col_offset + row] = sq;
                    private.d_conv[conv_col_offset + row] -=
                        sq * in_final_sum / (fp)in_mod_elem;
                }
            }
        }

        // work: build padded in2 with squared values
        {
            const int add_rows = public.in2_pad_add_rows;
            const int add_cols = public.in2_pad_add_cols;
            for (col = 0; col < in2_pad_cols; col++) {
                const int pad_col_offset = col * in2_pad_rows_stride;
                const int col_gt = (col > (add_cols - 1));
                const int col_lt = (col < (add_cols + in2_cols));
                for (row = 0; row < in2_pad_rows; row++) {
                    if (row > (add_rows - 1) &&
                        row < (add_rows + in2_rows) && col_gt && col_lt) {
                        ori_row = row - add_rows;
                        ori_col = col - add_cols;
                        private.d_in2_pad[pad_col_offset + row] =
                            private.d_in2_sqr
                                [ori_col * in2_rows_stride + ori_row];
                    } else {
                        private.d_in2_pad[pad_col_offset + row] = 0;
                    }
                }
            }
        }

        // cumulative sum over rows (column-wise) for squared in2
        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {
            pos_ori = ei_new * in2_pad_rows_stride;
            sum = 0;
            const int end_pos = pos_ori + in2_pad_rows;
            for (position = pos_ori; position < end_pos; position++) {
                sum = private.d_in2_pad[position] += sum;
            }
        }

        // work: build in2_sub (squared case)
        {
            const int sel_rowlow = public.in2_pad_cumv_sel_rowlow - 1;
            const int sel_collow = public.in2_pad_cumv_sel_collow - 1;
            const int sel2_rowlow = public.in2_pad_cumv_sel2_rowlow - 1;
            const int sel2_collow = public.in2_pad_cumv_sel2_collow - 1;

            for (col = 0; col < in2_sub_cols; col++) {
                const int col_sel = col + sel_collow;
                const int col_sel2 = col + sel2_collow;
                const int pad_col_sel_base =
                    col_sel * in2_pad_rows_stride + sel_rowlow;
                const int pad_col_sel2_base =
                    col_sel2 * in2_pad_rows_stride + sel2_rowlow;
                const int dst_col_offset = col * in2_sub_rows_stride;
                for (row = 0; row < in2_sub_rows; row++) {
                    temp = private.d_in2_pad[pad_col_sel_base + row];
                    temp2 = private.d_in2_pad[pad_col_sel2_base + row];
                    private.d_in2_sub[dst_col_offset + row] = temp - temp2;
                }
            }
        }

        // cumulative sum over columns (row-wise) for in2_sub (squared)
        {
            const int in2_sub_elem = public.in2_sub_elem;
            const int step = in2_sub_rows;
            const int limit = in2_sub_rows + in2_sub_elem;
            for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {
                pos_ori = ei_new;
                sum = 0;
                for (position = pos_ori; position < limit; position += step) {
                    sum = private.d_in2_sub[position] += sum;
                }
            }
        }

        // work: final denominator and correlation
        {
            const int sel_rowlow = public.in2_sub_cumh_sel_rowlow - 1;
            const int sel_collow = public.in2_sub_cumh_sel_collow - 1;
            const int sel2_rowlow = public.in2_sub_cumh_sel2_rowlow - 1;
            const int sel2_collow = public.in2_sub_cumh_sel2_collow - 1;
            const int conv_rows_local = conv_rows;
            const int conv_cols_local = conv_cols;

            for (col = 0; col < conv_cols_local; col++) {
                const int col_sel = col + sel_collow;
                const int col_sel2 = col + sel2_collow;
                const int sub_col_sel_base =
                    col_sel * in2_sub_rows_stride + sel_rowlow;
                const int sub_col_sel2_base =
                    col_sel2 * in2_sub_rows_stride + sel2_rowlow;
                const int conv_col_offset = col * conv_rows_local;
                const int sub2_col_offset = col * conv_rows_local;
                for (row = 0; row < conv_rows_local; row++) {
                    temp = private.d_in2_sub[sub_col_sel_base + row];
                    temp2 = private.d_in2_sub[sub_col_sel2_base + row];
                    temp2 = temp - temp2;
                    temp2 -=
                        (private.d_in2_sub2_sqr[sub2_col_offset + row] /
                         (fp)in_mod_elem);
                    if (temp2 < 0) {
                        temp2 = 0;
                    }
                    temp2 = sqrt(temp2);
                    temp2 *= denomT;
                    private.d_conv[conv_col_offset + row] /=
                        temp2;
                }
            }
        }

        //====================================================================================================
        //	TEMPLATE MASK CREATE
        //====================================================================================================

        cent = public.sSize + public.tSize + 1;
        pointer = public.frame_no - 1 + private.point_no * public.frames;
        tMask_row = cent + private.d_tRowLoc[pointer] -
                    private.d_Row[private.point_no] - 1;
        tMask_col = cent + private.d_tColLoc[pointer] -
                    private.d_Col[private.point_no] - 1;

        for (ei_new = 0; ei_new < public.tMask_elem; ei_new++) {
            private.d_tMask[ei_new] = 0;
        }
        private.d_tMask[tMask_col * public.tMask_rows + tMask_row] = 1;

        // mask convolution
        {
            const int tMask_rows = public.tMask_rows;
            const int tMask_cols = public.tMask_cols;
            const int mask_rows = public.mask_rows;
            const int mask_cols = public.mask_cols;
            const int mask_conv_rows = public.mask_conv_rows;
            const int mask_conv_cols = public.mask_conv_cols;
            const int conv_rows_local = conv_rows;

            for (col = 1; col <= mask_conv_cols; col++) {

                j = col + public.mask_conv_joffset;
                jp1 = j + 1;
                ja1 = (mask_cols < jp1) ? (jp1 - mask_cols) : 1;
                ja2 = (tMask_cols < j) ? tMask_cols : j;

                const int mask_conv_col_offset =
                    (col - 1) * conv_rows_local;

                for (row = 1; row <= mask_conv_rows; row++) {

                    i = row + public.mask_conv_ioffset;
                    ip1 = i + 1;

                    ia1 = (mask_rows < ip1) ? (ip1 - mask_rows) : 1;
                    ia2 = (tMask_rows < i) ? tMask_rows : i;

                    s = 0;

                    for (ja = ja1; ja <= ja2; ja++) {
                        jb = jp1 - ja;
                        (void)jb; // jb unused but kept for structural parity
                        const int tmask_col_base =
                            (ja - 1) * tMask_rows - 1;
                        for (ia = ia1; ia <= ia2; ia++) {
                            ib = ip1 - ia;
                            (void)ib; // ib unused but kept
                            s += private.d_tMask[tmask_col_base + ia];
                        }
                    }

                    private.d_mask_conv[mask_conv_col_offset + (row - 1)] =
                        private.d_conv[mask_conv_col_offset + (row - 1)] * s;
                }
            }
        }

        fin_max_val = 0;
        fin_max_coo = 0;
        for (i = 0; i < public.mask_conv_elem; i++) {
            fp val = private.d_mask_conv[i];
            if (val > fin_max_val) {
                fin_max_val = val;
                fin_max_coo = i;
            }
        }

        // convert coordinate to row/col form
        largest_row =
            (fin_max_coo + 1) % public.mask_conv_rows - 1;
        largest_col = (fin_max_coo + 1) / public.mask_conv_rows;
        if ((fin_max_coo + 1) % public.mask_conv_rows == 0) {
            largest_row = public.mask_conv_rows - 1;
            largest_col = largest_col - 1;
        }

        largest_row = largest_row + 1;
        largest_col = largest_col + 1;
        offset_row =
            largest_row - in_mod_rows - (public.sSize - public.tSize);
        offset_col =
            largest_col - in_mod_cols - (public.sSize - public.tSize);
        pointer = private.point_no * public.frames + public.frame_no;
        private.d_tRowLoc[pointer] =
            private.d_Row[private.point_no] + offset_row;
        private.d_tColLoc[pointer] =
            private.d_Col[private.point_no] + offset_col;
    }

    // if the last frame in the batch, update template
    if (public.frame_no != 0 && (public.frame_no) % 10 == 0) {

        const int frame_rows = public.frame_rows;
        const int in_mod_rows = public.in_mod_rows;
        const int in_mod_cols = public.in_mod_cols;

        loc_pointer = private.point_no * public.frames + public.frame_no;
        private.d_Row[private.point_no] = private.d_tRowLoc[loc_pointer];
        private.d_Col[private.point_no] = private.d_tColLoc[loc_pointer];

        d_in = &private.d_T[private.in_pointer];

        const int base_row = private.d_Row[private.point_no] - 26;
        const int base_col = private.d_Col[private.point_no] - 26;
        const fp alpha = public.alpha;
        const fp one_minus_alpha = (fp)1.0 - alpha;

        for (col = 0; col < in_mod_cols; col++) {
            const int ori_col_base = (base_col + col) * frame_rows;
            const int dst_col_offset = col * in_mod_rows;
            for (row = 0; row < in_mod_rows; row++) {
                ori_row = base_row + row;
                ori_pointer = ori_col_base + ori_row;
                const int idx = dst_col_offset + row;
                d_in[idx] =
                    alpha * d_in[idx] +
                    one_minus_alpha * public.d_frame[ori_pointer];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    heartwall_kernel_time +=
        (kernel_end.tv_sec - kernel_start.tv_sec) +
        (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
