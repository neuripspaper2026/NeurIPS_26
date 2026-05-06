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
    fp largest_value_current = 0;
    fp largest_value = 0;
    int largest_coordinate_current = 0;
    int largest_coordinate = 0;
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

    const int frame_rows = public.frame_rows;
    const int in_mod_rows = public.in_mod_rows;
    const int in_mod_cols = public.in_mod_cols;
    const int in_mod_elem = public.in_mod_elem;
    const int in2_rows = public.in2_rows;
    const int in2_cols = public.in2_cols;
    const int in2_pad_rows = public.in2_pad_rows;
    const int in2_pad_cols = public.in2_pad_cols;
    const int in2_sub_rows = public.in2_sub_rows;
    const int in2_sub_cols = public.in2_sub_cols;
    const int in2_sub_elem = public.in2_sub_elem;
    const int in2_sub2_sqr_rows = public.in2_sub2_sqr_rows;
    const int in2_sub2_sqr_cols = public.in2_sub2_sqr_cols;
    const int conv_rows = public.conv_rows;
    const int conv_cols = public.conv_cols;
    const int mask_conv_rows = public.mask_conv_rows;
    const int mask_conv_cols = public.mask_conv_cols;
    const int tMask_rows = public.tMask_rows;
    const int tMask_cols = public.tMask_cols;
    const int tMask_elem = public.tMask_elem;
    const int mask_rows = public.mask_rows;
    const int mask_cols = public.mask_cols;

    const int in2_pad_add_rows = public.in2_pad_add_rows;
    const int in2_pad_add_cols = public.in2_pad_add_cols;
    const int in2_pad_cumv_sel_rowlow = public.in2_pad_cumv_sel_rowlow;
    const int in2_pad_cumv_sel_collow = public.in2_pad_cumv_sel_collow;
    const int in2_pad_cumv_sel2_rowlow = public.in2_pad_cumv_sel2_rowlow;
    const int in2_pad_cumv_sel2_collow = public.in2_pad_cumv_sel2_collow;
    const int in2_sub_cumh_sel_rowlow = public.in2_sub_cumh_sel_rowlow;
    const int in2_sub_cumh_sel_collow = public.in2_sub_cumh_sel_collow;
    const int in2_sub_cumh_sel2_rowlow = public.in2_sub_cumh_sel2_rowlow;
    const int in2_sub_cumh_sel2_collow = public.in2_sub_cumh_sel2_collow;

    const int sSize = public.sSize;
    const int tSize = public.tSize;
    const int frames = public.frames;
    const int frame_no = public.frame_no;
    const int point_no = private.point_no;

    const int conv_rows_stride = conv_rows;
    const int in_mod_rows_stride = in_mod_rows;
    const int in2_rows_stride = in2_rows;
    const int in2_pad_rows_stride = in2_pad_rows;
    const int in2_sub_rows_stride = in2_sub_rows;
    const int in2_sub2_sqr_rows_stride = in2_sub2_sqr_rows;
    const int tMask_rows_stride = tMask_rows;

    //======================================================================================================================================================
    //	GENERATE TEMPLATE
    //======================================================================================================================================================

    // generate templates based on the first frame only
    if (frame_no == 0) {

        // update temporary row/col coordinates
        pointer = point_no * frames + frame_no;
        private.d_tRowLoc[pointer] = private.d_Row[point_no];
        private.d_tColLoc[pointer] = private.d_Col[point_no];

        // pointers to: current frame, template for current point
        d_in = &private.d_T[private.in_pointer];

        const int base_row = private.d_Row[point_no] - 26;
        const int base_col = private.d_Col[point_no] - 26;

        // update template, limit the number of working threads to the size of template
        for (col = 0; col < in_mod_cols; col++) {
            int col_offset = col * in_mod_rows;
            int ori_col_base = (base_col + col) * frame_rows;
            for (row = 0; row < in_mod_rows; row++) {
                ori_row = base_row + row;
                ori_pointer = ori_col_base + ori_row;
                d_in[col_offset + row] = public.d_frame[ori_pointer];
            }
        }
    }

    //======================================================================================================================================================
    //	PROCESS POINTS
    //======================================================================================================================================================

    // process points in all frames except for the first one
    if (frame_no != 0) {
        in2_rowlow = private.d_Row[point_no] - sSize; // (1 to n+1)
        in2_collow = private.d_Col[point_no] - sSize;

        // work
        for (col = 0; col < in2_cols; col++) {
            int in2_index_base = col * in2_rows;
            int frame_col_base = (col + in2_collow - 1) * frame_rows;
            for (row = 0; row < in2_rows; row++) {

                ori_row = row + in2_rowlow - 1;
                temp = public.d_frame[frame_col_base + ori_row];
                int idx = in2_index_base + row;
                private.d_in2[idx] = temp;
                private.d_in2_sqr[idx] = temp * temp;
            }
        }

        // variables
        d_in = &private.d_T[private.in_pointer];

        // work
        const int in_mod_rows_minus_1 = in_mod_rows - 1;
        for (col = 0; col < in_mod_cols; col++) {
            int in_mod_col_offset = col * in_mod_rows;
            for (row = 0; row < in_mod_rows; row++) {

                rot_row = in_mod_rows_minus_1 - row;
                rot_col = in_mod_rows_minus_1 - col;
                pointer = rot_col * in_mod_rows + rot_row;

                temp = d_in[pointer];
                private.d_in_mod[in_mod_col_offset + row] = temp;
                private.d_in_sqr[pointer] = temp * temp;
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

        // work
        const int joffset = public.joffset;
        const int ioffset = public.ioffset;

        for (col = 1; col <= conv_cols; col++) {

            j = col + joffset;
            jp1 = j + 1;
            if (in2_cols < jp1) {
                ja1 = jp1 - in2_cols;
            } else {
                ja1 = 1;
            }
            if (in_mod_cols < j) {
                ja2 = in_mod_cols;
            } else {
                ja2 = j;
            }

            int conv_col_offset = (col - 1) * conv_rows_stride;

            for (row = 1; row <= conv_rows; row++) {

                i = row + ioffset;
                ip1 = i + 1;

                if (in2_rows < ip1) {
                    ia1 = ip1 - in2_rows;
                } else {
                    ia1 = 1;
                }
                if (in_mod_rows < i) {
                    ia2 = in_mod_rows;
                } else {
                    ia2 = i;
                }

                s = 0;

                for (ja = ja1; ja <= ja2; ja++) {
                    jb = jp1 - ja;
                    int in_mod_col_base = in_mod_rows_stride * (ja - 1);
                    int in2_col_base = in2_rows_stride * (jb - 1);
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        s += private.d_in_mod[in_mod_col_base + ia - 1] *
                             private.d_in2[in2_col_base + ib - 1];
                    }
                }

                private.d_conv[conv_col_offset + (row - 1)] = s;
            }
        }

        // work
        for (col = 0; col < in2_pad_cols; col++) {
            int pad_col_offset = col * in2_pad_rows_stride;
            for (row = 0; row < in2_pad_rows; row++) {

                if (row > (in2_pad_add_rows - 1) &&
                    row < (in2_pad_add_rows + in2_rows) &&
                    col > (in2_pad_add_cols - 1) &&
                    col < (in2_pad_add_cols + in2_cols)) {
                    ori_row = row - in2_pad_add_rows;
                    ori_col = col - in2_pad_add_cols;
                    private.d_in2_pad[pad_col_offset + row] =
                        private.d_in2[ori_col * in2_rows_stride + ori_row];
                } else {
                    private.d_in2_pad[pad_col_offset + row] = 0;
                }
            }
        }

        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {

            pos_ori = ei_new * in2_pad_rows_stride;

            sum = 0;
            int pos_end = pos_ori + in2_pad_rows;
            for (position = pos_ori; position < pos_end; position++) {
                private.d_in2_pad[position] += sum;
                sum = private.d_in2_pad[position];
            }
        }

        // work
        for (col = 0; col < in2_sub_cols; col++) {
            int sub_col_offset = col * in2_sub_rows_stride;
            for (row = 0; row < in2_sub_rows; row++) {

                ori_row = row + in2_pad_cumv_sel_rowlow - 1;
                ori_col = col + in2_pad_cumv_sel_collow - 1;
                temp = private.d_in2_pad[ori_col * in2_pad_rows_stride + ori_row];

                ori_row = row + in2_pad_cumv_sel2_rowlow - 1;
                ori_col = col + in2_pad_cumv_sel2_collow - 1;
                temp2 = private.d_in2_pad[ori_col * in2_pad_rows_stride + ori_row];

                private.d_in2_sub[sub_col_offset + row] = temp - temp2;
            }
        }

        for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {

            pos_ori = ei_new;

            sum = 0;
            int pos_end = pos_ori + in2_sub_elem * in2_sub_rows_stride;
            for (position = pos_ori; position < pos_end; position += in2_sub_rows_stride) {
                private.d_in2_sub[position] += sum;
                sum = private.d_in2_sub[position];
            }
        }

        // work
        for (col = 0; col < in2_sub2_sqr_cols; col++) {
            int sqr_col_offset = col * in2_sub2_sqr_rows_stride;
            int conv_col_offset2 = col * in2_sub2_sqr_rows_stride;
            for (row = 0; row < in2_sub2_sqr_rows; row++) {

                ori_row = row + in2_sub_cumh_sel_rowlow - 1;
                ori_col = col + in2_sub_cumh_sel_collow - 1;
                temp = private.d_in2_sub[ori_col * in2_sub_rows_stride + ori_row];

                ori_row = row + in2_sub_cumh_sel2_rowlow - 1;
                ori_col = col + in2_sub_cumh_sel2_collow - 1;
                temp2 = private.d_in2_sub[ori_col * in2_sub_rows_stride + ori_row];

                temp2 = temp - temp2;
                fp temp2_sqr = temp2 * temp2;
                private.d_in2_sub2_sqr[sqr_col_offset + row] = temp2_sqr;

                private.d_conv[conv_col_offset2 + row] -= temp2_sqr * in_final_sum / (fp)in_mod_elem;
            }
        }

        // work
        for (col = 0; col < in2_pad_cols; col++) {
            int pad_col_offset = col * in2_pad_rows_stride;
            for (row = 0; row < in2_pad_rows; row++) {

                if (row > (in2_pad_add_rows - 1) &&
                    row < (in2_pad_add_rows + in2_rows) &&
                    col > (in2_pad_add_cols - 1) &&
                    col < (in2_pad_add_cols + in2_cols)) {
                    ori_row = row - in2_pad_add_rows;
                    ori_col = col - in2_pad_add_cols;
                    private.d_in2_pad[pad_col_offset + row] =
                        private.d_in2_sqr[ori_col * in2_rows_stride + ori_row];
                } else {
                    private.d_in2_pad[pad_col_offset + row] = 0;
                }
            }
        }

        // work
        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {

            pos_ori = ei_new * in2_pad_rows_stride;

            sum = 0;
            int pos_end = pos_ori + in2_pad_rows;
            for (position = pos_ori; position < pos_end; position++) {
                private.d_in2_pad[position] += sum;
                sum = private.d_in2_pad[position];
            }
        }

        // work
        for (col = 0; col < in2_sub_cols; col++) {
            int sub_col_offset = col * in2_sub_rows_stride;
            for (row = 0; row < in2_sub_rows; row++) {

                ori_row = row + in2_pad_cumv_sel_rowlow - 1;
                ori_col = col + in2_pad_cumv_sel_collow - 1;
                temp = private.d_in2_pad[ori_col * in2_pad_rows_stride + ori_row];

                ori_row = row + in2_pad_cumv_sel2_rowlow - 1;
                ori_col = col + in2_pad_cumv_sel2_collow - 1;
                temp2 = private.d_in2_pad[ori_col * in2_pad_rows_stride + ori_row];

                private.d_in2_sub[sub_col_offset + row] = temp - temp2;
            }
        }

        for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {

            pos_ori = ei_new;

            sum = 0;
            int pos_end = pos_ori + in2_sub_elem * in2_sub_rows_stride;
            for (position = pos_ori; position < pos_end; position += in2_sub_rows_stride) {
                private.d_in2_sub[position] += sum;
                sum = private.d_in2_sub[position];
            }
        }

        // work
        for (col = 0; col < conv_cols; col++) {
            int conv_col_offset3 = col * conv_rows_stride;
            for (row = 0; row < conv_rows; row++) {

                ori_row = row + in2_sub_cumh_sel_rowlow - 1;
                ori_col = col + in2_sub_cumh_sel_collow - 1;
                temp = private.d_in2_sub[ori_col * in2_sub_rows_stride + ori_row];

                ori_row = row + in2_sub_cumh_sel2_rowlow - 1;
                ori_col = col + in2_sub_cumh_sel2_collow - 1;
                temp2 = private.d_in2_sub[ori_col * in2_sub_rows_stride + ori_row];

                temp2 = temp - temp2;

                temp2 = temp2 -
                        (private.d_in2_sub2_sqr[conv_col_offset3 + row] /
                         (fp)in_mod_elem);

                if (temp2 < 0) {
                    temp2 = 0;
                }
                temp2 = sqrt(temp2);

                temp2 = denomT * temp2;

                private.d_conv[conv_col_offset3 + row] /=
                    temp2;
            }
        }

        //====================================================================================================
        //	TEMPLATE MASK CREATE
        //====================================================================================================

        cent = sSize + tSize + 1;
        pointer = frame_no - 1 + point_no * frames;
        tMask_row = cent + private.d_tRowLoc[pointer] - private.d_Row[point_no] - 1;
        tMask_col = cent + private.d_tColLoc[pointer] - private.d_Col[point_no] - 1;

        for (ei_new = 0; ei_new < tMask_elem; ei_new++) {
            private.d_tMask[ei_new] = 0;
        }
        private.d_tMask[tMask_col * tMask_rows_stride + tMask_row] = 1;

        const int mask_conv_joffset = public.mask_conv_joffset;
        const int mask_conv_ioffset = public.mask_conv_ioffset;

        for (col = 1; col <= mask_conv_cols; col++) {

            j = col + mask_conv_joffset;
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

            int mask_conv_col_offset = (col - 1) * conv_rows_stride;
            int conv_col_offset4 = (col - 1) * conv_rows_stride;

            for (row = 1; row <= mask_conv_rows; row++) {

                i = row + mask_conv_ioffset;
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

                s = 0;

                for (ja = ja1; ja <= ja2; ja++) {
                    jb = jp1 - ja;
                    (void)jb; // jb is not used in the computation (value is 1 multiplier)
                    int tMask_col_base = tMask_rows_stride * (ja - 1);
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        (void)ib; // ib is not used because multiplier is 1
                        s += private.d_tMask[tMask_col_base + ia - 1];
                    }
                }

                private.d_mask_conv[mask_conv_col_offset + (row - 1)] =
                    private.d_conv[conv_col_offset4 + (row - 1)] * s;
            }
        }

        fin_max_val = 0;
        fin_max_coo = 0;
        const int mask_conv_elem = public.mask_conv_elem;
        for (i = 0; i < mask_conv_elem; i++) {
            fp val = private.d_mask_conv[i];
            if (val > fin_max_val) {
                fin_max_val = val;
                fin_max_coo = i;
            }
        }

        largest_row =
            (fin_max_coo + 1) % mask_conv_rows - 1;
        largest_col = (fin_max_coo + 1) / mask_conv_rows;
        if ((fin_max_coo + 1) % mask_conv_rows == 0) {
            largest_row = mask_conv_rows - 1;
            largest_col = largest_col - 1;
        }

        largest_row = largest_row + 1;
        largest_col = largest_col + 1;
        offset_row =
            largest_row - in_mod_rows - (sSize - tSize);
        offset_col =
            largest_col - in_mod_cols - (sSize - tSize);
        pointer = point_no * frames + frame_no;
        private.d_tRowLoc[pointer] = private.d_Row[point_no] + offset_row;
        private.d_tColLoc[pointer] = private.d_Col[point_no] + offset_col;
    }

    // if the last frame in the bath, update template
    if (frame_no != 0 && (frame_no) % 10 == 0) {

        loc_pointer = point_no * frames + frame_no;
        private.d_Row[point_no] = private.d_tRowLoc[loc_pointer];
        private.d_Col[point_no] = private.d_tColLoc[loc_pointer];

        d_in = &private.d_T[private.in_pointer];

        const int base_row_upd = private.d_Row[point_no] - 26;
        const int base_col_upd = private.d_Col[point_no] - 26;
        const fp alpha = public.alpha;
        const fp one_minus_alpha = (fp)1.0 - alpha;

        for (col = 0; col < in_mod_cols; col++) {
            int in_mod_col_offset = col * in_mod_rows;
            int frame_col_base = (base_col_upd + col) * frame_rows;
            for (row = 0; row < in_mod_rows; row++) {

                ori_row = base_row_upd + row;
                ori_pointer = frame_col_base + ori_row;

                int idx = in_mod_col_offset + row;
                d_in[idx] = alpha * d_in[idx] +
                            one_minus_alpha * public.d_frame[ori_pointer];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    heartwall_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
