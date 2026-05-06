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
    const int frames = public.frames;
    const int in_mod_rows = public.in_mod_rows;
    const int in_mod_cols = public.in_mod_cols;
    const int in_mod_elem = public.in_mod_elem;
    const int in2_rows = public.in2_rows;
    const int in2_cols = public.in2_cols;
    const int in2_pad_rows = public.in2_pad_rows;
    const int in2_pad_cols = public.in2_pad_cols;
    const int in2_pad_add_rows = public.in2_pad_add_rows;
    const int in2_pad_add_cols = public.in2_pad_add_cols;
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
    const int sSize = public.sSize;
    const int tSize = public.tSize;
    const int frame_no = public.frame_no;
    const int point_no = private.point_no;

    fp *restrict d_frame = public.d_frame;
    fp *restrict d_T = private.d_T;
    fp *restrict d_Row = private.d_Row;
    fp *restrict d_Col = private.d_Col;
    fp *restrict d_tRowLoc = private.d_tRowLoc;
    fp *restrict d_tColLoc = private.d_tColLoc;
    fp *restrict d_in2 = private.d_in2;
    fp *restrict d_in2_sqr = private.d_in2_sqr;
    fp *restrict d_in_mod = private.d_in_mod;
    fp *restrict d_in_sqr = private.d_in_sqr;
    fp *restrict d_conv = private.d_conv;
    fp *restrict d_in2_pad = private.d_in2_pad;
    fp *restrict d_in2_sub = private.d_in2_sub;
    fp *restrict d_in2_sub2_sqr = private.d_in2_sub2_sqr;
    fp *restrict d_tMask = private.d_tMask;
    fp *restrict d_mask_conv = private.d_mask_conv;

    //======================================================================================================================================================
    //	GENERATE TEMPLATE
    //======================================================================================================================================================

    // generate templates based on the first frame only
    if (frame_no == 0) {

        // update temporary row/col coordinates
        pointer = point_no * frames + frame_no;
        d_tRowLoc[pointer] = d_Row[point_no];
        d_tColLoc[pointer] = d_Col[point_no];

        // pointers to: current frame, template for current point
        d_in = &d_T[private.in_pointer];

        const int base_row = d_Row[point_no] - 26;
        const int base_col = d_Col[point_no] - 26;

        // update template, limit the number of working threads to the size of
        // template
        for (col = 0; col < in_mod_cols; col++) {
            const int ori_col_base = (base_col + col) * frame_rows;
            const int out_col_base = col * in_mod_rows;
            for (row = 0; row < in_mod_rows; row++) {

                // figure out row/col location in corresponding new template
                // area in image and give to every thread (get top left corner
                // and progress down and right)
                ori_row = base_row + row;
                ori_pointer = ori_col_base + ori_row;

                // update template
                d_in[out_col_base + row] = d_frame[ori_pointer];
            }
        }
    }

    //======================================================================================================================================================
    //	PROCESS POINTS
    //======================================================================================================================================================

    // process points in all frames except for the first one
    if (frame_no != 0) {
        in2_rowlow = d_Row[point_no] - sSize;
        in2_collow = d_Col[point_no] - sSize;

        const int in2_rows_m1 = in2_rows - 1;
        const int in2_cols_m1 = in2_cols - 1;
        const int in_mod_rows_m1 = in_mod_rows - 1;
        const int in_mod_cols_m1 = in_mod_cols - 1;

        // work
        for (col = 0; col < in2_cols; col++) {
            const int ori_col_base = (in2_collow + col - 1) * frame_rows;
            const int out_col_base = col * in2_rows;
            for (row = 0; row < in2_rows; row++) {

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + in2_rowlow - 1;
                temp = d_frame[ori_col_base + ori_row];
                d_in2[out_col_base + row] = temp;
                d_in2_sqr[out_col_base + row] = temp * temp;
            }
        }

        // variables
        d_in = &d_T[private.in_pointer];

        // work
        for (col = 0; col < in_mod_cols; col++) {
            const int out_col_base = col * in_mod_rows;
            for (row = 0; row < in_mod_rows; row++) {

                // rotated coordinates
                rot_row = in_mod_rows_m1 - row;
                rot_col = in_mod_rows_m1 - col;
                pointer = rot_col * in_mod_rows + rot_row;

                // execution
                temp = d_in[pointer];
                d_in_mod[out_col_base + row] = temp;
                d_in_sqr[pointer] = temp * temp;
            }
        }

        in_final_sum = 0;
        for (i = 0; i < in_mod_elem; i++) {
            in_final_sum += d_in[i];
        }

        in_sqr_final_sum = 0;
        for (i = 0; i < in_mod_elem; i++) {
            in_sqr_final_sum += d_in_sqr[i];
        }

        mean = in_final_sum / in_mod_elem;
        mean_sqr = mean * mean;
        variance = (in_sqr_final_sum / in_mod_elem) - mean_sqr;
        deviation = sqrt(variance);

        denomT = sqrt((fp)(in_mod_elem - 1)) * deviation;

        const int conv_cols_loc = public.conv_cols;
        const int conv_rows_loc = conv_rows;
        const int joffset = public.joffset;
        const int ioffset = public.ioffset;

        // work
        for (col = 1; col <= conv_cols_loc; col++) {

            // column setup
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

            const int conv_col_base = (col - 1) * conv_rows_loc;

            for (row = 1; row <= conv_rows_loc; row++) {

                // row range setup
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

                // getting data
                for (ja = ja1; ja <= ja2; ja++) {
                    jb = jp1 - ja;
                    const int in_mod_base = in_mod_rows * (ja - 1);
                    const int in2_base = in2_rows * (jb - 1);
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        s += d_in_mod[in_mod_base + ia - 1] *
                             d_in2[in2_base + ib - 1];
                    }
                }

                d_conv[conv_col_base + (row - 1)] = s;
            }
        }

        const int in2_pad_add_rows_loc = in2_pad_add_rows;
        const int in2_pad_add_cols_loc = in2_pad_add_cols;
        const int in2_rows_loc = in2_rows;
        const int in2_cols_loc = in2_cols;

        // work
        for (col = 0; col < in2_pad_cols; col++) {
            const int pad_col_base = col * in2_pad_rows;
            for (row = 0; row < in2_pad_rows; row++) {

                if (row > (in2_pad_add_rows_loc - 1) &&
                    row < (in2_pad_add_rows_loc + in2_rows_loc) &&
                    col > (in2_pad_add_cols_loc - 1) &&
                    col < (in2_pad_add_cols_loc + in2_cols_loc)) {
                    ori_row = row - in2_pad_add_rows_loc;
                    ori_col = col - in2_pad_add_cols_loc;
                    d_in2_pad[pad_col_base + row] =
                        d_in2[ori_col * in2_rows_loc + ori_row];
                } else {
                    d_in2_pad[pad_col_base + row] = 0;
                }
            }
        }

        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {

            pos_ori = ei_new * in2_pad_rows;

            sum = 0;
            for (position = pos_ori; position < pos_ori + in2_pad_rows;
                 position++) {
                d_in2_pad[position] += sum;
                sum = d_in2_pad[position];
            }
        }

        const int in2_pad_rows_loc = in2_pad_rows;
        const int in2_sub_rows_loc = in2_sub_rows;
        const int in2_sub_cols_loc = in2_sub_cols;
        const int in2_sub_elem_loc = in2_sub_elem;

        const int in2_pad_cumv_sel_rowlow = public.in2_pad_cumv_sel_rowlow;
        const int in2_pad_cumv_sel_collow = public.in2_pad_cumv_sel_collow;
        const int in2_pad_cumv_sel2_rowlow = public.in2_pad_cumv_sel2_rowlow;
        const int in2_pad_cumv_sel2_collow = public.in2_pad_cumv_sel2_collow;

        // work
        for (col = 0; col < in2_sub_cols_loc; col++) {
            const int sub_col_base = col * in2_sub_rows_loc;
            const int col_offset1 =
                (col + in2_pad_cumv_sel_collow - 1) * in2_pad_rows_loc;
            const int col_offset2 =
                (col + in2_pad_cumv_sel2_collow - 1) * in2_pad_rows_loc;
            for (row = 0; row < in2_sub_rows_loc; row++) {

                ori_row = row + in2_pad_cumv_sel_rowlow - 1;
                temp = d_in2_pad[col_offset1 + ori_row];

                ori_row = row + in2_pad_cumv_sel2_rowlow - 1;
                temp2 = d_in2_pad[col_offset2 + ori_row];

                d_in2_sub[sub_col_base + row] = temp - temp2;
            }
        }

        for (ei_new = 0; ei_new < in2_sub_rows_loc; ei_new++) {

            pos_ori = ei_new;

            sum = 0;
            for (position = pos_ori;
                 position < pos_ori + in2_sub_elem_loc;
                 position += in2_sub_rows_loc) {
                d_in2_sub[position] += sum;
                sum = d_in2_sub[position];
            }
        }

        const int in2_sub2_sqr_rows_loc = in2_sub2_sqr_rows;
        const int in2_sub2_sqr_cols_loc = in2_sub2_sqr_cols;

        const int in2_sub_cumh_sel_rowlow = public.in2_sub_cumh_sel_rowlow;
        const int in2_sub_cumh_sel_collow = public.in2_sub_cumh_sel_collow;
        const int in2_sub_cumh_sel2_rowlow = public.in2_sub_cumh_sel2_rowlow;
        const int in2_sub_cumh_sel2_collow = public.in2_sub_cumh_sel2_collow;

        // work
        for (col = 0; col < in2_sub2_sqr_cols_loc; col++) {
            const int sub2_col_base = col * in2_sub2_sqr_rows_loc;
            const int conv_col_base = col * in2_sub2_sqr_rows_loc;
            const int col_offset1 =
                (col + in2_sub_cumh_sel_collow - 1) * in2_sub_rows_loc;
            const int col_offset2 =
                (col + in2_sub_cumh_sel2_collow - 1) * in2_sub_rows_loc;
            for (row = 0; row < in2_sub2_sqr_rows_loc; row++) {

                ori_row = row + in2_sub_cumh_sel_rowlow - 1;
                temp = d_in2_sub[col_offset1 + ori_row];

                ori_row = row + in2_sub_cumh_sel2_rowlow - 1;
                temp2 = d_in2_sub[col_offset2 + ori_row];

                temp2 = temp - temp2;

                temp2 = temp2 * temp2;

                d_in2_sub2_sqr[sub2_col_base + row] = temp2;

                d_conv[conv_col_base + row] -= temp2 * in_final_sum / in_mod_elem;
            }
        }

        // work
        for (col = 0; col < in2_pad_cols; col++) {
            const int pad_col_base = col * in2_pad_rows_loc;
            for (row = 0; row < in2_pad_rows_loc; row++) {

                if (row > (in2_pad_add_rows_loc - 1) &&
                    row < (in2_pad_add_rows_loc + in2_rows_loc) &&
                    col > (in2_pad_add_cols_loc - 1) &&
                    col < (in2_pad_add_cols_loc + in2_cols_loc)) {
                    ori_row = row - in2_pad_add_rows_loc;
                    ori_col = col - in2_pad_add_cols_loc;
                    d_in2_pad[pad_col_base + row] =
                        d_in2_sqr[ori_col * in2_rows_loc + ori_row];
                } else {
                    d_in2_pad[pad_col_base + row] = 0;
                }
            }
        }

        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {

            pos_ori = ei_new * in2_pad_rows_loc;

            sum = 0;
            for (position = pos_ori; position < pos_ori + in2_pad_rows_loc;
                 position++) {
                d_in2_pad[position] += sum;
                sum = d_in2_pad[position];
            }
        }

        // work
        for (col = 0; col < in2_sub_cols_loc; col++) {
            const int sub_col_base = col * in2_sub_rows_loc;
            const int col_offset1 =
                (col + in2_pad_cumv_sel_rowlow - 1 + 0) * in2_pad_rows_loc;
            const int col_offset2 =
                (col + in2_pad_cumv_sel2_collow - 1 + 0) * in2_pad_rows_loc;

            for (row = 0; row < in2_sub_rows_loc; row++) {

                ori_row = row + in2_pad_cumv_sel_rowlow - 1;
                temp = d_in2_pad[(col + in2_pad_cumv_sel_collow - 1) *
                                     in2_pad_rows_loc +
                                 ori_row];

                ori_row = row + in2_pad_cumv_sel2_rowlow - 1;
                temp2 = d_in2_pad[(col + in2_pad_cumv_sel2_collow - 1) *
                                      in2_pad_rows_loc +
                                  ori_row];

                d_in2_sub[sub_col_base + row] = temp - temp2;
            }
        }

        for (ei_new = 0; ei_new < in2_sub_rows_loc; ei_new++) {

            pos_ori = ei_new;

            sum = 0;
            for (position = pos_ori;
                 position < pos_ori + in2_sub_elem_loc;
                 position += in2_sub_rows_loc) {
                d_in2_sub[position] += sum;
                sum = d_in2_sub[position];
            }
        }

        const int conv_cols_loc2 = conv_cols;
        const int conv_rows_loc2 = conv_rows;

        // work
        for (col = 0; col < conv_cols_loc2; col++) {
            const int conv_col_base = col * conv_rows_loc2;
            const int col_offset1 =
                (col + in2_sub_cumh_sel_collow - 1) * in2_sub_rows_loc;
            const int col_offset2 =
                (col + in2_sub_cumh_sel2_collow - 1) * in2_sub_rows_loc;
            for (row = 0; row < conv_rows_loc2; row++) {

                ori_row = row + in2_sub_cumh_sel_rowlow - 1;
                temp = d_in2_sub[col_offset1 + ori_row];

                ori_row = row + in2_sub_cumh_sel2_rowlow - 1;
                temp2 = d_in2_sub[col_offset2 + ori_row];

                temp2 = temp - temp2;

                temp2 = temp2 -
                        (d_in2_sub2_sqr[conv_col_base + row] / in_mod_elem);

                if (temp2 < 0) {
                    temp2 = 0;
                }
                temp2 = sqrt(temp2);

                temp2 = denomT * temp2;

                d_conv[conv_col_base + row] /= temp2;
            }
        }

        //====================================================================================================
        //	TEMPLATE MASK CREATE
        //====================================================================================================

        cent = sSize + tSize + 1;
        pointer = frame_no - 1 + point_no * frames;
        tMask_row = cent + d_tRowLoc[pointer] - d_Row[point_no] - 1;
        tMask_col = cent + d_tColLoc[pointer] - d_Col[point_no] - 1;

        // work
        for (ei_new = 0; ei_new < tMask_elem; ei_new++) {
            d_tMask[ei_new] = 0;
        }
        d_tMask[tMask_col * tMask_rows + tMask_row] = 1;

        const int mask_conv_cols_loc = mask_conv_cols;
        const int mask_conv_rows_loc = mask_conv_rows;
        const int mask_conv_joffset = public.mask_conv_joffset;
        const int mask_conv_ioffset = public.mask_conv_ioffset;

        // work
        for (col = 1; col <= mask_conv_cols_loc; col++) {

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

            const int mask_conv_col_base = (col - 1) * conv_rows;

            for (row = 1; row <= mask_conv_rows_loc; row++) {

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
                    (void)jb;
                    const int tMask_base = tMask_rows * (ja - 1);
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        (void)ib;
                        s += d_tMask[tMask_base + ia - 1];
                    }
                }

                d_mask_conv[mask_conv_col_base + (row - 1)] =
                    d_conv[mask_conv_col_base + (row - 1)] * s;
            }
        }

        fin_max_val = 0;
        fin_max_coo = 0;
        const int mask_conv_elem = public.mask_conv_elem;
        for (i = 0; i < mask_conv_elem; i++) {
            if (d_mask_conv[i] > fin_max_val) {
                fin_max_val = d_mask_conv[i];
                fin_max_coo = i;
            }
        }

        largest_row = (fin_max_coo + 1) % mask_conv_rows - 1;
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
        d_tRowLoc[pointer] = d_Row[point_no] + offset_row;
        d_tColLoc[pointer] = d_Col[point_no] + offset_col;
    }

    // if the last frame in the bath, update template
    if (frame_no != 0 && (frame_no % 10) == 0) {

        loc_pointer = point_no * frames + frame_no;
        d_Row[point_no] = d_tRowLoc[loc_pointer];
        d_Col[point_no] = d_tColLoc[loc_pointer];

        d_in = &d_T[private.in_pointer];

        const int base_row2 = d_Row[point_no] - 26;
        const int base_col2 = d_Col[point_no] - 26;
        const fp alpha = public.alpha;
        const fp one_minus_alpha = (fp)1.0 - alpha;

        for (col = 0; col < in_mod_cols; col++) {
            const int ori_col_base = (base_col2 + col) * frame_rows;
            const int out_col_base = col * in_mod_rows;
            for (row = 0; row < in_mod_rows; row++) {

                ori_row = base_row2 + row;
                ori_pointer = ori_col_base + ori_row;

                const int idx = out_col_base + row;
                d_in[idx] =
                    alpha * d_in[idx] + one_minus_alpha * d_frame[ori_pointer];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    heartwall_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
