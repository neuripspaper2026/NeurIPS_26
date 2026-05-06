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

    const int in_mod_rows           = public.in_mod_rows;
    const int in_mod_cols           = public.in_mod_cols;
    const int in_mod_elem           = public.in_mod_elem;
    const int in2_rows              = public.in2_rows;
    const int in2_cols              = public.in2_cols;
    const int in2_pad_rows          = public.in2_pad_rows;
    const int in2_pad_cols          = public.in2_pad_cols;
    const int in2_sub_rows          = public.in2_sub_rows;
    const int in2_sub_cols          = public.in2_sub_cols;
    const int in2_sub_elem          = public.in2_sub_elem;
    const int in2_sub2_sqr_rows     = public.in2_sub2_sqr_rows;
    const int in2_sub2_sqr_cols     = public.in2_sub2_sqr_cols;
    const int conv_rows             = public.conv_rows;
    const int conv_cols             = public.conv_cols;
    const int mask_conv_rows        = public.mask_conv_rows;
    const int mask_conv_cols        = public.mask_conv_cols;
    const int tMask_rows            = public.tMask_rows;
    const int tMask_cols            = public.tMask_cols;
    const int tMask_elem            = public.tMask_elem;
    const int frame_rows            = public.frame_rows;
    const int frames                = public.frames;
    const int sSize                 = public.sSize;
    const int tSize                 = public.tSize;
    const int mask_rows             = public.mask_rows;
    const int mask_cols             = public.mask_cols;
    const int mask_conv_elem        = public.mask_conv_elem;
    const int in2_pad_add_rows      = public.in2_pad_add_rows;
    const int in2_pad_add_cols      = public.in2_pad_add_cols;
    const int in2_pad_cumv_sel_rowlow  = public.in2_pad_cumv_sel_rowlow;
    const int in2_pad_cumv_sel_collow  = public.in2_pad_cumv_sel_collow;
    const int in2_pad_cumv_sel2_rowlow = public.in2_pad_cumv_sel2_rowlow;
    const int in2_pad_cumv_sel2_collow = public.in2_pad_cumv_sel2_collow;
    const int in2_sub_cumh_sel_rowlow  = public.in2_sub_cumh_sel_rowlow;
    const int in2_sub_cumh_sel_collow  = public.in2_sub_cumh_sel_collow;
    const int in2_sub_cumh_sel2_rowlow = public.in2_sub_cumh_sel2_rowlow;
    const int in2_sub_cumh_sel2_collow = public.in2_sub_cumh_sel2_collow;
    const int joffset               = public.joffset;
    const int ioffset               = public.ioffset;
    const int mask_conv_joffset     = public.mask_conv_joffset;
    const int mask_conv_ioffset     = public.mask_conv_ioffset;
    const fp  alpha                 = public.alpha;

    fp *restrict d_frame        = public.d_frame;
    fp *restrict d_T            = private.d_T;
    fp *restrict d_Row          = private.d_Row;
    fp *restrict d_Col          = private.d_Col;
    fp *restrict d_tRowLoc      = private.d_tRowLoc;
    fp *restrict d_tColLoc      = private.d_tColLoc;
    fp *restrict d_in2          = private.d_in2;
    fp *restrict d_in2_sqr      = private.d_in2_sqr;
    fp *restrict d_in_mod       = private.d_in_mod;
    fp *restrict d_in_sqr       = private.d_in_sqr;
    fp *restrict d_conv         = private.d_conv;
    fp *restrict d_in2_pad      = private.d_in2_pad;
    fp *restrict d_in2_sub      = private.d_in2_sub;
    fp *restrict d_in2_sub2_sqr = private.d_in2_sub2_sqr;
    fp *restrict d_tMask        = private.d_tMask;
    fp *restrict d_mask_conv    = private.d_mask_conv;

    //======================================================================================================================================================
    //	GENERATE TEMPLATE
    //======================================================================================================================================================

    if (public.frame_no == 0) {

        pointer = private.point_no * frames + public.frame_no;
        d_tRowLoc[pointer] = d_Row[private.point_no];
        d_tColLoc[pointer] = d_Col[private.point_no];

        d_in = &d_T[private.in_pointer];

        const int point_row = (int)d_Row[private.point_no];
        const int point_col = (int)d_Col[private.point_no];

        for (col = 0; col < in_mod_cols; col++) {
            const int col_offset = col * in_mod_rows;
            const int ori_col_base = point_col - 25 + col - 1;
            const int frame_col_offset = ori_col_base * frame_rows;
            for (row = 0; row < in_mod_rows; row++) {
                ori_row = point_row - 25 + row - 1;
                ori_pointer = frame_col_offset + ori_row;
                d_in[col_offset + row] = d_frame[ori_pointer];
            }
        }
    }

    //======================================================================================================================================================
    //	PROCESS POINTS
    //======================================================================================================================================================

    if (public.frame_no != 0) {
        in2_rowlow = (int)d_Row[private.point_no] - sSize;
        in2_collow = (int)d_Col[private.point_no] - sSize;

        for (col = 0; col < in2_cols; col++) {
            const int ori_col_base = col + in2_collow - 1;
            const int frame_col_offset = ori_col_base * frame_rows;
            const int dst_col_offset = col * in2_rows;
            for (row = 0; row < in2_rows; row++) {
                ori_row = row + in2_rowlow - 1;
                temp = d_frame[frame_col_offset + ori_row];
                d_in2[dst_col_offset + row] = temp;
                d_in2_sqr[dst_col_offset + row] = temp * temp;
            }
        }

        d_in = &d_T[private.in_pointer];

        for (col = 0; col < in_mod_cols; col++) {
            const int dst_col_offset = col * in_mod_rows;
            const int rot_col_base = (in_mod_rows - 1) - col;
            const int rot_col_offset = rot_col_base * in_mod_rows;
            for (row = 0; row < in_mod_rows; row++) {
                rot_row = (in_mod_rows - 1) - row;
                pointer = rot_col_offset + rot_row;
                temp = d_in[pointer];
                d_in_mod[dst_col_offset + row] = temp;
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

        for (col = 1; col <= conv_cols; col++) {

            j = col + joffset;
            jp1 = j + 1;
            ja1 = (in2_cols < jp1) ? (jp1 - in2_cols) : 1;
            ja2 = (in_mod_cols < j) ? in_mod_cols : j;

            const int conv_col_offset = (col - 1) * conv_rows;

            for (row = 1; row <= conv_rows; row++) {

                i = row + ioffset;
                ip1 = i + 1;

                ia1 = (in2_rows < ip1) ? (ip1 - in2_rows) : 1;
                ia2 = (in_mod_rows < i) ? in_mod_rows : i;

                s = 0;

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

                d_conv[conv_col_offset + (row - 1)] = s;
            }
        }

        for (col = 0; col < in2_pad_cols; col++) {
            const int pad_col_offset = col * in2_pad_rows;
            if (col > (in2_pad_add_cols - 1) && col < (in2_pad_add_cols + in2_cols)) {
                ori_col = col - in2_pad_add_cols;
                const int in2_col_offset = ori_col * in2_rows;
                for (row = 0; row < in2_pad_rows; row++) {
                    if (row > (in2_pad_add_rows - 1) &&
                        row < (in2_pad_add_rows + in2_rows)) {
                        ori_row = row - in2_pad_add_rows;
                        d_in2_pad[pad_col_offset + row] =
                            d_in2[in2_col_offset + ori_row];
                    } else {
                        d_in2_pad[pad_col_offset + row] = 0;
                    }
                }
            } else {
                for (row = 0; row < in2_pad_rows; row++) {
                    d_in2_pad[pad_col_offset + row] = 0;
                }
            }
        }

        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {

            pos_ori = ei_new * in2_pad_rows;

            sum = 0;
            const int end_pos = pos_ori + in2_pad_rows;
            for (position = pos_ori; position < end_pos; position++) {
                d_in2_pad[position] += sum;
                sum = d_in2_pad[position];
            }
        }

        for (col = 0; col < in2_sub_cols; col++) {
            const int sub_col_offset = col * in2_sub_rows;

            ori_col = col + in2_pad_cumv_sel_collow - 1;
            const int pad_col_offset1 = ori_col * in2_pad_rows;

            ori_col = col + in2_pad_cumv_sel2_collow - 1;
            const int pad_col_offset2 = ori_col * in2_pad_rows;

            for (row = 0; row < in2_sub_rows; row++) {

                ori_row = row + in2_pad_cumv_sel_rowlow - 1;
                temp = d_in2_pad[pad_col_offset1 + ori_row];

                ori_row = row + in2_pad_cumv_sel2_rowlow - 1;
                temp2 = d_in2_pad[pad_col_offset2 + ori_row];

                d_in2_sub[sub_col_offset + row] = temp - temp2;
            }
        }

        for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {

            pos_ori = ei_new;

            sum = 0;
            const int end_pos = pos_ori + in2_sub_elem;
            for (position = pos_ori; position < end_pos; position += in2_sub_rows) {
                d_in2_sub[position] += sum;
                sum = d_in2_sub[position];
            }
        }

        for (col = 0; col < in2_sub2_sqr_cols; col++) {
            const int sub2_col_offset = col * in2_sub2_sqr_rows;
            const int conv_col_offset = col * in2_sub2_sqr_rows;

            ori_col = col + in2_sub_cumh_sel_collow - 1;
            const int sub_col_offset1 = ori_col * in2_sub_rows;

            ori_col = col + in2_sub_cumh_sel2_collow - 1;
            const int sub_col_offset2 = ori_col * in2_sub_rows;

            for (row = 0; row < in2_sub2_sqr_rows; row++) {

                ori_row = row + in2_sub_cumh_sel_rowlow - 1;
                temp = d_in2_sub[sub_col_offset1 + ori_row];

                ori_row = row + in2_sub_cumh_sel2_rowlow - 1;
                temp2 = d_in2_sub[sub_col_offset2 + ori_row];

                temp2 = temp - temp2;

                d_in2_sub2_sqr[sub2_col_offset + row] = temp2 * temp2;

                d_conv[conv_col_offset + row] =
                    d_conv[conv_col_offset + row] -
                    temp2 * in_final_sum / in_mod_elem;
            }
        }

        for (col = 0; col < in2_pad_cols; col++) {
            const int pad_col_offset = col * in2_pad_rows;
            if (col > (in2_pad_add_cols - 1) && col < (in2_pad_add_cols + in2_cols)) {
                ori_col = col - in2_pad_add_cols;
                const int in2_col_offset = ori_col * in2_rows;
                for (row = 0; row < in2_pad_rows; row++) {
                    if (row > (in2_pad_add_rows - 1) &&
                        row < (in2_pad_add_rows + in2_rows)) {
                        ori_row = row - in2_pad_add_rows;
                        d_in2_pad[pad_col_offset + row] =
                            d_in2_sqr[in2_col_offset + ori_row];
                    } else {
                        d_in2_pad[pad_col_offset + row] = 0;
                    }
                }
            } else {
                for (row = 0; row < in2_pad_rows; row++) {
                    d_in2_pad[pad_col_offset + row] = 0;
                }
            }
        }

        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {

            pos_ori = ei_new * in2_pad_rows;

            sum = 0;
            const int end_pos = pos_ori + in2_pad_rows;
            for (position = pos_ori; position < end_pos; position++) {
                d_in2_pad[position] += sum;
                sum = d_in2_pad[position];
            }
        }

        for (col = 0; col < in2_sub_cols; col++) {
            const int sub_col_offset = col * in2_sub_rows;

            ori_col = col + in2_pad_cumv_sel_rowlow - 1;
            const int pad_col_offset1 = (col + in2_pad_cumv_sel_collow - 1) * in2_pad_rows;

            ori_col = col + in2_pad_cumv_sel2_collow - 1;
            const int pad_col_offset2 = ori_col * in2_pad_rows;

            for (row = 0; row < in2_sub_rows; row++) {

                ori_row = row + in2_pad_cumv_sel_rowlow - 1;
                temp = d_in2_pad[pad_col_offset1 + ori_row];

                ori_row = row + in2_pad_cumv_sel2_rowlow - 1;
                temp2 = d_in2_pad[pad_col_offset2 + ori_row];

                d_in2_sub[sub_col_offset + row] = temp - temp2;
            }
        }

        for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {

            pos_ori = ei_new;

            sum = 0;
            const int end_pos = pos_ori + in2_sub_elem;
            for (position = pos_ori; position < end_pos; position += in2_sub_rows) {
                d_in2_sub[position] += sum;
                sum = d_in2_sub[position];
            }
        }

        for (col = 0; col < conv_cols; col++) {
            const int conv_col_offset = col * conv_rows;

            ori_col = col + in2_sub_cumh_sel_collow - 1;
            const int sub_col_offset1 = ori_col * in2_sub_rows;

            ori_col = col + in2_sub_cumh_sel2_collow - 1;
            const int sub_col_offset2 = ori_col * in2_sub_rows;

            for (row = 0; row < conv_rows; row++) {

                ori_row = row + in2_sub_cumh_sel_rowlow - 1;
                temp = d_in2_sub[sub_col_offset1 + ori_row];

                ori_row = row + in2_sub_cumh_sel2_rowlow - 1;
                temp2 = d_in2_sub[sub_col_offset2 + ori_row];

                temp2 = temp - temp2;

                temp2 = temp2 -
                        (d_in2_sub2_sqr[conv_col_offset + row] / in_mod_elem);

                if (temp2 < 0) {
                    temp2 = 0;
                }
                temp2 = sqrt(temp2);

                temp2 = denomT * temp2;

                d_conv[conv_col_offset + row] =
                    d_conv[conv_col_offset + row] / temp2;
            }
        }

        //====================================================================================================
        //	TEMPLATE MASK CREATE
        //====================================================================================================

        cent = sSize + tSize + 1;
        pointer = public.frame_no - 1 + private.point_no * frames;
        tMask_row =
            cent + (int)d_tRowLoc[pointer] - (int)d_Row[private.point_no] - 1;
        tMask_col =
            cent + (int)d_tColLoc[pointer] - (int)d_Col[private.point_no] - 1;

        for (ei_new = 0; ei_new < tMask_elem; ei_new++) {
            d_tMask[ei_new] = 0;
        }
        d_tMask[tMask_col * tMask_rows + tMask_row] = 1;

        for (col = 1; col <= mask_conv_cols; col++) {

            j = col + mask_conv_joffset;
            jp1 = j + 1;
            ja1 = (mask_cols < jp1) ? (jp1 - mask_cols) : 1;
            ja2 = (tMask_cols < j) ? tMask_cols : j;

            const int mask_conv_col_offset = (col - 1) * conv_rows;
            const int conv_col_offset = (col - 1) * conv_rows;

            for (row = 1; row <= mask_conv_rows; row++) {

                i = row + mask_conv_ioffset;
                ip1 = i + 1;

                ia1 = (mask_rows < ip1) ? (ip1 - mask_rows) : 1;
                ia2 = (tMask_rows < i) ? tMask_rows : i;

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

                d_mask_conv[mask_conv_col_offset + (row - 1)] =
                    d_conv[conv_col_offset + (row - 1)] * s;
            }
        }

        fin_max_val = 0;
        fin_max_coo = 0;
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
        pointer = private.point_no * frames + public.frame_no;
        d_tRowLoc[pointer] = d_Row[private.point_no] + offset_row;
        d_tColLoc[pointer] = d_Col[private.point_no] + offset_col;
    }

    if (public.frame_no != 0 && (public.frame_no) % 10 == 0) {

        loc_pointer = private.point_no * frames + public.frame_no;
        d_Row[private.point_no] = d_tRowLoc[loc_pointer];
        d_Col[private.point_no] = d_tColLoc[loc_pointer];

        d_in = &d_T[private.in_pointer];

        const int point_row = (int)d_Row[private.point_no];
        const int point_col = (int)d_Col[private.point_no];

        for (col = 0; col < in_mod_cols; col++) {
            const int col_offset = col * in_mod_rows;
            const int ori_col_base = point_col - 25 + col - 1;
            const int frame_col_offset = ori_col_base * frame_rows;
            for (row = 0; row < in_mod_rows; row++) {
                ori_row = point_row - 25 + row - 1;
                ori_pointer = frame_col_offset + ori_row;
                const int idx = col_offset + row;
                d_in[idx] = alpha * d_in[idx] +
                            (1.00 - alpha) * d_frame[ori_pointer];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    heartwall_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
