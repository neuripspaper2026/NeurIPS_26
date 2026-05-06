<<<CODE>>>
#include <time.h>

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

        int base_row = private.d_Row[private.point_no] - 26;
        int base_col = private.d_Col[private.point_no] - 26;
        fp *frame = public.d_frame;
        int frame_rows = public.frame_rows;
        int in_mod_rows = public.in_mod_rows;

        // update template, limit the number of working threads to the size of
        // template
        for (col = 0; col < public.in_mod_cols; col++) {
            int col_offset = (base_col + col) * frame_rows;
            int d_in_col_offset = col * in_mod_rows;
            for (row = 0; row < in_mod_rows; row++) {
                d_in[d_in_col_offset + row] = frame[col_offset + base_row + row];
            }
        }
    }

    //======================================================================================================================================================
    //	PROCESS POINTS
    //======================================================================================================================================================

    // process points in all frames except for the first one
    if (public.frame_no != 0) {
        in2_rowlow = private.d_Row[private.point_no] - public.sSize;
        in2_collow = private.d_Col[private.point_no] - public.sSize;

        int base_row = in2_rowlow - 1;
        int base_col = in2_collow - 1;
        fp *frame = public.d_frame;
        int frame_rows = public.frame_rows;
        int in2_rows = public.in2_rows;

        // work
        for (col = 0; col < public.in2_cols; col++) {
            int col_offset = (base_col + col) * frame_rows;
            int d_in2_col_offset = col * in2_rows;
            for (row = 0; row < in2_rows; row++) {
                temp = frame[col_offset + base_row + row];
                private.d_in2[d_in2_col_offset + row] = temp;
                private.d_in2_sqr[d_in2_col_offset + row] = temp * temp;
            }
        }

        // variables
        d_in = &private.d_T[private.in_pointer];

        int in_mod_rows = public.in_mod_rows;
        int in_mod_rows_m1 = in_mod_rows - 1;

        // work
        for (col = 0; col < public.in_mod_cols; col++) {
            int rot_col = in_mod_rows_m1 - col;
            int rot_col_offset = rot_col * in_mod_rows;
            int col_offset = col * in_mod_rows;
            for (row = 0; row < in_mod_rows; row++) {
                int rot_row = in_mod_rows_m1 - row;
                pointer = rot_col_offset + rot_row;
                temp = d_in[pointer];
                private.d_in_mod[col_offset + row] = temp;
                private.d_in_sqr[pointer] = temp * temp;
            }
        }

        in_final_sum = 0;
        for (i = 0; i < public.in_mod_elem; i++) {
            in_final_sum += d_in[i];
        }

        in_sqr_final_sum = 0;
        fp *d_in_sqr = private.d_in_sqr;
        for (i = 0; i < public.in_mod_elem; i++) {
            in_sqr_final_sum += d_in_sqr[i];
        }

        fp inv_in_mod_elem = 1.0 / public.in_mod_elem;
        mean = in_final_sum * inv_in_mod_elem;
        mean_sqr = mean * mean;
        variance = (in_sqr_final_sum * inv_in_mod_elem) - mean_sqr;
        deviation = sqrt(variance);

        denomT = sqrt((fp)(public.in_mod_elem - 1)) * deviation;

        int conv_rows = public.conv_rows;
        int joffset = public.joffset;
        int ioffset = public.ioffset;
        int in2_cols = public.in2_cols;
        int in2_rows = public.in2_rows;
        int in_mod_cols = public.in_mod_cols;
        fp *d_in_mod = private.d_in_mod;
        fp *d_in2 = private.d_in2;

        // work
        for (col = 1; col <= public.conv_cols; col++) {

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

            int conv_col_offset = (col - 1) * conv_rows;

            for (row = 1; row <= conv_rows; row++) {

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
                    int ja_offset = in_mod_rows * (ja - 1);
                    int jb_offset = in2_rows * (jb - 1);
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        s += d_in_mod[ja_offset + ia - 1] * d_in2[jb_offset + ib - 1];
                    }
                }

                private.d_conv[conv_col_offset + (row - 1)] = s;
            }
        }

        int in2_pad_rows = public.in2_pad_rows;
        int in2_pad_add_rows = public.in2_pad_add_rows;
        int in2_pad_add_cols = public.in2_pad_add_cols;
        int in2_pad_add_rows_plus_in2_rows = in2_pad_add_rows + in2_rows;
        int in2_pad_add_cols_plus_in2_cols = in2_pad_add_cols + public.in2_cols;

        // work
        for (col = 0; col < public.in2_pad_cols; col++) {
            int col_offset = col * in2_pad_rows;
            if (col > (in2_pad_add_cols - 1) && col < in2_pad_add_cols_plus_in2_cols) {
                ori_col = col - in2_pad_add_cols;
                int ori_col_offset = ori_col * in2_rows;
                for (row = 0; row < public.in2_pad_rows; row++) {
                    if (row > (in2_pad_add_rows - 1) && row < in2_pad_add_rows_plus_in2_rows) {
                        ori_row = row - in2_pad_add_rows;
                        private.d_in2_pad[col_offset + row] = d_in2[ori_col_offset + ori_row];
                    } else {
                        private.d_in2_pad[col_offset + row] = 0;
                    }
                }
            } else {
                for (row = 0; row < in2_pad_rows; row++) {
                    private.d_in2_pad[col_offset + row] = 0;
                }
            }
        }

        fp *d_in2_pad = private.d_in2_pad;

        for (ei_new = 0; ei_new < public.in2_pad_cols; ei_new++) {

            // figure out column position
            pos_ori = ei_new * in2_pad_rows;

            // loop through all rows
            sum = 0;
            int end_pos = pos_ori + in2_pad_rows;
            for (position = pos_ori; position < end_pos; position++) {
                d_in2_pad[position] += sum;
                sum = d_in2_pad[position];
            }
        }

        int in2_pad_cumv_sel_rowlow_m1 = public.in2_pad_cumv_sel_rowlow - 1;
        int in2_pad_cumv_sel_collow_m1 = public.in2_pad_cumv_sel_collow - 1;
        int in2_pad_cumv_sel2_rowlow_m1 = public.in2_pad_cumv_sel2_rowlow - 1;
        int in2_pad_cumv_sel2_collow_m1 = public.in2_pad_cumv_sel2_collow - 1;
        int in2_sub_rows = public.in2_sub_rows;

        // work
        for (col = 0; col < public.in2_sub_cols; col++) {
            int ori_col1 = col + in2_pad_cumv_sel_collow_m1;
            int ori_col2 = col + in2_pad_cumv_sel2_collow_m1;
            int ori_col1_offset = ori_col1 * in2_pad_rows;
            int ori_col2_offset = ori_col2 * in2_pad_rows;
            int col_offset = col * in2_sub_rows;
            for (row = 0; row < in2_sub_rows; row++) {
                ori_row = row + in2_pad_cumv_sel_rowlow_m1;
                temp = d_in2_pad[ori_col1_offset + ori_row];

                ori_row = row + in2_pad_cumv_sel2_rowlow_m1;
                temp2 = d_in2_pad[ori_col2_offset + ori_row];

                private.d_in2_sub[col_offset + row] = temp - temp2;
            }
        }

        fp *d_in2_sub = private.d_in2_sub;

        for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {

            // figure out row position
            pos_ori = ei_new;

            // loop through all rows
            sum = 0;
            int end_pos = pos_ori + public.in2_sub_elem;
            for (position = pos_ori; position < end_pos; position += in2_sub_rows) {
                d_in2_sub[position] += sum;
                sum = d_in2_sub[position];
            }
        }

        int in2_sub_cumh_sel_rowlow_m1 = public.in2_sub_cumh_sel_rowlow - 1;
        int in2_sub_cumh_sel_collow_m1 = public.in2_sub_cumh_sel_collow - 1;
        int in2_sub_cumh_sel2_rowlow_m1 = public.in2_sub_cumh_sel2_rowlow - 1;
        int in2_sub_cumh_sel2_collow_m1 = public.in2_sub_cumh_sel2_collow - 1;
        fp in_final_sum_div_in_mod_elem = in_final_sum * inv_in_mod_elem;
        fp *d_conv = private.d_conv;

        // work
        for (col = 0; col < public.in2_sub2_sqr_cols; col++) {
            int ori_col1 = col + in2_sub_cumh_sel_collow_m1;
            int ori_col2 = col + in2_sub_cumh_sel2_collow_m1;
            int ori_col1_offset = ori_col1 * in2_sub_rows;
            int ori_col2_offset = ori_col2 * in2_sub_rows;
            int col_offset = col * public.in2_sub2_sqr_rows;
            for (row = 0; row < public.in2_sub2_sqr_rows; row++) {
                ori_row = row + in2_sub_cumh_sel_rowlow_m1;
                temp = d_in2_sub[ori_col1_offset + ori_row];

                ori_row = row + in2_sub_cumh_sel2_rowlow_m1;
                temp2 = d_in2_sub[ori_col2_offset + ori_row];

                temp2 = temp - temp2;

                int idx = col_offset + row;
                private.d_in2_sub2_sqr[idx] = temp2 * temp2;

                d_conv[idx] -= temp2 * in_final_sum_div_in_mod_elem;
            }
        }

        // work
        for
