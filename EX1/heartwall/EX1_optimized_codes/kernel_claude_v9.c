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

    const int frame_no = public.frame_no;
    const int point_no = private.point_no;
    const int frames = public.frames;
    const int in_mod_cols = public.in_mod_cols;
    const int in_mod_rows = public.in_mod_rows;
    const int in_mod_elem = public.in_mod_elem;
    const int frame_rows = public.frame_rows;
    const int in2_cols = public.in2_cols;
    const int in2_rows = public.in2_rows;
    const int conv_cols = public.conv_cols;
    const int conv_rows = public.conv_rows;
    const int in2_pad_cols = public.in2_pad_cols;
    const int in2_pad_rows = public.in2_pad_rows;
    const int in2_sub_cols = public.in2_sub_cols;
    const int in2_sub_rows = public.in2_sub_rows;
    const int in2_sub_elem = public.in2_sub_elem;
    const int in2_sub2_sqr_cols = public.in2_sub2_sqr_cols;
    const int in2_sub2_sqr_rows = public.in2_sub2_sqr_rows;
    const int tMask_elem = public.tMask_elem;
    const int tMask_rows = public.tMask_rows;
    const int tMask_cols = public.tMask_cols;
    const int mask_conv_cols = public.mask_conv_cols;
    const int mask_conv_rows = public.mask_conv_rows;
    const int mask_conv_elem = public.mask_conv_elem;
    const int sSize = public.sSize;
    const int tSize = public.tSize;
    const fp alpha = public.alpha;

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

        const int row_offset = private.d_Row[point_no] - 26;
        const int col_offset = private.d_Col[point_no] - 26;

        // update template, limit the number of working threads to the size of
        // template
        for (col = 0; col < in_mod_cols; col++) {
            const int ori_col_base = col_offset + col;
            const int col_base = col * in_mod_rows;
            const int ori_col_base_fr = ori_col_base * frame_rows;
            for (row = 0; row < in_mod_rows; row++) {

                // figure out row/col location in corresponding new template
                // area in image and give to every thread (get top left corner
                // and progress down and right)
                ori_row = row_offset + row;
                ori_pointer = ori_col_base_fr + ori_row;

                // update template
                d_in[col_base + row] = public.d_frame[ori_pointer];
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

        const int in2_rowlow_offset = in2_rowlow - 1;
        const int in2_collow_offset = in2_collow - 1;

        // work
        for (col = 0; col < in2_cols; col++) {
            const int ori_col_base = col + in2_collow_offset;
            const int col_base = col * in2_rows;
            const int ori_col_base_fr = ori_col_base * frame_rows;
            for (row = 0; row < in2_rows; row++) {

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + in2_rowlow_offset;
                temp = public.d_frame[ori_col_base_fr + ori_row];
                private.d_in2[col_base + row] = temp;
                private.d_in2_sqr[col_base + row] = temp * temp;
            }
        }

        // variables
        d_in = &private.d_T[private.in_pointer];

        const int in_mod_rows_m1 = in_mod_rows - 1;

        // work
        for (col = 0; col < in_mod_cols; col++) {
            const int rot_col_base = (in_mod_rows_m1 - col) * in_mod_rows;
            const int col_base = col * in_mod_rows;
            for (row = 0; row < in_mod_rows; row++) {

                // rotated coordinates
                rot_row = in_mod_rows_m1 - row;
                pointer = rot_col_base + rot_row;

                // execution
                temp = d_in[pointer];
                private.d_in_mod[col_base + row] = temp;
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

        mean = in_final_sum / in_mod_elem; // gets mean (average) value of element in ROI
        mean_sqr = mean * mean;
        variance = (in_sqr_final_sum / in_mod_elem) - mean_sqr;        // gets variance of ROI
        deviation = sqrt(variance); // gets standard deviation of ROI

        denomT = sqrt((fp)(in_mod_elem - 1)) * deviation;

        const int joffset = public.joffset;
        const int ioffset = public.ioffset;

        // work
        for (col = 1; col <= conv_cols; col++) {

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
                    const int ja_base = in_mod_rows * (ja - 1);
                    const int jb_base = in2_rows * (jb - 1);
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        s += private.d_in_mod[ja_base + ia - 1] * private.d_in2[jb_base + ib - 1];
                    }
                }

                private.d_conv[(col - 1) * conv_rows + (row - 1)] = s;
            }
        }

        const int in2_pad_add_rows = public.in2_pad_add_rows;
        const int in2_pad_add_cols = public.in2_pad_add_cols;
        const int in2_pad_add_rows_plus_in2_rows = in2_pad_add_rows + in2_rows;
        const int in2_pad_add_cols_plus_in2_cols = in2_pad_add_cols + in2_cols;

        // work
        for (col = 0; col < in2_pad_cols; col++) {
            const int col_base = col * in2_pad_rows;
            const int in_bounds_col = (col > (in2_pad_add_cols - 1) && col < in2_pad_add_cols_plus_in2_cols);
            const int ori_col_base = in_bounds_col ? (col - in2_pad_add_cols) * in2_rows : 0;
            for (row = 0; row < in2_pad_rows; row++) {

                // execution
                if (in_bounds_col && row > (in2_pad_add_rows - 1) && row < in2_pad_add_rows_plus_in2_rows) {
                    ori_row = row - in2_pad_add_rows;
                    private.d_in2_pad[col_base + row] = private.d_in2[ori_col_base + ori_row];
                } else { // do if otherwise
                    private.d_in2_pad[col_base + row] = 0;
                }
            }
        }

        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {

            // figure out column position
            pos_ori = ei_new * in2_pad_rows;

            // loop through all rows
            sum = 0;
            for (position = pos_ori; position < pos_ori + in2_pad_rows; position++) {
                private.d_in2_pad[position] += sum;
                sum = private.d_in2_pad[position];
            }
        }

        const int in2_pad_cumv_sel_rowlow = public.in2_pad_cumv_sel_rowlow;
        const int in2_pad_cumv_sel_collow = public.in2_pad_cumv_sel_collow;
        const int in2_pad_cumv_sel2_rowlow = public.in2_pad_cumv_sel2_rowlow;
        const int in2_pad_cumv_sel2_collow = public.in2_pad_cumv_sel2_collow;

        // work
        for (col = 0; col < in2_sub_cols; col++) {
            const int ori_col_base1 = (col + in2_pad_cumv_sel_collow - 1) * in2_pad_rows;
            const int ori_col_base2 = (col + in2_pad_cumv_sel2_collow - 1) * in2_pad_rows;
            const int col_base = col * in2_sub_rows;
            for (row = 0; row < in2_sub_rows; row++) {

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + in2_pad_cumv_sel_rowlow - 1;
                temp = private.d_in2_pad[ori_col_base1 + ori_row];

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + in2_pad_cumv_sel2_rowlow - 1;
                temp2 = private.d_in2_pad[ori_col_base2 + ori_row];

                // subtraction
                private.d_in2_sub[col_base + row] = temp - temp2;
            }
        }

        for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {

            // figure out row position
            pos_ori = ei_new;

            // loop through all rows
            sum = 0;
            for (position = pos_ori; position < pos_ori + in2_sub_elem; position += in2_sub_rows) {
                private.d_in2_sub[position] += sum;
                sum = private.d_in2_sub[position];
            }
        }

        const int in2_sub_cumh_sel_rowlow = public.in2_sub_cumh_sel_rowlow;
        const int in2_sub_cumh_sel_collow = public.in2_sub_cumh_sel_collow;
        const int in2_sub_cumh_sel2_rowlow = public.in2_sub_cumh_sel2_rowlow;
        const int in2_sub_cumh_sel2_collow = public.in2_sub_cumh_sel2_collow;
        const fp in_final_sum_div_in_mod_elem = in_final_sum / in_mod_elem;

        // work
        for (col = 0; col < in2_sub2_sqr_cols; col++) {
            const int ori_col_base1 = (col + in2_sub_cumh_sel_collow - 1) * in2_sub_rows;
            const int ori_col_base2 = (col + in2_sub_cumh_sel2_collow - 1) * in2_sub_rows;
            const int col_base = col * in2_sub2_sqr_rows;
            for (row = 0; row < in2_sub2_sqr_rows; row++) {

                // figure out corresponding location in old matrix and copy
                // values to new
