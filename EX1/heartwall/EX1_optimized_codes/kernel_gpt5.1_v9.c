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

        const int frame_rows = public.frame_rows;
        const int in_mod_rows = public.in_mod_rows;

        // update template, limit the number of working threads to the size of
        // template
        for (col = 0; col < public.in_mod_cols; col++) {
            const int col_in_offset = col * in_mod_rows;
            const int base_ori_col = private.d_Col[private.point_no] - 26 + col;
            for (row = 0; row < in_mod_rows; row++) {

                // figure out row/col location in corresponding new template
                // area in image and give to every thread (get top left corner
                // and progress down and right)
                ori_row = private.d_Row[private.point_no] - 26 + row;
                ori_col = base_ori_col;
                ori_pointer = ori_col * frame_rows + ori_row;

                // update template
                d_in[col_in_offset + row] = public.d_frame[ori_pointer];
            }
        }
    }

    //======================================================================================================================================================
    //	PROCESS POINTS
    //======================================================================================================================================================

    // process points in all frames except for the first one
    if (public.frame_no != 0) {
        const int sSize = public.sSize;
        const int in2_rows = public.in2_rows;
        const int in2_cols = public.in2_cols;
        const int in_mod_rows = public.in_mod_rows;
        const int in_mod_cols = public.in_mod_cols;
        const int in2_pad_rows = public.in2_pad_rows;
        const int in2_pad_cols = public.in2_pad_cols;
        const int conv_rows = public.conv_rows;
        const int conv_cols = public.conv_cols;
        const int mask_conv_rows = public.mask_conv_rows;
        const int mask_conv_cols = public.mask_conv_cols;
        const int tMask_rows = public.tMask_rows;
        const int tMask_cols = public.tMask_cols;
        const int in2_sub_rows = public.in2_sub_rows;
        const int in2_sub_cols = public.in2_sub_cols;

        in2_rowlow = private.d_Row[private.point_no] - sSize; // (1 to n+1)
        in2_collow = private.d_Col[private.point_no] - sSize;

        const int frame_rows = public.frame_rows;

        // work
        for (col = 0; col < in2_cols; col++) {
            const int col_in2_offset = col * in2_rows;
            const int base_ori_col = col + in2_collow - 1;
            const int base_frame_col_offset = base_ori_col * frame_rows;
            for (row = 0; row < in2_rows; row++) {

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + in2_rowlow - 1;
                ori_col = base_ori_col;
                temp = public.d_frame[base_frame_col_offset + ori_row];
                private.d_in2[col_in2_offset + row] = temp;
                private.d_in2_sqr[col_in2_offset + row] = temp * temp;
            }
        }

        // variables
        d_in = &private.d_T[private.in_pointer];

        // work
        for (col = 0; col < in_mod_cols; col++) {
            const int col_inmod_offset = col * in_mod_rows;
            const int rot_col_base = (in_mod_rows - 1) - col;
            for (row = 0; row < in_mod_rows; row++) {

                // rotated coordinates
                rot_row = (in_mod_rows - 1) - row;
                rot_col = rot_col_base;
                pointer = rot_col * in_mod_rows + rot_row;

                // execution
                temp = d_in[pointer];
                private.d_in_mod[col_inmod_offset + row] = temp;
                private.d_in_sqr[pointer] = temp * temp;
            }
        }

        in_final_sum = 0;
        for (i = 0; i < public.in_mod_elem; i++) {
            in_final_sum += d_in[i];
        }

        in_sqr_final_sum = 0;
        for (i = 0; i < public.in_mod_elem; i++) {
            in_sqr_final_sum += private.d_in_sqr[i];
        }

        mean =
            in_final_sum /
            public.in_mod_elem; // gets mean (average) value of element in ROI
        mean_sqr = mean * mean;
        variance = (in_sqr_final_sum / public.in_mod_elem) -
                   mean_sqr;        // gets variance of ROI
        deviation = sqrt(variance); // gets standard deviation of ROI

        denomT = sqrt((fp)(public.in_mod_elem - 1)) * deviation;

        // work
        const int in2_rows_local = in2_rows;
        const int in2_cols_local = in2_cols;
        for (col = 1; col <= conv_cols; col++) {

            // column setup
            j = col + public.joffset;
            jp1 = j + 1;
            if (in2_cols_local < jp1) {
                ja1 = jp1 - in2_cols_local;
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
                i = row + public.ioffset;
                ip1 = i + 1;

                if (in2_rows_local < ip1) {
                    ia1 = ip1 - in2_rows_local;
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
                const int in_mod_rows_local = in_mod_rows;
                const int in2_rows_local2 = in2_rows_local;
                for (ja = ja1; ja <= ja2; ja++) {
                    jb = jp1 - ja;
                    const int in_mod_base = in_mod_rows_local * (ja - 1);
                    const int in2_base = in2_rows_local2 * (jb - 1);
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        s += private.d_in_mod[in_mod_base + ia - 1] *
                             private.d_in2[in2_base + ib - 1];
                    }
                }

                private.d_conv[(col - 1) * conv_rows + (row - 1)] = s;
            }
        }

        // work
        const int in2_pad_add_rows = public.in2_pad_add_rows;
        const int in2_pad_add_cols = public.in2_pad_add_cols;
        for (col = 0; col < in2_pad_cols; col++) {
            const int col_pad_offset = col * in2_pad_rows;
            for (row = 0; row < in2_pad_rows; row++) {

                // execution
                if (row > (in2_pad_add_rows - 1) && // do if has numbers in original array
                    row < (in2_pad_add_rows + in2_rows) &&
                    col > (in2_pad_add_cols - 1) &&
                    col < (in2_pad_add_cols + in2_cols)) {
                    ori_row = row - in2_pad_add_rows;
                    ori_col = col - in2_pad_add_cols;
                    private.d_in2_pad[col_pad_offset + row] =
                        private.d_in2[ori_col * in2_rows + ori_row];
                } else { // do if otherwise
                    private.d_in2_pad[col_pad_offset + row] = 0;
                }
            }
        }

        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {

            // figure out column position
            pos_ori = ei_new * in2_pad_rows;

            // loop through all rows
            sum = 0;
            location = pos_ori + in2_pad_rows;
            for (position = pos_ori; position < location; position++) {
                private.d_in2_pad[position] = private.d_in2_pad[position] + sum;
                sum = private.d_in2_pad[position];
            }
        }

        // work
        const int in2_pad_cumv_sel_rowlow = public.in2_pad_cumv_sel_rowlow;
        const int in2_pad_cumv_sel_collow = public.in2_pad_cumv_sel_collow;
        const int in2_pad_cumv_sel2_rowlow = public.in2_pad_cumv_sel2_rowlow;
        const int in2_pad_cumv_sel2_collow = public.in2_pad_cumv_sel2_collow;
        for (col = 0; col < in2_sub_cols; col++) {
            const int col_sub_offset = col * in2_sub_rows;
            const int col_sel = col + in2_pad_cumv_sel_collow - 1;
            const int col_sel2 = col + in2_pad_cumv_sel2_collow - 1;
            const int base_sel = col_sel * in2_pad_rows;
            const int base_sel2 = col_sel2 * in2_pad_rows;
            for (row = 0; row < in2_sub_rows; row++) {

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + in2_pad_cumv_sel_rowlow - 1;
                ori_col = col_sel;
                temp = private.d_in2_pad[base_sel + ori_row];

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + in2_pad_cumv_sel2_rowlow - 1;
                ori_col = col_sel2;
                temp2 = private.d_in2_pad[base_sel2 + ori_row];

                // subtraction
                private.d_in2_sub[col_sub_offset + row] = temp - temp2;
            }
        }

        const int in2_sub_elem = public.in2_sub_elem;
        for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {

            // figure out row position
            pos_ori = ei_new;

            // loop through all rows
            sum = 0;
            location = pos_ori + in2_sub_elem;
            for (position = pos_ori; position < location;
                 position += in2_sub_rows) {
                private.d_in2_sub[position] = private.d_in2_sub[position] + sum;
                sum = private.d_in2_sub[position];
            }
        }

        // work
        const int in2_sub_cumh_sel_rowlow = public.in2_sub_cumh_sel_rowlow;
        const int in2_sub_cumh_sel_collow = public.in2_sub_cumh_sel_collow;
        const int in2_sub_cumh_sel2_rowlow = public.in2_sub_cumh_sel2_rowlow;
        const int in2_sub_cumh_sel2_collow = public.in2_sub_cumh_sel2_collow;
        const int in2_sub2_sqr_rows = public.in2_sub2_sqr_rows;
        const int in2_sub2_sqr_cols = public.in2_sub2_sqr_cols;
        for (col = 0; col < in2_sub2_sqr_cols; col++) {
            const int col_sqr_offset = col * in2_sub2_sqr_rows;
            const int col_sel = col + in2_sub_cumh_sel_collow - 1;
            const int col_sel2 = col + in2_sub_cumh_sel2_collow - 1;
            const int base_sel = col_sel * in2_sub_rows;
            const int base_sel2 = col_sel2 * in2_sub_rows;
            const int conv_base = col * in2_sub2_sqr_rows;
            for (row = 0; row < in2_sub2_sqr_rows; row++) {

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + in2_sub_cumh_sel_rowlow - 1;
                ori_col = col_sel;
                temp = private.d_in2_sub[base_sel + ori_row];

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + in2_sub_cumh_sel2_rowlow - 1;
                ori_col = col_sel2;
                temp2 = private.d_in2_sub[base_sel2 + ori_row];

                // subtraction
                temp2 = temp - temp2;

                // squaring
                private.d_in2_sub2_sqr[col_sqr_offset + row] = temp2 * temp2;

                // numerator
                private.d_conv[conv_base + row] =
                    private.d_conv[conv_base + row] -
                    temp2 * in_final_sum / public.in_mod_elem;
            }
        }

        // work
        for (col = 0; col < in2_pad_cols; col++) {
            const int col_pad_offset = col * in2_pad_rows;
            for (row = 0; row < in2_pad_rows; row++) {

                // execution
                if (row > (in2_pad_add_rows - 1) && // do if has numbers in original array
                    row < (in2_pad_add_rows + in2_rows) &&
                    col > (in2_pad_add_cols - 1) &&
                    col < (in2_pad_add_cols + in2_cols)) {
                    ori_row = row - in2_pad_add_rows;
                    ori_col = col - in2_pad_add_cols;
                    private.d_in2_pad[col_pad_offset + row] =
                        private.d_in2_sqr[ori_col * in2_rows + ori_row];
                } else { // do if otherwise
                    private.d_in2_pad[col_pad_offset + row] = 0;
                }
            }
        }

        // work
        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {

            // figure out column position
            pos_ori = ei_new * in2_pad_rows;

            // loop through all rows
            sum = 0;
            location = pos_ori + in2_pad_rows;
            for (position = pos_ori; position < location; position++) {
                private.d_in2_pad[position] = private.d_in2_pad[position] + sum;
                sum = private.d_in2_pad[position];
            }
        }

        // work
        for (col = 0; col < in2_sub_cols; col++) {
            const int col_sub_offset = col * in2_sub_rows;
            const int col_sel = col + in2_pad_cumv_sel_rowlow - 1;
            const int col_sel2 = col + in2_pad_cumv_sel2_collow - 1;
            const int base_sel = col_sel * in2_pad_rows;
            const int base_sel2 = col_sel2 * in2_pad_rows;
            for (row = 0; row < in2_sub_rows; row++) {

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + in2_pad_cumv_sel_rowlow - 1;
                ori_col = col_sel;
                temp = private.d_in2_pad[base_sel + ori_row];

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + in2_pad_cumv_sel2_rowlow - 1;
                ori_col = col_sel2;
                temp2 = private.d_in2_pad[base_sel2 + ori_row];

                // subtract
                private.d_in2_sub[col_sub_offset + row] = temp - temp2;
            }
        }

        for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {

            // figure out row position
            pos_ori = ei_new;

            // loop through all rows
            sum = 0;
            location = pos_ori + in2_sub_elem;
            for (position = pos_ori; position < location;
                 position += in2_sub_rows) {
                private.d_in2_sub[position] = private.d_in2_sub[position] + sum;
                sum = private.d_in2_sub[position];
            }
        }

        // work
        for (col = 0; col < conv_cols; col++) {
            const int col_conv_offset = col * conv_rows;
            const int col_sel = col + in2_sub_cumh_sel_collow - 1;
            const int col_sel2 = col + in2_sub_cumh_sel2_collow - 1;
            const int base_sel = col_sel * in2_sub_rows;
            const int base_sel2 = col_sel2 * in2_sub_rows;
            for (row = 0; row < conv_rows; row++) {

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + in2_sub_cumh_sel_rowlow - 1;
                ori_col = col_sel;
                temp = private.d_in2_sub[base_sel + ori_row];

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + in2_sub_cumh_sel2_rowlow - 1;
                ori_col = col_sel2;
                temp2 = private.d_in2_sub[base_sel2 + ori_row];

                // subtract
                temp2 = temp - temp2;

                // diff_local_sums
                temp2 = temp2 -
                        (private.d_in2_sub2_sqr[col_conv_offset + row] /
                         public.in_mod_elem);

                // denominator A
                if (temp2 < 0) {
                    temp2 = 0;
                }
                temp2 = sqrt(temp2);

                // denominator
                temp2 = denomT * temp2;

                // correlation
                private.d_conv[col_conv_offset + row] =
                    private.d_conv[col_conv_offset + row] / temp2;
            }
        }

        //====================================================================================================
        //	TEMPLATE MASK CREATE
        //====================================================================================================

        // parameters
        cent = sSize + public.tSize + 1;
        pointer = public.frame_no - 1 + private.point_no * public.frames;
        tMask_row = cent + private.d_tRowLoc[pointer] -
                    private.d_Row[private.point_no] - 1;
        tMask_col = cent + private.d_tColLoc[pointer] -
                    private.d_Col[private.point_no] - 1;

        // work
        for (ei_new = 0; ei_new < public.tMask_elem; ei_new++) {
            private.d_tMask[ei_new] = 0;
        }
        private.d_tMask[tMask_col * tMask_rows + tMask_row] = 1;

        // work
        for (col = 1; col <= mask_conv_cols; col++) {

            // col setup
            j = col + public.mask_conv_joffset;
            jp1 = j + 1;
            if (public.mask_cols < jp1) {
                ja1 = jp1 - public.mask_cols;
            } else {
                ja1 = 1;
            }
            if (tMask_cols < j) {
                ja2 = tMask_cols;
            } else {
                ja2 = j;
            }

            for (row = 1; row <= mask_conv_rows; row++) {

                // row setup
                i = row + public.mask_conv_ioffset;
                ip1 = i + 1;

                if (public.mask_rows < ip1) {
                    ia1 = ip1 - public.mask_rows;
                } else {
                    ia1 = 1;
                }
                if (tMask_rows < i) {
                    ia2 = tMask_rows;
                } else {
                    ia2 = i;
                }

                s = 0;

                // get data
                const int tMask_rows_local = tMask_rows;
                for (ja = ja1; ja <= ja2; ja++) {
                    jb = jp1 - ja;
                    (void)jb; // jb is not used but kept for structural parity
                    const int base_tmask = tMask_rows_local * (ja - 1);
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        (void)ib; // ib is not used but kept for structural parity
                        s += private.d_tMask[base_tmask + ia - 1];
                    }
                }

                private.d_mask_conv[(col - 1) * conv_rows + (row - 1)] =
                    private.d_conv[(col - 1) * conv_rows + (row - 1)] * s;
            }
        }

        fin_max_val = 0;
        fin_max_coo = 0;
        for (i = 0; i < public.mask_conv_elem; i++) {
            if (private.d_mask_conv[i] > fin_max_val) {
                fin_max_val = private.d_mask_conv[i];
                fin_max_coo = i;
            }
        }

        // convert coordinate to row/col form
        largest_row =
            (fin_max_coo + 1) % mask_conv_rows - 1;       // (0-n) row
        largest_col = (fin_max_coo + 1) / mask_conv_rows; // (0-n) column
        if ((fin_max_coo + 1) % mask_conv_rows == 0) {
            largest_row = mask_conv_rows - 1;
            largest_col = largest_col - 1;
        }

        // calculate offset
        largest_row =
            largest_row + 1; // compensate to match MATLAB format (1-n)
        largest_col =
            largest_col + 1; // compensate to match MATLAB format (1-n)
        offset_row =
            largest_row - in_mod_rows - (sSize - public.tSize);
        offset_col =
            largest_col - in_mod_cols - (sSize - public.tSize);
        pointer = private.point_no * public.frames + public.frame_no;
        private.d_tRowLoc[pointer] = private.d_Row[private.point_no] + offset_row;
        private.d_tColLoc[pointer] = private.d_Col[private.point_no] + offset_col;
    }

    // if the last frame in the bath, update template
    if (public.frame_no != 0 && (public.frame_no) % 10 == 0) {

        // update coordinate
        loc_pointer = private.point_no * public.frames + public.frame_no;
        private.d_Row[private.point_no] = private.d_tRowLoc[loc_pointer];
        private.d_Col[private.point_no] = private.d_tColLoc[loc_pointer];

        // pointers to: current frame, template for current point
        d_in = &private.d_T[private.in_pointer];

        const int frame_rows = public.frame_rows;
        const int in_mod_rows = public.in_mod_rows;

        // update template, limit the number of working threads to the size of
        // template
        for (col = 0; col < public.in_mod_cols; col++) {
            const int col_in_offset = col * in_mod_rows;
            const int base_ori_col = private.d_Col[private.point_no] - 26 + col;
            for (row = 0; row < in_mod_rows; row++) {

                // figure out row/col location in corresponding new template
                // area in image and give to every thread (get top left corner
                // and progress down and right)
                ori_row = private.d_Row[private.point_no] - 26 + row;
                ori_col = base_ori_col;
                ori_pointer = ori_col * frame_rows + ori_row;

                // update template
                const fp old_val = d_in[col_in_offset + row];
                const fp new_val = public.d_frame[ori_pointer];
                d_in[col_in_offset + row] =
                    public.alpha * old_val +
                    (1.00 - public.alpha) * new_val;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    heartwall_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
