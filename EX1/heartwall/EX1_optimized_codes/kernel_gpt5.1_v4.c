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

        const int in_mod_rows = public.in_mod_rows;
        const int in_mod_cols = public.in_mod_cols;
        const int frame_rows = public.frame_rows;
        const int base_row = private.d_Row[private.point_no] - 26;
        const int base_col = private.d_Col[private.point_no] - 26;

        // update template, limit the number of working threads to the size of
        // template
        for (col = 0; col < in_mod_cols; col++) {
            const int ori_col_base = (base_col + col) * frame_rows + base_row;
            const int d_in_col_offset = col * in_mod_rows;
            for (row = 0; row < in_mod_rows; row++) {
                ori_pointer = ori_col_base + row;
                d_in[d_in_col_offset + row] = public.d_frame[ori_pointer];
            }
        }
    } else {

        //======================================================================================================================================================
        //	PROCESS POINTS
        //======================================================================================================================================================

        in2_rowlow = private.d_Row[private.point_no] - public.sSize; // (1 to n+1)
        in2_collow = private.d_Col[private.point_no] - public.sSize;

        const int in2_rows = public.in2_rows;
        const int in2_cols = public.in2_cols;
        const int frame_rows = public.frame_rows;

        // work
        for (col = 0; col < in2_cols; col++) {
            const int ori_col_base = (col + in2_collow - 1) * frame_rows + (in2_rowlow - 1);
            const int d_in2_col_offset = col * in2_rows;
            for (row = 0; row < in2_rows; row++) {
                ori_pointer = ori_col_base + row;
                temp = public.d_frame[ori_pointer];
                location = d_in2_col_offset + row;
                private.d_in2[location] = temp;
                private.d_in2_sqr[location] = temp * temp;
            }
        }

        // variables
        d_in = &private.d_T[private.in_pointer];

        const int in_mod_rows = public.in_mod_rows;
        const int in_mod_cols = public.in_mod_cols;
        const int in_mod_elem = public.in_mod_elem;

        // work
        for (col = 0; col < in_mod_cols; col++) {
            const int col_offset = col * in_mod_rows;
            const int rot_col = (in_mod_rows - 1) - col;
            const int rot_col_base = rot_col * in_mod_rows;
            for (row = 0; row < in_mod_rows; row++) {
                rot_row = (in_mod_rows - 1) - row;
                pointer = rot_col_base + rot_row;
                temp = d_in[pointer];
                location = col_offset + row;
                private.d_in_mod[location] = temp;
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
        variance = (in_sqr_final_sum / in_mod_elem) - mean_sqr; // gets variance of ROI
        deviation = sqrt(variance);                             // gets standard deviation of ROI

        denomT = sqrt((fp)(in_mod_elem - 1)) * deviation;

        const int conv_cols = public.conv_cols;
        const int conv_rows = public.conv_rows;
        const int in2_pad_rows = public.in2_pad_rows;
        const int in2_pad_cols = public.in2_pad_cols;
        const int in2_sub_rows = public.in2_sub_rows;
        const int in2_sub_cols = public.in2_sub_cols;
        const int in2_sub_elem = public.in2_sub_elem;
        const int in2_sub2_sqr_rows = public.in2_sub2_sqr_rows;
        const int in2_sub2_sqr_cols = public.in2_sub2_sqr_cols;

        const int in2_pad_add_rows = public.in2_pad_add_rows;
        const int in2_pad_add_cols = public.in2_pad_add_cols;
        const int in2_rows_local = public.in2_rows;
        const int in2_cols_local = public.in2_cols;
        const int in2_pad_rows_local = public.in2_pad_rows;

        const int in2_pad_cumv_sel_rowlow = public.in2_pad_cumv_sel_rowlow;
        const int in2_pad_cumv_sel_collow = public.in2_pad_cumv_sel_collow;
        const int in2_pad_cumv_sel2_rowlow = public.in2_pad_cumv_sel2_rowlow;
        const int in2_pad_cumv_sel2_collow = public.in2_pad_cumv_sel2_collow;

        const int in2_sub_cumh_sel_rowlow = public.in2_sub_cumh_sel_rowlow;
        const int in2_sub_cumh_sel_collow = public.in2_sub_cumh_sel_collow;
        const int in2_sub_cumh_sel2_rowlow = public.in2_sub_cumh_sel2_rowlow;
        const int in2_sub_cumh_sel2_collow = public.in2_sub_cumh_sel2_collow;

        const int joffset = public.joffset;
        const int ioffset = public.ioffset;

        const int in2_cols_local2 = public.in2_cols;
        const int in2_rows_local2 = public.in2_rows;

        // work
        for (col = 1; col <= conv_cols; col++) {

            // column setup
            j = col + joffset;
            jp1 = j + 1;
            ja1 = (in2_cols_local2 < jp1) ? (jp1 - in2_cols_local2) : 1;
            ja2 = (in_mod_cols < j) ? in_mod_cols : j;

            for (row = 1; row <= conv_rows; row++) {

                // row range setup
                i = row + ioffset;
                ip1 = i + 1;

                ia1 = (in2_rows_local2 < ip1) ? (ip1 - in2_rows_local2) : 1;
                ia2 = (in_mod_rows < i) ? in_mod_rows : i;

                s = 0;

                // getting data
                for (ja = ja1; ja <= ja2; ja++) {
                    jb = jp1 - ja;
                    const int in_mod_col_base = (ja - 1) * in_mod_rows;
                    const int in2_col_base = (jb - 1) * in2_rows_local2;
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        s += private.d_in_mod[in_mod_col_base + (ia - 1)] *
                             private.d_in2[in2_col_base + (ib - 1)];
                    }
                }

                private.d_conv[(col - 1) * conv_rows + (row - 1)] = s;
            }
        }

        // work
        for (col = 0; col < in2_pad_cols; col++) {
            const int col_base_pad = col * in2_pad_rows_local;
            const int col_shift = col - in2_pad_add_cols;
            const int in2_col_base = col_shift * in2_rows_local;
            for (row = 0; row < in2_pad_rows_local; row++) {

                // execution
                if (row > (in2_pad_add_rows - 1) && // do if has numbers in original array
                    row < (in2_pad_add_rows + in2_rows_local) && col > (in2_pad_add_cols - 1) &&
                    col < (in2_pad_add_cols + in2_cols_local)) {
                    ori_row = row - in2_pad_add_rows;
                    ori_pointer = in2_col_base + ori_row;
                    private.d_in2_pad[col_base_pad + row] = private.d_in2[ori_pointer];
                } else { // do if otherwise
                    private.d_in2_pad[col_base_pad + row] = 0;
                }
            }
        }

        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {

            // figure out column position
            pos_ori = ei_new * in2_pad_rows_local;

            // loop through all rows
            sum = 0;
            for (position = pos_ori; position < pos_ori + in2_pad_rows_local; position++) {
                sum += private.d_in2_pad[position];
                private.d_in2_pad[position] = sum;
            }
        }

        // work
        for (col = 0; col < in2_sub_cols; col++) {
            const int col_offset_sub = col * in2_sub_rows;
            const int col_base_pad_1 = (col + in2_pad_cumv_sel_collow - 1) * in2_pad_rows_local +
                                       (in2_pad_cumv_sel_rowlow - 1);
            const int col_base_pad_2 = (col + in2_pad_cumv_sel2_collow - 1) * in2_pad_rows_local +
                                       (in2_pad_cumv_sel2_rowlow - 1);
            for (row = 0; row < in2_sub_rows; row++) {

                temp = private.d_in2_pad[col_base_pad_1 + row];
                temp2 = private.d_in2_pad[col_base_pad_2 + row];

                private.d_in2_sub[col_offset_sub + row] = temp - temp2;
            }
        }

        for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {

            // figure out row position
            pos_ori = ei_new;

            // loop through all rows
            sum = 0;
            for (position = pos_ori; position < pos_ori + in2_sub_elem;
                 position += in2_sub_rows) {
                sum += private.d_in2_sub[position];
                private.d_in2_sub[position] = sum;
            }
        }

        // work
        for (col = 0; col < in2_sub2_sqr_cols; col++) {
            const int col_offset_sub2 = col * in2_sub2_sqr_rows;
            const int col_base_sub_1 =
                (col + in2_sub_cumh_sel_collow - 1) * in2_sub_rows +
                (in2_sub_cumh_sel_rowlow - 1);
            const int col_base_sub_2 =
                (col + in2_sub_cumh_sel2_collow - 1) * in2_sub_rows +
                (in2_sub_cumh_sel2_rowlow - 1);
            const int conv_col_base = col * public.in2_sub2_sqr_rows;
            for (row = 0; row < in2_sub2_sqr_rows; row++) {

                temp = private.d_in2_sub[col_base_sub_1 + row];
                temp2 = private.d_in2_sub[col_base_sub_2 + row];

                temp2 = temp - temp2;

                private.d_in2_sub2_sqr[col_offset_sub2 + row] = temp2 * temp2;

                private.d_conv[conv_col_base + row] =
                    private.d_conv[conv_col_base + row] -
                    temp2 * in_final_sum / in_mod_elem;
            }
        }

        // work
        for (col = 0; col < in2_pad_cols; col++) {
            const int col_base_pad = col * in2_pad_rows_local;
            const int col_shift = col - in2_pad_add_cols;
            const int in2_col_base = col_shift * in2_rows_local;
            for (row = 0; row < in2_pad_rows_local; row++) {

                // execution
                if (row > (in2_pad_add_rows - 1) && // do if has numbers in original array
                    row < (in2_pad_add_rows + in2_rows_local) && col > (in2_pad_add_cols - 1) &&
                    col < (in2_pad_add_cols + in2_cols_local)) {
                    ori_row = row - in2_pad_add_rows;
                    ori_pointer = in2_col_base + ori_row;
                    private.d_in2_pad[col_base_pad + row] =
                        private.d_in2_sqr[ori_pointer];
                } else { // do if otherwise
                    private.d_in2_pad[col_base_pad + row] = 0;
                }
            }
        }

        // work
        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {

            // figure out column position
            pos_ori = ei_new * in2_pad_rows_local;

            // loop through all rows
            sum = 0;
            for (position = pos_ori; position < pos_ori + in2_pad_rows_local; position++) {
                sum += private.d_in2_pad[position];
                private.d_in2_pad[position] = sum;
            }
        }

        // work
        for (col = 0; col < in2_sub_cols; col++) {
            const int col_offset_sub = col * in2_sub_rows;
            const int col_base_pad_1 = (col + in2_pad_cumv_sel_rowlow - 1 +
                                        (in2_pad_cumv_sel_collow - 1) * in2_pad_rows_local);
            const int col_base_pad_2 = (col + in2_pad_cumv_sel2_rowlow - 1 +
                                        (in2_pad_cumv_sel2_collow - 1) * in2_pad_rows_local);
            for (row = 0; row < in2_sub_rows; row++) {

                temp = private.d_in2_pad[col_base_pad_1 + row];
                temp2 = private.d_in2_pad[col_base_pad_2 + row];

                private.d_in2_sub[col_offset_sub + row] = temp - temp2;
            }
        }

        for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {

            // figure out row position
            pos_ori = ei_new;

            // loop through all rows
            sum = 0;
            for (position = pos_ori; position < pos_ori + in2_sub_elem;
                 position += in2_sub_rows) {
                sum += private.d_in2_sub[position];
                private.d_in2_sub[position] = sum;
            }
        }

        // work
        const int conv_cols_local = conv_cols;
        const int conv_rows_local = conv_rows;
        for (col = 0; col < conv_cols_local; col++) {
            const int conv_col_base = col * conv_rows_local;
            const int sub_col_base_1 =
                (col + in2_sub_cumh_sel_collow - 1) * in2_sub_rows +
                (in2_sub_cumh_sel_rowlow - 1);
            const int sub_col_base_2 =
                (col + in2_sub_cumh_sel2_collow - 1) * in2_sub_rows +
                (in2_sub_cumh_sel2_rowlow - 1);
            for (row = 0; row < conv_rows_local; row++) {

                temp = private.d_in2_sub[sub_col_base_1 + row];
                temp2 = private.d_in2_sub[sub_col_base_2 + row];

                temp2 = temp - temp2;

                temp2 = temp2 - (private.d_in2_sub2_sqr[conv_col_base + row] /
                                 in_mod_elem);

                if (temp2 < 0) {
                    temp2 = 0;
                }
                temp2 = sqrt(temp2);

                temp2 = denomT * temp2;

                private.d_conv[conv_col_base + row] =
                    private.d_conv[conv_col_base + row] / temp2;
            }
        }

        //====================================================================================================
        //	TEMPLATE MASK CREATE
        //====================================================================================================

        // parameters
        cent = public.sSize + public.tSize + 1;
        pointer = public.frame_no - 1 + private.point_no * public.frames;
        tMask_row = cent + private.d_tRowLoc[pointer] -
                    private.d_Row[private.point_no] - 1;
        tMask_col = cent + private.d_tColLoc[pointer] -
                    private.d_Col[private.point_no] - 1;

        const int tMask_rows = public.tMask_rows;
        const int tMask_cols = public.tMask_cols;
        const int tMask_elem = public.tMask_elem;
        const int mask_conv_cols = public.mask_conv_cols;
        const int mask_conv_rows = public.mask_conv_rows;
        const int mask_cols = public.mask_cols;
        const int mask_rows = public.mask_rows;
        const int mask_conv_joffset = public.mask_conv_joffset;
        const int mask_conv_ioffset = public.mask_conv_ioffset;
        const int conv_rows_global = public.conv_rows;

        // work
        for (ei_new = 0; ei_new < tMask_elem; ei_new++) {
            private.d_tMask[ei_new] = 0;
        }
        private.d_tMask[tMask_col * tMask_rows + tMask_row] = 1;

        // work
        for (col = 1; col <= mask_conv_cols; col++) {

            // col setup
            j = col + mask_conv_joffset;
            jp1 = j + 1;
            ja1 = (mask_cols < jp1) ? (jp1 - mask_cols) : 1;
            ja2 = (tMask_cols < j) ? tMask_cols : j;

            for (row = 1; row <= mask_conv_rows; row++) {

                // row setup
                i = row + mask_conv_ioffset;
                ip1 = i + 1;

                ia1 = (mask_rows < ip1) ? (ip1 - mask_rows) : 1;
                ia2 = (tMask_rows < i) ? tMask_rows : i;

                s = 0;

                // get data
                for (ja = ja1; ja <= ja2; ja++) {
                    jb = jp1 - ja;
                    const int tMask_col_base = (ja - 1) * tMask_rows;
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        (void)jb;
                        (void)ib;
                        s += private.d_tMask[tMask_col_base + (ia - 1)];
                    }
                }

                location = (col - 1) * conv_rows_global + (row - 1);
                private.d_mask_conv[location] =
                    private.d_conv[location] * s;
            }
        }

        fin_max_val = 0;
        fin_max_coo = 0;
        const int mask_conv_elem = public.mask_conv_elem;
        for (i = 0; i < mask_conv_elem; i++) {
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
        largest_row = largest_row + 1; // compensate to match MATLAB format (1-n)
        largest_col = largest_col + 1; // compensate to match MATLAB format (1-n)
        offset_row =
            largest_row - in_mod_rows - (public.sSize - public.tSize);
        offset_col =
            largest_col - in_mod_cols - (public.sSize - public.tSize);
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

        // update template, limit the number of working threads to the size of
        // template
        d_in = &private.d_T[private.in_pointer];

        const int in_mod_rows = public.in_mod_rows;
        const int in_mod_cols = public.in_mod_cols;
        const int frame_rows = public.frame_rows;
        const int base_row = private.d_Row[private.point_no] - 26;
        const int base_col = private.d_Col[private.point_no] - 26;
        const fp alpha = public.alpha;
        const fp one_minus_alpha = (fp)1.0 - alpha;

        for (col = 0; col < in_mod_cols; col++) {
            const int ori_col_base = (base_col + col) * frame_rows + base_row;
            const int d_in_col_offset = col * in_mod_rows;
            for (row = 0; row < in_mod_rows; row++) {

                ori_pointer = ori_col_base + row;
                location = d_in_col_offset + row;

                d_in[location] =
                    alpha * d_in[location] +
                    one_minus_alpha * public.d_frame[ori_pointer];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    heartwall_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
