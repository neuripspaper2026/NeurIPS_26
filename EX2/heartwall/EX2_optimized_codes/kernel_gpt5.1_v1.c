#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

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

        // update template, limit the number of working threads to the size of
        // template
        for (col = 0; col < in_mod_cols; col++) {
#pragma omp simd
            for (row = 0; row < in_mod_rows; row++) {

                // figure out row/col location in corresponding new template
                // area in image and give to every thread (get top left corner
                // and progress down and right)
                ori_row = private.d_Row[private.point_no] - 25 + row - 1;
                ori_col = private.d_Col[private.point_no] - 25 + col - 1;
                ori_pointer = ori_col * frame_rows + ori_row;

                // update template
                d_in[col * in_mod_rows + row] =
                    public.d_frame[ori_pointer];
            }
        }
    }

    //======================================================================================================================================================
    //	PROCESS POINTS
    //======================================================================================================================================================

    // process points in all frames except for the first one
    if (public.frame_no != 0) {

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
        const int mask_conv_elem = public.mask_conv_elem;
        const int frame_rows = public.frame_rows;
        const int in_mod_rows = public.in_mod_rows;
        const int in_mod_cols = public.in_mod_cols;
        const int in_mod_elem = public.in_mod_elem;

        in2_rowlow =
            private.d_Row[private.point_no] - public.sSize; // (1 to n+1)
        in2_collow = private.d_Col[private.point_no] - public.sSize;

        // work
        for (col = 0; col < in2_cols; col++) {
#pragma omp simd
            for (row = 0; row < in2_rows; row++) {

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + in2_rowlow - 1;
                ori_col = col + in2_collow - 1;
                temp = public.d_frame[ori_col * frame_rows + ori_row];
                private.d_in2[col * in2_rows + row] = temp;
                private.d_in2_sqr[col * in2_rows + row] = temp * temp;
            }
        }

        // variables
        d_in = &private.d_T[private.in_pointer];

        // work
        for (col = 0; col < in_mod_cols; col++) {
#pragma omp simd
            for (row = 0; row < in_mod_rows; row++) {

                // rotated coordinates
                rot_row = (in_mod_rows - 1) - row;
                rot_col = (in_mod_rows - 1) - col;
                pointer = rot_col * in_mod_rows + rot_row;

                // execution
                temp = d_in[pointer];
                private.d_in_mod[col * in_mod_rows + row] = temp;
                private.d_in_sqr[pointer] = temp * temp;
            }
        }

        in_final_sum = 0;
#pragma omp simd reduction(+ : in_final_sum)
        for (i = 0; i < in_mod_elem; i++) {
            in_final_sum = in_final_sum + d_in[i];
        }

        in_sqr_final_sum = 0;
#pragma omp simd reduction(+ : in_sqr_final_sum)
        for (i = 0; i < in_mod_elem; i++) {
            in_sqr_final_sum = in_sqr_final_sum + private.d_in_sqr[i];
        }

        mean =
            in_final_sum /
            in_mod_elem; // gets mean (average) value of element in ROI
        mean_sqr = mean * mean;
        variance = (in_sqr_final_sum / in_mod_elem) -
                   mean_sqr;        // gets variance of ROI
        deviation = sqrt(variance); // gets standard deviation of ROI

        denomT = sqrt((fp)(in_mod_elem - 1)) * deviation;

        // work
        for (col = 1; col <= conv_cols; col++) {

            // column setup
            j = col + public.joffset;
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
                i = row + public.ioffset;
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
#pragma omp simd reduction(+ : s)
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        s = s +
                            private.d_in_mod[in_mod_rows * (ja - 1) +
                                             ia - 1] *
                            private.d_in2[in2_rows * (jb - 1) + ib - 1];
                    }
                }

                private.d_conv[(col - 1) * conv_rows + (row - 1)] = s;
            }
        }

        // work
        for (col = 0; col < in2_pad_cols; col++) {
#pragma omp simd
            for (row = 0; row < in2_pad_rows; row++) {

                // execution
                if (row > (public.in2_pad_add_rows -
                           1) && // do if has numbers in original array
                    row < (public.in2_pad_add_rows + in2_rows) &&
                    col > (public.in2_pad_add_cols - 1) &&
                    col < (public.in2_pad_add_cols + in2_cols)) {
                    ori_row = row - public.in2_pad_add_rows;
                    ori_col = col - public.in2_pad_add_cols;
                    private.d_in2_pad[col * in2_pad_rows + row] =
                        private.d_in2[ori_col * in2_rows + ori_row];
                } else { // do if otherwise
                    private.d_in2_pad[col * in2_pad_rows + row] = 0;
                }
            }
        }

        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {

            // figure out column position
            pos_ori = ei_new * in2_pad_rows;

            // loop through all rows
            sum = 0;
#pragma omp simd
            for (position = pos_ori; position < pos_ori + in2_pad_rows;
                 position = position + 1) {
                private.d_in2_pad[position] = private.d_in2_pad[position] + sum;
                sum = private.d_in2_pad[position];
            }
        }

        // work
        for (col = 0; col < in2_sub_cols; col++) {
#pragma omp simd
            for (row = 0; row < in2_sub_rows; row++) {

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + public.in2_pad_cumv_sel_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel_collow - 1;
                temp =
                    private.d_in2_pad[ori_col * in2_pad_rows + ori_row];

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + public.in2_pad_cumv_sel2_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel2_collow - 1;
                temp2 =
                    private.d_in2_pad[ori_col * in2_pad_rows + ori_row];

                // subtraction
                private.d_in2_sub[col * in2_sub_rows + row] = temp - temp2;
            }
        }

        for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {

            // figure out row position
            pos_ori = ei_new;

            // loop through all rows
            sum = 0;
#pragma omp simd
            for (position = pos_ori; position < pos_ori + in2_sub_elem;
                 position = position + in2_sub_rows) {
                private.d_in2_sub[position] =
                    private.d_in2_sub[position] + sum;
                sum = private.d_in2_sub[position];
            }
        }

        // work
        for (col = 0; col < in2_sub2_sqr_cols; col++) {
#pragma omp simd
            for (row = 0; row < in2_sub2_sqr_rows; row++) {

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + public.in2_sub_cumh_sel_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel_collow - 1;
                temp =
                    private.d_in2_sub[ori_col * in2_sub_rows + ori_row];

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + public.in2_sub_cumh_sel2_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel2_collow - 1;
                temp2 =
                    private.d_in2_sub[ori_col * in2_sub_rows + ori_row];

                // subtraction
                temp2 = temp - temp2;

                // squaring
                private.d_in2_sub2_sqr[col * in2_sub2_sqr_rows + row] =
                    temp2 * temp2;

                // numerator
                private.d_conv[col * in2_sub2_sqr_rows + row] =
                    private.d_conv[col * in2_sub2_sqr_rows + row] -
                    temp2 * in_final_sum / in_mod_elem;
            }
        }

        // work
        for (col = 0; col < in2_pad_cols; col++) {
#pragma omp simd
            for (row = 0; row < in2_pad_rows; row++) {

                // execution
                if (row > (public.in2_pad_add_rows -
                           1) && // do if has numbers in original array
                    row < (public.in2_pad_add_rows + in2_rows) &&
                    col > (public.in2_pad_add_cols - 1) &&
                    col < (public.in2_pad_add_cols + in2_cols)) {
                    ori_row = row - public.in2_pad_add_rows;
                    ori_col = col - public.in2_pad_add_cols;
                    private.d_in2_pad[col * in2_pad_rows + row] =
                        private.d_in2_sqr[ori_col * in2_rows + ori_row];
                } else { // do if otherwise
                    private.d_in2_pad[col * in2_pad_rows + row] = 0;
                }
            }
        }

        // work
        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {

            // figure out column position
            pos_ori = ei_new * in2_pad_rows;

            // loop through all rows
            sum = 0;
#pragma omp simd
            for (position = pos_ori; position < pos_ori + in2_pad_rows;
                 position = position + 1) {
                private.d_in2_pad[position] = private.d_in2_pad[position] + sum;
                sum = private.d_in2_pad[position];
            }
        }

        // work
        for (col = 0; col < in2_sub_cols; col++) {
#pragma omp simd
            for (row = 0; row < in2_sub_rows; row++) {

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + public.in2_pad_cumv_sel_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel_collow - 1;
                temp =
                    private.d_in2_pad[ori_col * in2_pad_rows + ori_row];

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + public.in2_pad_cumv_sel2_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel2_collow - 1;
                temp2 =
                    private.d_in2_pad[ori_col * in2_pad_rows + ori_row];

                // subtract
                private.d_in2_sub[col * in2_sub_rows + row] = temp - temp2;
            }
        }

        for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {

            // figure out row position
            pos_ori = ei_new;

            // loop through all rows
            sum = 0;
#pragma omp simd
            for (position = pos_ori; position < pos_ori + in2_sub_elem;
                 position = position + in2_sub_rows) {
                private.d_in2_sub[position] =
                    private.d_in2_sub[position] + sum;
                sum = private.d_in2_sub[position];
            }
        }

        // work
        for (col = 0; col < conv_cols; col++) {
#pragma omp simd
            for (row = 0; row < conv_rows; row++) {

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + public.in2_sub_cumh_sel_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel_collow - 1;
                temp =
                    private.d_in2_sub[ori_col * in2_sub_rows + ori_row];

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + public.in2_sub_cumh_sel2_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel2_collow - 1;
                temp2 =
                    private.d_in2_sub[ori_col * in2_sub_rows + ori_row];

                // subtract
                temp2 = temp - temp2;

                // diff_local_sums
                temp2 = temp2 -
                        (private.d_in2_sub2_sqr[col * conv_rows + row] /
                         in_mod_elem);

                // denominator A
                if (temp2 < 0) {
                    temp2 = 0;
                }
                temp2 = sqrt(temp2);

                // denominator
                temp2 = denomT * temp2;

                // correlation
                private.d_conv[col * conv_rows + row] =
                    private.d_conv[col * conv_rows + row] / temp2;
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

        // work
#pragma omp simd
        for (ei_new = 0; ei_new < tMask_elem; ei_new++) {
            private.d_tMask[ei_new] = 0;
        }
        private.d_tMask[tMask_col * tMask_rows + tMask_row] = 1;

        // work
        // for(col=1; col<=public.conv_cols; col++){
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

            // for(row=1; row<=public.conv_rows; row++){
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
                for (ja = ja1; ja <= ja2; ja++) {
                    jb = jp1 - ja;
#pragma omp simd reduction(+ : s)
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        s = s +
                            private.d_tMask[tMask_rows * (ja - 1) + ia -
                                            1] *
                            1;
                    }
                }

                private.d_mask_conv[(col - 1) * conv_rows + (row - 1)] =
                    private.d_conv[(col - 1) * conv_rows + (row - 1)] *
                    s;
            }
        }

        fin_max_val = 0;
        fin_max_coo = 0;
#pragma omp simd
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
        largest_row =
            largest_row + 1; // compensate to match MATLAB format (1-n)
        largest_col =
            largest_col + 1; // compensate to match MATLAB format (1-n)
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
        const int in_mod_rows2 = public.in_mod_rows;
        const int in_mod_cols2 = public.in_mod_cols;
        const int frame_rows2 = public.frame_rows;

        for (col = 0; col < in_mod_cols2; col++) {
#pragma omp simd
            for (row = 0; row < in_mod_rows2; row++) {

                // figure out row/col location in corresponding new template
                // area in image and give to every thread (get top left corner
                // and progress down and right)
                ori_row = private.d_Row[private.point_no] - 25 + row - 1;
                ori_col = private.d_Col[private.point_no] - 25 + col - 1;
                ori_pointer = ori_col * frame_rows2 + ori_row;

                // update template
                d_in[col * in_mod_rows2 + row] =
                    public.alpha * d_in[col * in_mod_rows2 + row] +
                    (1.00 - public.alpha) * public.d_frame[ori_pointer];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    heartwall_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
