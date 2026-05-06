<<<CODE>>>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <string.h>

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

        // update template, limit the number of working threads to the size of
        // template
        #pragma omp parallel for collapse(2) private(row, col, ori_row, ori_col, ori_pointer) schedule(static)
        for (col = 0; col < public.in_mod_cols; col++) {
            for (row = 0; row < public.in_mod_rows; row++) {

                // figure out row/col location in corresponding new template
                // area in image and give to every thread (get top left corner
                // and progress down and right)
                ori_row = private.d_Row[private.point_no] - 25 + row - 1;
                ori_col = private.d_Col[private.point_no] - 25 + col - 1;
                ori_pointer = ori_col * public.frame_rows + ori_row;

                // update template
                d_in[col * public.in_mod_rows + row] =
                    public.d_frame[ori_pointer];
            }
        }
    }

    //======================================================================================================================================================
    //	PROCESS POINTS
    //======================================================================================================================================================

    // process points in all frames except for the first one
    if (public.frame_no != 0) {
        in2_rowlow =
            private.d_Row[private.point_no] - public.sSize; // (1 to n+1)
        in2_collow = private.d_Col[private.point_no] - public.sSize;

        // work
        #pragma omp parallel for collapse(2) private(row, col, ori_row, ori_col, temp) schedule(static)
        for (col = 0; col < public.in2_cols; col++) {
            for (row = 0; row < public.in2_rows; row++) {

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + in2_rowlow - 1;
                ori_col = col + in2_collow - 1;
                temp = public.d_frame[ori_col * public.frame_rows + ori_row];
                private.d_in2[col * public.in2_rows + row] = temp;
                private.d_in2_sqr[col * public.in2_rows + row] = temp * temp;
            }
        }

        // variables
        d_in = &private.d_T[private.in_pointer];

        // work
        #pragma omp parallel for collapse(2) private(row, col, rot_row, rot_col, pointer, temp) schedule(static)
        for (col = 0; col < public.in_mod_cols; col++) {
            for (row = 0; row < public.in_mod_rows; row++) {

                // rotated coordinates
                rot_row = (public.in_mod_rows - 1) - row;
                rot_col = (public.in_mod_rows - 1) - col;
                pointer = rot_col * public.in_mod_rows + rot_row;

                // execution
                temp = d_in[pointer];
                private.d_in_mod[col * public.in_mod_rows + row] = temp;
                private.d_in_sqr[pointer] = temp * temp;
            }
        }

        in_final_sum = 0;
        #pragma omp parallel for reduction(+:in_final_sum) schedule(static)
        for (i = 0; i < public.in_mod_elem; i++) {
            in_final_sum = in_final_sum + d_in[i];
        }

        in_sqr_final_sum = 0;
        #pragma omp parallel for reduction(+:in_sqr_final_sum) schedule(static)
        for (i = 0; i < public.in_mod_elem; i++) {
            in_sqr_final_sum = in_sqr_final_sum + private.d_in_sqr[i];
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
        #pragma omp parallel for collapse(2) private(row, col, j, jp1, ja1, ja2, i, ip1, ia1, ia2, s, ja, jb, ia, ib) schedule(dynamic, 4)
        for (col = 1; col <= public.conv_cols; col++) {

            // column setup
            j = col + public.joffset;
            jp1 = j + 1;
            if (public.in2_cols < jp1) {
                ja1 = jp1 - public.in2_cols;
            } else {
                ja1 = 1;
            }
            if (public.in_mod_cols < j) {
                ja2 = public.in_mod_cols;
            } else {
                ja2 = j;
            }

            for (row = 1; row <= public.conv_rows; row++) {

                // row range setup
                i = row + public.ioffset;
                ip1 = i + 1;

                if (public.in2_rows < ip1) {
                    ia1 = ip1 - public.in2_rows;
                } else {
                    ia1 = 1;
                }
                if (public.in_mod_rows < i) {
                    ia2 = public.in_mod_rows;
                } else {
                    ia2 = i;
                }

                s = 0;

                // getting data
                for (ja = ja1; ja <= ja2; ja++) {
                    jb = jp1 - ja;
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        s = s +
                            private.d_in_mod[public.in_mod_rows * (ja - 1) +
                                             ia - 1] *
                                private
                                    .d_in2[public.in2_rows * (jb - 1) + ib - 1];
                    }
                }

                private.d_conv[(col - 1) * public.conv_rows + (row - 1)] = s;
            }
        }

        // work
        #pragma omp parallel for collapse(2) private(row, col, ori_row, ori_col) schedule(static)
        for (col = 0; col < public.in2_pad_cols; col++) {
            for (row = 0; row < public.in2_pad_rows; row++) {

                // execution
                if (row > (public.in2_pad_add_rows -
                           1) && // do if has numbers in original array
                    row < (public.in2_pad_add_rows + public.in2_rows) &&
                    col > (public.in2_pad_add_cols - 1) &&
                    col < (public.in2_pad_add_cols + public.in2_cols)) {
                    ori_row = row - public.in2_pad_add_rows;
                    ori_col = col - public.in2_pad_add_cols;
                    private.d_in2_pad[col * public.in2_pad_rows + row] =
                        private.d_in2[ori_col * public.in2_rows + ori_row];
                } else { // do if otherwise
                    private.d_in2_pad[col * public.in2_pad_rows + row] = 0;
                }
            }
        }

        #pragma omp parallel for private(ei_new, pos_ori, sum, position) schedule(static)
        for (ei_new = 0; ei_new < public.in2_pad_cols; ei_new++) {

            // figure out column position
            pos_ori = ei_new * public.in2_pad_rows;

            // loop through all rows
            sum = 0;
            for (position = pos_ori; position < pos_ori + public.in2_pad_rows;
                 position = position + 1) {
                private.d_in2_pad[position] = private.d_in2_pad[position] + sum;
                sum = private.d_in2_pad[position];
            }
        }

        // work
        #pragma omp parallel for collapse(2) private(row, col, ori_row, ori_col, temp, temp2) schedule(static)
        for (col = 0; col < public.in2_sub_cols; col++) {
            for (row = 0; row < public.in2_sub_rows; row++) {

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + public.in2_pad_cumv_sel_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel_collow - 1;
                temp =
                    private.d_in2_pad[ori_col * public.in2_pad_rows + ori_row];

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + public.in2_pad_cumv_sel2_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel2_collow - 1;
                temp2 =
                    private.d_in2_pad[ori_col * public.in2_pad_rows + ori_row];

                // subtraction
                private.d_in2_sub[col * public.in2_sub_rows + row] = temp - temp2;
            }
        }

        #pragma omp parallel for private(ei_new, pos_ori, sum, position) schedule(static)
        for (ei_new = 0; ei_new < public.in2_sub_rows; ei_new++) {

            // figure out row position
            pos_ori = ei_new;

            // loop through all rows
            sum = 0;
            for (position = pos_ori; position < pos_ori + public.in2_sub_elem;
                 position = position + public.in2_sub_rows) {
                private.d_in2_sub[position] = private.d_in2_sub[position] + sum;
                sum = private.d_in2_sub[position];
            }
        }

        // work
        #pragma omp parallel for collapse(2) private(row, col, ori_row, ori_col, temp, temp2) schedule(static)
        for (col = 0; col < public.in2_sub2_sqr_cols; col++) {
            for (row = 0; row < public.in2_sub2_sqr_rows; row++) {

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + public.in2_sub_cumh_sel_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel_collow - 1;
                temp =
                    private.d_in2_sub[ori_col * public.in2_sub_rows + ori_row];

                // figure out corresponding location in old matrix and copy
                // values to new matrix
                ori_row = row + public.in2_sub_cumh_sel2_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel2_collow - 1;
                temp2 =
                    private.d_in2_sub[ori_col * public.in2_sub_rows + ori_row];

                // subtraction
                temp2 = temp - temp2;

                // squaring
                private.d_in2_sub2_sqr[col * public.in2_sub2_sqr_rows + row] =
                    temp2 * temp2;

                // numerator
                private.d_conv[col * public.in2_sub2_sqr_rows + row] =
                    private.d_conv[col * public.in2_sub2_sqr_rows + row] -
                    temp2 * in_final_sum / public.in_mod_elem;
            }
        }

        // work
        #pragma omp parallel for collapse(2) private(row, col, ori_row, ori_col) schedule(static)
        for (col = 0; col < public.in2_pad_cols; col++) {
            for (row = 0; row < public.in2_pad_rows; row++) {

                // execution
                if (row > (public.in2_pad_add_rows -
                           1) && // do if has numbers in original array
                    row < (public.in2_pad_add_rows + public.in2_rows) &&
                    col > (public.in2_pad_add_cols - 1) &&
                    col < (public.in2_pad_add_cols + public.in2_cols)) {
                    ori_row = row - public.in2_pad_add_rows;
                    ori_col = col - public.in2_pad_add_cols;
                    private.d_in2_pad[col * public.in2_pad_rows + row] =
                        private.d_in2_sqr[ori_col * public.in2_rows + ori_row];
