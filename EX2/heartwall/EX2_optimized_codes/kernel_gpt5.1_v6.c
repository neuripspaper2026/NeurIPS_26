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
        const int base_row = private.d_Row[private.point_no] - 26;
        const int base_col = private.d_Col[private.point_no] - 26;

        // update template, limit the number of working threads to the size of template
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) private(row, ori_row, ori_col, ori_pointer) schedule(static)
        #endif
        for (col = 0; col < in_mod_cols; col++) {
            for (row = 0; row < in_mod_rows; row++) {

                // figure out row/col location in corresponding new template
                // area in image and give to every thread (get top left corner
                // and progress down and right)
                ori_row = base_row + row;
                ori_col = base_col + col;
                ori_pointer = ori_col * frame_rows + ori_row;

                // update template
                d_in[col * in_mod_rows + row] = public.d_frame[ori_pointer];
            }
        }
    }

    //======================================================================================================================================================
    //	PROCESS POINTS
    //======================================================================================================================================================

    // process points in all frames except for the first one
    if (public.frame_no != 0) {
        const int sSize = public.sSize;
        in2_rowlow = private.d_Row[private.point_no] - sSize; // (1 to n+1)
        in2_collow = private.d_Col[private.point_no] - sSize;

        const int in2_rows = public.in2_rows;
        const int in2_cols = public.in2_cols;
        const int frame_rows = public.frame_rows;

        // work
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) private(row, ori_row, ori_col, temp) schedule(static)
        #endif
        for (col = 0; col < in2_cols; col++) {
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

        const int in_mod_rows = public.in_mod_rows;
        const int in_mod_cols = public.in_mod_cols;
        const int in_mod_elem = public.in_mod_elem;

        // work
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) private(row, rot_row, rot_col, pointer, temp) schedule(static)
        #endif
        for (col = 0; col < in_mod_cols; col++) {
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
        for (i = 0; i < in_mod_elem; i++) {
            in_final_sum += d_in[i];
        }

        in_sqr_final_sum = 0;
        for (i = 0; i < in_mod_elem; i++) {
            in_sqr_final_sum += private.d_in_sqr[i];
        }

        mean =
            in_final_sum /
            in_mod_elem; // gets mean (average) value of element in ROI
        mean_sqr = mean * mean;
        variance = (in_sqr_final_sum / in_mod_elem) -
                   mean_sqr;        // gets variance of ROI
        deviation = sqrt(variance); // gets standard deviation of ROI

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

        const int joffset = public.joffset;
        const int ioffset = public.ioffset;
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

        // work: convolution
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) private(row,i,ip1,ja1,ja2,ia1,ia2,j,jp1,ja,jb,ia,ib,s) schedule(static)
        #endif
        for (col = 1; col <= conv_cols; col++) {

            // column setup
            j = col + joffset;
            jp1 = j + 1;
            if (public.in2_cols < jp1) {
                ja1 = jp1 - public.in2_cols;
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
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        s += private.d_in_mod[in_mod_rows * (ja - 1) + ia - 1] *
                             private.d_in2[in2_rows * (jb - 1) + ib - 1];
                    }
                }

                private.d_conv[(col - 1) * conv_rows + (row - 1)] = s;
            }
        }

        // work: pad in2 to in2_pad
        {
            const int in2_rows_l = in2_rows;
            const int in2_cols_l = in2_cols;
            const int pad_rows = in2_pad_rows;
            const int pad_cols = in2_pad_cols;
            const int pad_add_rows = in2_pad_add_rows;
            const int pad_add_cols = in2_pad_add_cols;

            #ifdef _OPENMP
            #pragma omp parallel for collapse(2) private(row,ori_row,ori_col,temp) schedule(static)
            #endif
            for (col = 0; col < pad_cols; col++) {
                for (row = 0; row < pad_rows; row++) {

                    if (row > (pad_add_rows - 1) &&
                        row < (pad_add_rows + in2_rows_l) &&
                        col > (pad_add_cols - 1) &&
                        col < (pad_add_cols + in2_cols_l)) {
                        ori_row = row - pad_add_rows;
                        ori_col = col - pad_add_cols;
                        temp = private.d_in2[ori_col * in2_rows_l + ori_row];
                        private.d_in2_pad[col * pad_rows + row] = temp;
                    } else {
                        private.d_in2_pad[col * pad_rows + row] = 0;
                    }
                }
            }
        }

        // column-wise cumulative sum on in2_pad
        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {

            // figure out column position
            pos_ori = ei_new * in2_pad_rows;

            // loop through all rows
            sum = 0;
            for (position = pos_ori; position < pos_ori + in2_pad_rows;
                 position = position + 1) {
                private.d_in2_pad[position] = private.d_in2_pad[position] + sum;
                sum = private.d_in2_pad[position];
            }
        }

        // work: build in2_sub via differences of cumv selections
        {
            const int pad_rows = in2_pad_rows;
            const int sub_rows = in2_sub_rows;
            const int sub_cols = in2_sub_cols;
            const int sel1_rowlow = in2_pad_cumv_sel_rowlow;
            const int sel1_collow = in2_pad_cumv_sel_collow;
            const int sel2_rowlow = in2_pad_cumv_sel2_rowlow;
            const int sel2_collow = in2_pad_cumv_sel2_collow;

            #ifdef _OPENMP
            #pragma omp parallel for collapse(2) private(row,ori_row,ori_col,temp,temp2) schedule(static)
            #endif
            for (col = 0; col < sub_cols; col++) {
                for (row = 0; row < sub_rows; row++) {

                    ori_row = row + sel1_rowlow - 1;
                    ori_col = col + sel1_collow - 1;
                    temp = private.d_in2_pad[ori_col * pad_rows + ori_row];

                    ori_row = row + sel2_rowlow - 1;
                    ori_col = col + sel2_collow - 1;
                    temp2 = private.d_in2_pad[ori_col * pad_rows + ori_row];

                    private.d_in2_sub[col * sub_rows + row] = temp - temp2;
                }
            }
        }

        // row-wise cumulative sum on in2_sub
        for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {

            // figure out row position
            pos_ori = ei_new;

            // loop through all rows
            sum = 0;
            for (position = pos_ori; position < pos_ori + in2_sub_elem;
                 position = position + in2_sub_rows) {
                private.d_in2_sub[position] = private.d_in2_sub[position] + sum;
                sum = private.d_in2_sub[position];
            }
        }

        // build in2_sub2_sqr and update numerator conv
        {
            const int pad_rows = in2_sub_rows;
            const int sub2_rows = in2_sub2_sqr_rows;
            const int sub2_cols = in2_sub2_sqr_cols;
            const int sel1_rowlow = in2_sub_cumh_sel_rowlow;
            const int sel1_collow = in2_sub_cumh_sel_collow;
            const int sel2_rowlow = in2_sub_cumh_sel2_rowlow;
            const int sel2_collow = in2_sub_cumh_sel2_collow;
            const fp inv_in_mod_elem = in_final_sum / in_mod_elem;

            #ifdef _OPENMP
            #pragma omp parallel for collapse(2) private(row,ori_row,ori_col,temp,temp2) schedule(static)
            #endif
            for (col = 0; col < sub2_cols; col++) {
                for (row = 0; row < sub2_rows; row++) {

                    ori_row = row + sel1_rowlow - 1;
                    ori_col = col + sel1_collow - 1;
                    temp = private.d_in2_sub[ori_col * pad_rows + ori_row];

                    ori_row = row + sel2_rowlow - 1;
                    ori_col = col + sel2_collow - 1;
                    temp2 = private.d_in2_sub[ori_col * pad_rows + ori_row];

                    temp2 = temp - temp2;

                    private.d_in2_sub2_sqr[col * sub2_rows + row] =
                        temp2 * temp2;

                    private.d_conv[col * sub2_rows + row] =
                        private.d_conv[col * sub2_rows + row] -
                        temp2 * inv_in_mod_elem;
                }
            }
        }

        // work: pad in2_sqr to in2_pad
        {
            const int in2_rows_l = in2_rows;
            const int in2_cols_l = in2_cols;
            const int pad_rows = in2_pad_rows;
            const int pad_cols = in2_pad_cols;
            const int pad_add_rows = in2_pad_add_rows;
            const int pad_add_cols = in2_pad_add_cols;

            #ifdef _OPENMP
            #pragma omp parallel for collapse(2) private(row,ori_row,ori_col,temp) schedule(static)
            #endif
            for (col = 0; col < pad_cols; col++) {
                for (row = 0; row < pad_rows; row++) {

                    if (row > (pad_add_rows - 1) &&
                        row < (pad_add_rows + in2_rows_l) &&
                        col > (pad_add_cols - 1) &&
                        col < (pad_add_cols + in2_cols_l)) {
                        ori_row = row - pad_add_rows;
                        ori_col = col - pad_add_cols;
                        temp =
                            private.d_in2_sqr[ori_col * in2_rows_l + ori_row];
                        private.d_in2_pad[col * pad_rows + row] = temp;
                    } else {
                        private.d_in2_pad[col * pad_rows + row] = 0;
                    }
                }
            }
        }

        // column-wise cumulative sum on in2_pad (for squared image)
        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {

            // figure out column position
            pos_ori = ei_new * in2_pad_rows;

            // loop through all rows
            sum = 0;
            for (position = pos_ori; position < pos_ori + in2_pad_rows;
                 position = position + 1) {
                private.d_in2_pad[position] = private.d_in2_pad[position] + sum;
                sum = private.d_in2_pad[position];
            }
        }

        // work: build in2_sub (squared) via differences of cumv selections
        {
            const int pad_rows = in2_pad_rows;
            const int sub_rows = in2_sub_rows;
            const int sub_cols = in2_sub_cols;
            const int sel1_rowlow = in2_pad_cumv_sel_rowlow;
            const int sel1_collow = in2_pad_cumv_sel_collow;
            const int sel2_rowlow = in2_pad_cumv_sel2_rowlow;
            const int sel2_collow = in2_pad_cumv_sel2_collow;

            #ifdef _OPENMP
            #pragma omp parallel for collapse(2) private(row,ori_row,ori_col,temp,temp2) schedule(static)
            #endif
            for (col = 0; col < sub_cols; col++) {
                for (row = 0; row < sub_rows; row++) {

                    ori_row = row + sel1_rowlow - 1;
                    ori_col = col + sel1_collow - 1;
                    temp = private.d_in2_pad[ori_col * pad_rows + ori_row];

                    ori_row = row + sel2_rowlow - 1;
                    ori_col = col + sel2_collow - 1;
                    temp2 = private.d_in2_pad[ori_col * pad_rows + ori_row];

                    private.d_in2_sub[col * sub_rows + row] = temp - temp2;
                }
            }
        }

        // row-wise cumulative sum on in2_sub (squared)
        for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {

            // figure out row position
            pos_ori = ei_new;

            // loop through all rows
            sum = 0;
            for (position = pos_ori; position < pos_ori + in2_sub_elem;
                 position = position + in2_sub_rows) {
                private.d_in2_sub[position] = private.d_in2_sub[position] + sum;
                sum = private.d_in2_sub[position];
            }
        }

        // work: final denominator and correlation
        {
            const int conv_cols_l = conv_cols;
            const int conv_rows_l = conv_rows;
            const int sub_rows = in2_sub_rows;
            const int sel1_rowlow = in2_sub_cumh_sel_rowlow;
            const int sel1_collow = in2_sub_cumh_sel_collow;
            const int sel2_rowlow = in2_sub_cumh_sel2_rowlow;
            const int sel2_collow = in2_sub_cumh_sel2_collow;
            const fp inv_in_mod_elem2 = (fp)1.0 / in_mod_elem;
            const fp denomT_local = denomT;

            #ifdef _OPENMP
            #pragma omp parallel for collapse(2) private(row,ori_row,ori_col,temp,temp2) schedule(static)
            #endif
            for (col = 0; col < conv_cols_l; col++) {
                for (row = 0; row < conv_rows_l; row++) {

                    ori_row = row + sel1_rowlow - 1;
                    ori_col = col + sel1_collow - 1;
                    temp = private.d_in2_sub[ori_col * sub_rows + ori_row];

                    ori_row = row + sel2_rowlow - 1;
                    ori_col = col + sel2_collow - 1;
                    temp2 = private.d_in2_sub[ori_col * sub_rows + ori_row];

                    temp2 = temp - temp2;

                    temp2 = temp2 -
                            (private
                                 .d_in2_sub2_sqr[col * conv_rows_l + row] *
                             inv_in_mod_elem2);

                    if (temp2 < 0) {
                        temp2 = 0;
                    }
                    temp2 = sqrt(temp2);

                    temp2 = denomT_local * temp2;

                    private.d_conv[col * conv_rows_l + row] =
                        private.d_conv[col * conv_rows_l + row] / temp2;
                }
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
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (ei_new = 0; ei_new < public.tMask_elem; ei_new++) {
            private.d_tMask[ei_new] = 0;
        }
        private.d_tMask[tMask_col * public.tMask_rows + tMask_row] = 1;

        const int mask_conv_cols = public.mask_conv_cols;
        const int mask_conv_rows = public.mask_conv_rows;
        const int mask_conv_joffset = public.mask_conv_joffset;
        const int mask_conv_ioffset = public.mask_conv_ioffset;

        // work
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) private(row,i,ip1,ja1,ja2,ia1,ia2,j,jp1,ja,jb,ia,ib,s) schedule(static)
        #endif
        for (col = 1; col <= mask_conv_cols; col++) {

            // col setup
            j = col + mask_conv_joffset;
            jp1 = j + 1;
            if (public.mask_cols < jp1) {
                ja1 = jp1 - public.mask_cols;
            } else {
                ja1 = 1;
            }
            if (public.tMask_cols < j) {
                ja2 = public.tMask_cols;
            } else {
                ja2 = j;
            }

            for (row = 1; row <= mask_conv_rows; row++) {

                // row setup
                i = row + mask_conv_ioffset;
                ip1 = i + 1;

                if (public.mask_rows < ip1) {
                    ia1 = ip1 - public.mask_rows;
                } else {
                    ia1 = 1;
                }
                if (public.tMask_rows < i) {
                    ia2 = public.tMask_rows;
                } else {
                    ia2 = i;
                }

                s = 0;

                // get data
                for (ja = ja1; ja <= ja2; ja++) {
                    jb = jp1 - ja;
                    (void)jb;
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        (void)ib;
                        s += private.d_tMask[public.tMask_rows * (ja - 1) + ia -
                                             1];
                    }
                }

                private.d_mask_conv[(col - 1) * public.conv_rows + (row - 1)] =
                    private.d_conv[(col - 1) * public.conv_rows + (row - 1)] *
                    s;
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
            (fin_max_coo + 1) % public.mask_conv_rows - 1;       // (0-n) row
        largest_col = (fin_max_coo + 1) / public.mask_conv_rows; // (0-n) column
        if ((fin_max_coo + 1) % public.mask_conv_rows == 0) {
            largest_row = public.mask_conv_rows - 1;
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

        // update template, limit the number of working threads to the size of
        // template
        d_in = &private.d_T[private.in_pointer];
        const int in_mod_rows = public.in_mod_rows;
        const int in_mod_cols = public.in_mod_cols;
        const int frame_rows = public.frame_rows;
        const fp alpha = public.alpha;
        const fp one_minus_alpha = (fp)1.0 - alpha;
        const int base_row_u = private.d_Row[private.point_no] - 26;
        const int base_col_u = private.d_Col[private.point_no] - 26;

        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) private(row,ori_row,ori_col,ori_pointer,temp) schedule(static)
        #endif
        for (col = 0; col < in_mod_cols; col++) {
            for (row = 0; row < in_mod_rows; row++) {

                // figure out row/col location in corresponding new template
                // area in image and give to every thread (get top left corner
                // and progress down and right)
                ori_row = base_row_u + row;
                ori_col = base_col_u + col;
                ori_pointer = ori_col * frame_rows + ori_row;

                // update template
                temp = d_in[col * in_mod_rows + row];
                d_in[col * in_mod_rows + row] =
                    alpha * temp +
                    one_minus_alpha * public.d_frame[ori_pointer];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    heartwall_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
