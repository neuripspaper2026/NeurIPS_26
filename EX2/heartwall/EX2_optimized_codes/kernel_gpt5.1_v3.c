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
        const fp *restrict d_frame = public.d_frame;
        fp *restrict d_T = d_in;
        const int base_row = private.d_Row[private.point_no] - 26;
        const int base_col = private.d_Col[private.point_no] - 26;

        // update template, limit the number of working threads to the size of template
        #pragma omp parallel for if(in_mod_cols * in_mod_rows > 1024) default(none) shared(in_mod_cols,in_mod_rows,frame_rows,d_frame,d_T,base_row,base_col) private(col,row,ori_row,ori_col,ori_pointer) schedule(static)
        for (col = 0; col < in_mod_cols; col++) {
            for (row = 0; row < in_mod_rows; row++) {

                // figure out row/col location in corresponding new template area in image and give to every thread (get top left corner and progress down and right)
                ori_row = base_row + row;
                ori_col = base_col + col;
                ori_pointer = ori_col * frame_rows + ori_row;

                // update template
                d_T[col * in_mod_rows + row] = d_frame[ori_pointer];
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

        const int in2_rows = public.in2_rows;
        const int in2_cols = public.in2_cols;
        const int frame_rows = public.frame_rows;
        const fp *restrict d_frame = public.d_frame;
        fp *restrict d_in2 = private.d_in2;
        fp *restrict d_in2_sqr = private.d_in2_sqr;
        const int in2_rowlow_loc = in2_rowlow;
        const int in2_collow_loc = in2_collow;

        // work
        #pragma omp parallel for if(in2_cols * in2_rows > 1024) default(none) shared(in2_cols,in2_rows,frame_rows,d_frame,d_in2,d_in2_sqr,in2_rowlow_loc,in2_collow_loc) private(col,row,ori_row,ori_col,temp) schedule(static)
        for (col = 0; col < in2_cols; col++) {
            for (row = 0; row < in2_rows; row++) {

                // figure out corresponding location in old matrix and copy values to new matrix
                ori_row = row + in2_rowlow_loc - 1;
                ori_col = col + in2_collow_loc - 1;
                temp = d_frame[ori_col * frame_rows + ori_row];
                d_in2[col * in2_rows + row] = temp;
                d_in2_sqr[col * in2_rows + row] = temp * temp;
            }
        }

        // variables
        d_in = &private.d_T[private.in_pointer];

        {
            const int in_mod_rows = public.in_mod_rows;
            const int in_mod_cols = public.in_mod_cols;
            const int in_mod_elem = public.in_mod_elem;
            fp *restrict d_in_mod = private.d_in_mod;
            fp *restrict d_in_sqr = private.d_in_sqr;
            fp *restrict d_T = d_in;

            // work: rotate template and compute square
            #pragma omp parallel for if(in_mod_cols * in_mod_rows > 1024) default(none) shared(in_mod_cols,in_mod_rows,d_in_mod,d_in_sqr,d_T) private(col,row,rot_row,rot_col,pointer,temp) schedule(static)
            for (col = 0; col < in_mod_cols; col++) {
                for (row = 0; row < in_mod_rows; row++) {

                    // rotated coordinates
                    rot_row = (in_mod_rows - 1) - row;
                    rot_col = (in_mod_rows - 1) - col;
                    pointer = rot_col * in_mod_rows + rot_row;

                    // execution
                    temp = d_T[pointer];
                    d_in_mod[col * in_mod_rows + row] = temp;
                    d_in_sqr[pointer] = temp * temp;
                }
            }

            in_final_sum = 0;
            in_sqr_final_sum = 0;

            #pragma omp parallel for if(in_mod_elem > 1024) default(none) shared(in_mod_elem,d_T,d_in_sqr) reduction(+:in_final_sum,in_sqr_final_sum)
            for (i = 0; i < in_mod_elem; i++) {
                in_final_sum += d_T[i];
                in_sqr_final_sum += d_in_sqr[i];
            }

            mean = in_final_sum / in_mod_elem; // gets mean (average) value of element in ROI
            mean_sqr = mean * mean;
            variance = (in_sqr_final_sum / in_mod_elem) - mean_sqr; // gets variance of ROI
            deviation = sqrt(variance);                             // gets standard deviation of ROI

            denomT = sqrt((fp)(in_mod_elem - 1)) * deviation;
        }

        {
            const int conv_rows = public.conv_rows;
            const int conv_cols = public.conv_cols;
            const int in_mod_rows = public.in_mod_rows;
            const int in_mod_cols = public.in_mod_cols;
            const int in2_rows = public.in2_rows;
            const int in2_cols = public.in2_cols;
            const int joffset = public.joffset;
            const int ioffset = public.ioffset;
            fp *restrict d_in_mod = private.d_in_mod;
            fp *restrict d_in2_loc = private.d_in2;
            fp *restrict d_conv = private.d_conv;

            // work: convolution
            #pragma omp parallel for if(conv_cols * conv_rows > 128) default(none) shared(conv_cols,conv_rows,in_mod_rows,in_mod_cols,in2_rows,in2_cols,joffset,ioffset,d_in_mod,d_in2_loc,d_conv) private(col,row,j,jp1,ja1,ja2,i,ip1,ia1,ia2,ja,jb,ia,ib,s) schedule(static)
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
                        for (ia = ia1; ia <= ia2; ia++) {
                            ib = ip1 - ia;
                            s += d_in_mod[in_mod_rows * (ja - 1) + ia - 1] *
                                 d_in2_loc[in2_rows * (jb - 1) + ib - 1];
                        }
                    }

                    d_conv[(col - 1) * conv_rows + (row - 1)] = s;
                }
            }
        }

        {
            const int in2_pad_rows = public.in2_pad_rows;
            const int in2_pad_cols = public.in2_pad_cols;
            const int in2_pad_add_rows = public.in2_pad_add_rows;
            const int in2_pad_add_cols = public.in2_pad_add_cols;
            const int in2_rows = public.in2_rows;
            const int in2_cols = public.in2_cols;
            fp *restrict d_in2_pad = private.d_in2_pad;
            fp *restrict d_in2_loc = private.d_in2;

            // work: padding
            #pragma omp parallel for if(in2_pad_cols * in2_pad_rows > 1024) default(none) shared(in2_pad_cols,in2_pad_rows,in2_pad_add_rows,in2_pad_add_cols,in2_rows,in2_cols,d_in2_pad,d_in2_loc) private(col,row,ori_row,ori_col,temp) schedule(static)
            for (col = 0; col < in2_pad_cols; col++) {
                for (row = 0; row < in2_pad_rows; row++) {

                    // execution
                    if (row > (in2_pad_add_rows - 1) && // do if has numbers in original array
                        row < (in2_pad_add_rows + in2_rows) &&
                        col > (in2_pad_add_cols - 1) &&
                        col < (in2_pad_add_cols + in2_cols)) {
                        ori_row = row - in2_pad_add_rows;
                        ori_col = col - in2_pad_add_cols;
                        d_in2_pad[col * in2_pad_rows + row] =
                            d_in2_loc[ori_col * in2_rows + ori_row];
                    } else { // do if otherwise
                        d_in2_pad[col * in2_pad_rows + row] = 0;
                    }
                }
            }

            // cumulative sum over rows per column
            for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {

                // figure out column position
                pos_ori = ei_new * in2_pad_rows;

                // loop through all rows
                sum = 0;
                for (position = pos_ori; position < pos_ori + in2_pad_rows;
                     position = position + 1) {
                    d_in2_pad[position] = d_in2_pad[position] + sum;
                    sum = d_in2_pad[position];
                }
            }
        }

        {
            const int in2_sub_rows = public.in2_sub_rows;
            const int in2_sub_cols = public.in2_sub_cols;
            const int in2_pad_rows = public.in2_pad_rows;
            const int in2_pad_cumv_sel_rowlow = public.in2_pad_cumv_sel_rowlow;
            const int in2_pad_cumv_sel_collow = public.in2_pad_cumv_sel_collow;
            const int in2_pad_cumv_sel2_rowlow = public.in2_pad_cumv_sel2_rowlow;
            const int in2_pad_cumv_sel2_collow = public.in2_pad_cumv_sel2_collow;
            fp *restrict d_in2_pad = private.d_in2_pad;
            fp *restrict d_in2_sub = private.d_in2_sub;

            // work: sub
            #pragma omp parallel for if(in2_sub_cols * in2_sub_rows > 1024) default(none) shared(in2_sub_cols,in2_sub_rows,in2_pad_rows,in2_pad_cumv_sel_rowlow,in2_pad_cumv_sel_collow,in2_pad_cumv_sel2_rowlow,in2_pad_cumv_sel2_collow,d_in2_pad,d_in2_sub) private(col,row,ori_row,ori_col,temp,temp2) schedule(static)
            for (col = 0; col < in2_sub_cols; col++) {
                for (row = 0; row < in2_sub_rows; row++) {

                    // figure out corresponding location in old matrix and copy values to new matrix
                    ori_row = row + in2_pad_cumv_sel_rowlow - 1;
                    ori_col = col + in2_pad_cumv_sel_collow - 1;
                    temp = d_in2_pad[ori_col * in2_pad_rows + ori_row];

                    // figure out corresponding location in old matrix and copy values to new matrix
                    ori_row = row + in2_pad_cumv_sel2_rowlow - 1;
                    ori_col = col + in2_pad_cumv_sel2_collow - 1;
                    temp2 = d_in2_pad[ori_col * in2_pad_rows + ori_row];

                    // subtraction
                    d_in2_sub[col * in2_sub_rows + row] = temp - temp2;
                }
            }

            const int in2_sub_elem = public.in2_sub_elem;

            for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {

                // figure out row position
                pos_ori = ei_new;

                // loop through all rows
                sum = 0;
                for (position = pos_ori; position < pos_ori + in2_sub_elem;
                     position = position + in2_sub_rows) {
                    d_in2_sub[position] = d_in2_sub[position] + sum;
                    sum = d_in2_sub[position];
                }
            }
        }

        {
            const int in2_sub2_sqr_rows = public.in2_sub2_sqr_rows;
            const int in2_sub2_sqr_cols = public.in2_sub2_sqr_cols;
            const int in2_sub_rows = public.in2_sub_rows;
            const int in2_sub_cumh_sel_rowlow = public.in2_sub_cumh_sel_rowlow;
            const int in2_sub_cumh_sel_collow = public.in2_sub_cumh_sel_collow;
            const int in2_sub_cumh_sel2_rowlow = public.in2_sub_cumh_sel2_rowlow;
            const int in2_sub_cumh_sel2_collow = public.in2_sub_cumh_sel2_collow;
            const int in_mod_elem = public.in_mod_elem;
            fp *restrict d_in2_sub = private.d_in2_sub;
            fp *restrict d_in2_sub2_sqr = private.d_in2_sub2_sqr;
            fp *restrict d_conv = private.d_conv;
            const fp in_final_sum_loc = in_final_sum;

            // work
            #pragma omp parallel for if(in2_sub2_sqr_cols * in2_sub2_sqr_rows > 1024) default(none) shared(in2_sub2_sqr_cols,in2_sub2_sqr_rows,in2_sub_rows,in2_sub_cumh_sel_rowlow,in2_sub_cumh_sel_collow,in2_sub_cumh_sel2_rowlow,in2_sub_cumh_sel2_collow,in_mod_elem,in_final_sum_loc,d_in2_sub,d_in2_sub2_sqr,d_conv) private(col,row,ori_row,ori_col,temp,temp2) schedule(static)
            for (col = 0; col < in2_sub2_sqr_cols; col++) {
                for (row = 0; row < in2_sub2_sqr_rows; row++) {

                    // figure out corresponding location in old matrix and copy values to new matrix
                    ori_row = row + in2_sub_cumh_sel_rowlow - 1;
                    ori_col = col + in2_sub_cumh_sel_collow - 1;
                    temp =
                        d_in2_sub[ori_col * in2_sub_rows + ori_row];

                    // figure out corresponding location in old matrix and copy values to new matrix
                    ori_row = row + in2_sub_cumh_sel2_rowlow - 1;
                    ori_col = col + in2_sub_cumh_sel2_collow - 1;
                    temp2 =
                        d_in2_sub[ori_col * in2_sub_rows + ori_row];

                    // subtraction
                    temp2 = temp - temp2;

                    // squaring
                    d_in2_sub2_sqr[col * in2_sub2_sqr_rows + row] =
                        temp2 * temp2;

                    // numerator
                    d_conv[col * in2_sub2_sqr_rows + row] =
                        d_conv[col * in2_sub2_sqr_rows + row] -
                        temp2 * in_final_sum_loc / in_mod_elem;
                }
            }
        }

        {
            const int in2_pad_rows = public.in2_pad_rows;
            const int in2_pad_cols = public.in2_pad_cols;
            const int in2_pad_add_rows = public.in2_pad_add_rows;
            const int in2_pad_add_cols = public.in2_pad_add_cols;
            const int in2_rows = public.in2_rows;
            const int in2_cols = public.in2_cols;
            fp *restrict d_in2_pad = private.d_in2_pad;
            fp *restrict d_in2_sqr_loc = private.d_in2_sqr;

            // work
            #pragma omp parallel for if(in2_pad_cols * in2_pad_rows > 1024) default(none) shared(in2_pad_cols,in2_pad_rows,in2_pad_add_rows,in2_pad_add_cols,in2_rows,in2_cols,d_in2_pad,d_in2_sqr_loc) private(col,row,ori_row,ori_col) schedule(static)
            for (col = 0; col < in2_pad_cols; col++) {
                for (row = 0; row < in2_pad_rows; row++) {

                    // execution
                    if (row > (in2_pad_add_rows - 1) && // do if has numbers in original array
                        row < (in2_pad_add_rows + in2_rows) &&
                        col > (in2_pad_add_cols - 1) &&
                        col < (in2_pad_add_cols + in2_cols)) {
                        ori_row = row - in2_pad_add_rows;
                        ori_col = col - in2_pad_add_cols;
                        d_in2_pad[col * in2_pad_rows + row] =
                            d_in2_sqr_loc[ori_col * in2_rows + ori_row];
                    } else { // do if otherwise
                        d_in2_pad[col * in2_pad_rows + row] = 0;
                    }
                }
            }

            // work
            for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {

                // figure out column position
                pos_ori = ei_new * in2_pad_rows;

                // loop through all rows
                sum = 0;
                for (position = pos_ori; position < pos_ori + in2_pad_rows;
                     position = position + 1) {
                    d_in2_pad[position] = d_in2_pad[position] + sum;
                    sum = d_in2_pad[position];
                }
            }
        }

        {
            const int in2_sub_rows = public.in2_sub_rows;
            const int in2_sub_cols = public.in2_sub_cols;
            const int in2_pad_rows = public.in2_pad_rows;
            const int in2_pad_cumv_sel_rowlow = public.in2_pad_cumv_sel_rowlow;
            const int in2_pad_cumv_sel_collow = public.in2_pad_cumv_sel_collow;
            const int in2_pad_cumv_sel2_rowlow = public.in2_pad_cumv_sel2_rowlow;
            const int in2_pad_cumv_sel2_collow = public.in2_pad_cumv_sel2_collow;
            const int in2_sub_elem = public.in2_sub_elem;
            fp *restrict d_in2_pad = private.d_in2_pad;
            fp *restrict d_in2_sub = private.d_in2_sub;

            // work
            #pragma omp parallel for if(in2_sub_cols * in2_sub_rows > 1024) default(none) shared(in2_sub_cols,in2_sub_rows,in2_pad_rows,in2_pad_cumv_sel_rowlow,in2_pad_cumv_sel_collow,in2_pad_cumv_sel2_rowlow,in2_pad_cumv_sel2_collow,d_in2_pad,d_in2_sub) private(col,row,ori_row,ori_col,temp,temp2) schedule(static)
            for (col = 0; col < in2_sub_cols; col++) {
                for (row = 0; row < in2_sub_rows; row++) {

                    // figure out corresponding location in old matrix and copy values to new matrix
                    ori_row = row + in2_pad_cumv_sel_rowlow - 1;
                    ori_col = col + in2_pad_cumv_sel_collow - 1;
                    temp =
                        d_in2_pad[ori_col * in2_pad_rows + ori_row];

                    // figure out corresponding location in old matrix and copy values to new matrix
                    ori_row = row + in2_pad_cumv_sel2_rowlow - 1;
                    ori_col = col + in2_pad_cumv_sel2_collow - 1;
                    temp2 =
                        d_in2_pad[ori_col * in2_pad_rows + ori_row];

                    // subtract
                    d_in2_sub[col * in2_sub_rows + row] = temp - temp2;
                }
            }

            for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {

                // figure out row position
                pos_ori = ei_new;

                // loop through all rows
                sum = 0;
                for (position = pos_ori; position < pos_ori + in2_sub_elem;
                     position = position + in2_sub_rows) {
                    d_in2_sub[position] = d_in2_sub[position] + sum;
                    sum = d_in2_sub[position];
                }
            }
        }

        {
            const int conv_rows = public.conv_rows;
            const int conv_cols = public.conv_cols;
            const int in2_sub_rows = public.in2_sub_rows;
            const int in2_sub_cumh_sel_rowlow = public.in2_sub_cumh_sel_rowlow;
            const int in2_sub_cumh_sel_collow = public.in2_sub_cumh_sel_collow;
            const int in2_sub_cumh_sel2_rowlow = public.in2_sub_cumh_sel2_rowlow;
            const int in2_sub_cumh_sel2_collow = public.in2_sub_cumh_sel2_collow;
            const int in_mod_elem = public.in_mod_elem;
            fp *restrict d_in2_sub = private.d_in2_sub;
            fp *restrict d_in2_sub2_sqr = private.d_in2_sub2_sqr;
            fp *restrict d_conv = private.d_conv;
            const fp denomT_loc = denomT;

            // work
            #pragma omp parallel for if(conv_cols * conv_rows > 128) default(none) shared(conv_cols,conv_rows,in2_sub_rows,in2_sub_cumh_sel_rowlow,in2_sub_cumh_sel_collow,in2_sub_cumh_sel2_rowlow,in2_sub_cumh_sel2_collow,in_mod_elem,denomT_loc,d_in2_sub,d_in2_sub2_sqr,d_conv) private(col,row,ori_row,ori_col,temp,temp2) schedule(static)
            for (col = 0; col < conv_cols; col++) {
                for (row = 0; row < conv_rows; row++) {

                    // figure out corresponding location in old matrix and copy values to new matrix
                    ori_row = row + in2_sub_cumh_sel_rowlow - 1;
                    ori_col = col + in2_sub_cumh_sel_collow - 1;
                    temp =
                        d_in2_sub[ori_col * in2_sub_rows + ori_row];

                    // figure out corresponding location in old matrix and copy values to new matrix
                    ori_row = row + in2_sub_cumh_sel2_rowlow - 1;
                    ori_col = col + in2_sub_cumh_sel2_collow - 1;
                    temp2 =
                        d_in2_sub[ori_col * in2_sub_rows + ori_row];

                    // subtract
                    temp2 = temp - temp2;

                    // diff_local_sums
                    temp2 = temp2 -
                            (d_in2_sub2_sqr[col * conv_rows + row] /
                             in_mod_elem);

                    // denominator A
                    if (temp2 < 0) {
                        temp2 = 0;
                    }
                    temp2 = sqrt(temp2);

                    // denominator
                    temp2 = denomT_loc * temp2;

                    // correlation
                    d_conv[col * conv_rows + row] =
                        d_conv[col * conv_rows + row] / temp2;
                }
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

        {
            const int tMask_elem = public.tMask_elem;
            const int tMask_rows = public.tMask_rows;
            const int conv_rows = public.conv_rows;
            const int mask_conv_cols = public.mask_conv_cols;
            const int mask_conv_rows = public.mask_conv_rows;
            const int mask_cols = public.mask_cols;
            const int mask_rows = public.mask_rows;
            const int mask_conv_joffset = public.mask_conv_joffset;
            const int mask_conv_ioffset = public.mask_conv_ioffset;
            fp *restrict d_tMask = private.d_tMask;
            fp *restrict d_mask_conv = private.d_mask_conv;
            fp *restrict d_conv = private.d_conv;

            // work
            #pragma omp parallel for if(tMask_elem > 256) default(none) shared(tMask_elem,d_tMask) private(ei_new) schedule(static)
            for (ei_new = 0; ei_new < tMask_elem; ei_new++) {
                d_tMask[ei_new] = 0;
            }
            d_tMask[tMask_col * tMask_rows + tMask_row] = 1;

            // work
            #pragma omp parallel for if(mask_conv_cols * mask_conv_rows > 128) default(none) shared(mask_conv_cols,mask_conv_rows,mask_cols,mask_rows,tMask_rows,mask_conv_joffset,mask_conv_ioffset,d_tMask,d_mask_conv,d_conv,conv_rows) private(col,row,j,jp1,ja1,ja2,i,ip1,ia1,ia2,ja,jb,ia,ib,s) schedule(static)
            for (col = 1; col <= mask_conv_cols; col++) {

                // col setup
                j = col + mask_conv_joffset;
                jp1 = j + 1;
                if (mask_cols < jp1) {
                    ja1 = jp1 - mask_cols;
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

                    // get data
                    for (ja = ja1; ja <= ja2; ja++) {
                        jb = jp1 - ja;
                        for (ia = ia1; ia <= ia2; ia++) {
                            ib = ip1 - ia;
                            s += d_tMask[tMask_rows * (ja - 1) + ia - 1] * 1;
                        }
                    }

                    d_mask_conv[(col - 1) * conv_rows + (row - 1)] =
                        d_conv[(col - 1) * conv_rows + (row - 1)] * s;
                }
            }

            fin_max_val = 0;
            fin_max_coo = 0;
            {
                const int mask_conv_elem = public.mask_conv_elem;
                #pragma omp parallel for if(mask_conv_elem > 512) default(none) shared(mask_conv_elem,d_mask_conv) reduction(max:fin_max_val)
                for (i = 0; i < mask_conv_elem; i++) {
                    if (d_mask_conv[i] > fin_max_val) {
                        fin_max_val = d_mask_conv[i];
                    }
                }

                // find coordinate of max (separate pass to respect reduction)
                for (i = 0; i < mask_conv_elem; i++) {
                    if (d_mask_conv[i] == fin_max_val) {
                        fin_max_coo = i;
                        break;
                    }
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
                largest_row - public.in_mod_rows - (public.sSize - public.tSize);
            offset_col =
                largest_col - public.in_mod_cols - (public.sSize - public.tSize);
            pointer = private.point_no * public.frames + public.frame_no;
            private.d_tRowLoc[pointer] = private.d_Row[private.point_no] + offset_row;
            private.d_tColLoc[pointer] = private.d_Col[private.point_no] + offset_col;
        }
    }

    // if the last frame in the bath, update template
    if (public.frame_no != 0 && (public.frame_no) % 10 == 0) {

        // update coordinate
        loc_pointer = private.point_no * public.frames + public.frame_no;
        private.d_Row[private.point_no] = private.d_tRowLoc[loc_pointer];
        private.d_Col[private.point_no] = private.d_tColLoc[loc_pointer];

        d_in = &private.d_T[private.in_pointer];

        const int in_mod_rows = public.in_mod_rows;
        const int in_mod_cols = public.in_mod_cols;
        const int frame_rows = public.frame_rows;
        const fp alpha = public.alpha;
        const fp one_minus_alpha = 1.00 - public.alpha;
        const int base_row = private.d_Row[private.point_no] - 26;
        const int base_col = private.d_Col[private.point_no] - 26;
        fp *restrict d_T = d_in;
        const fp *restrict d_frame = public.d_frame;

        // update template, limit the number of working threads to the size of template
        #pragma omp parallel for if(in_mod_cols * in_mod_rows > 1024) default(none) shared(in_mod_cols,in_mod_rows,frame_rows,d_T,d_frame,alpha,one_minus_alpha,base_row,base_col) private(col,row,ori_row,ori_col,ori_pointer) schedule(static)
        for (col = 0; col < in_mod_cols; col++) {
            for (row = 0; row < in_mod_rows; row++) {

                // figure out row/col location in corresponding new template area in image and give to every thread (get top left corner and progress down and right)
                ori_row = base_row + row;
                ori_col = base_col + col;
                ori_pointer = ori_col * frame_rows + ori_row;

                // update template
                d_T[col * in_mod_rows + row] =
                    alpha * d_T[col * in_mod_rows + row] +
                    one_minus_alpha * d_frame[ori_pointer];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    heartwall_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
