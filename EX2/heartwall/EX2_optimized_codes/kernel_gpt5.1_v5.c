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
        const int frame_rows  = public.frame_rows;
        const int base_row    = private.d_Row[private.point_no] - 26;
        const int base_col    = private.d_Col[private.point_no] - 26;

        // update template, limit the number of working threads to the size of template
        #pragma omp parallel for collapse(2) private(row, ori_row, ori_col, ori_pointer, temp) default(none) shared(in_mod_cols, in_mod_rows, frame_rows, base_row, base_col, public, d_in)
        for (col = 0; col < in_mod_cols; col++) {
            for (row = 0; row < in_mod_rows; row++) {

                ori_row = base_row + row;
                ori_col = base_col + col;
                ori_pointer = ori_col * frame_rows + ori_row;

                d_in[col * in_mod_rows + row] = public.d_frame[ori_pointer];
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

        const int in2_rows      = public.in2_rows;
        const int in2_cols      = public.in2_cols;
        const int frame_rows    = public.frame_rows;
        const int in2_pad_rows  = public.in2_pad_rows;
        const int in2_pad_cols  = public.in2_pad_cols;
        const int in2_sub_rows  = public.in2_sub_rows;
        const int in2_sub_cols  = public.in2_sub_cols;
        const int in2_sub_elem  = public.in2_sub_elem;
        const int in_mod_rows   = public.in_mod_rows;
        const int in_mod_cols   = public.in_mod_cols;
        const int in_mod_elem   = public.in_mod_elem;
        const int conv_rows     = public.conv_rows;
        const int conv_cols     = public.conv_cols;
        const int tMask_rows    = public.tMask_rows;
        const int tMask_cols    = public.tMask_cols;
        const int tMask_elem    = public.tMask_elem;
        const int mask_rows     = public.mask_rows;
        const int mask_cols     = public.mask_cols;
        const int mask_conv_rows = public.mask_conv_rows;
        const int mask_conv_cols = public.mask_conv_cols;
        const int mask_conv_elem = public.mask_conv_elem;

        // work: copy to d_in2 and d_in2_sqr
        #pragma omp parallel for collapse(2) private(row, ori_row, ori_col, temp) default(none) shared(in2_cols, in2_rows, in2_rowlow, in2_collow, frame_rows, public, private)
        for (col = 0; col < in2_cols; col++) {
            for (row = 0; row < in2_rows; row++) {

                ori_row = row + in2_rowlow - 1;
                ori_col = col + in2_collow - 1;
                temp = public.d_frame[ori_col * frame_rows + ori_row];
                private.d_in2[col * in2_rows + row] = temp;
                private.d_in2_sqr[col * in2_rows + row] = temp * temp;
            }
        }

        // variables
        d_in = &private.d_T[private.in_pointer];

        // work: rotate template and square
        #pragma omp parallel for collapse(2) private(row, rot_row, rot_col, pointer, temp) default(none) shared(in_mod_cols, in_mod_rows, d_in, private, public)
        for (col = 0; col < in_mod_cols; col++) {
            for (row = 0; row < in_mod_rows; row++) {

                rot_row = (in_mod_rows - 1) - row;
                rot_col = (in_mod_rows - 1) - col;
                pointer = rot_col * in_mod_rows + rot_row;

                temp = d_in[pointer];
                private.d_in_mod[col * in_mod_rows + row] = temp;
                private.d_in_sqr[pointer] = temp * temp;
            }
        }

        in_final_sum = 0;
        for (i = 0; i < in_mod_elem; i++) {
            in_final_sum = in_final_sum + d_in[i];
        }

        in_sqr_final_sum = 0;
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

        // work: main convolution
        #pragma omp parallel for private(col, row, j, jp1, ja1, ja2, i, ip1, ia1, ia2, ja, jb, ia, ib, s) default(none) shared(conv_cols, conv_rows, public, private)
        for (col = 1; col <= conv_cols; col++) {

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

            for (row = 1; row <= conv_rows; row++) {

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

                for (ja = ja1; ja <= ja2; ja++) {
                    jb = jp1 - ja;
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        s += private.d_in_mod[public.in_mod_rows * (ja - 1) +
                                              ia - 1] *
                             private.d_in2[public.in2_rows * (jb - 1) + ib - 1];
                    }
                }

                private.d_conv[(col - 1) * conv_rows + (row - 1)] = s;
            }
        }

        // work: pad d_in2_pad from d_in2
        #pragma omp parallel for collapse(2) private(row, ori_row, ori_col) default(none) shared(in2_pad_cols, in2_pad_rows, public, private)
        for (col = 0; col < in2_pad_cols; col++) {
            for (row = 0; row < in2_pad_rows; row++) {

                if (row > (public.in2_pad_add_rows - 1) &&
                    row < (public.in2_pad_add_rows + public.in2_rows) &&
                    col > (public.in2_pad_add_cols - 1) &&
                    col < (public.in2_pad_add_cols + public.in2_cols)) {
                    ori_row = row - public.in2_pad_add_rows;
                    ori_col = col - public.in2_pad_add_cols;
                    private.d_in2_pad[col * in2_pad_rows + row] =
                        private.d_in2[ori_col * public.in2_rows + ori_row];
                } else {
                    private.d_in2_pad[col * in2_pad_rows + row] = 0;
                }
            }
        }

        // column-wise cumulative sum over d_in2_pad
        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {

            pos_ori = ei_new * in2_pad_rows;

            sum = 0;
            for (position = pos_ori; position < pos_ori + in2_pad_rows;
                 position = position + 1) {
                private.d_in2_pad[position] =
                    private.d_in2_pad[position] + sum;
                sum = private.d_in2_pad[position];
            }
        }

        // work: d_in2_sub from d_in2_pad (first set)
        #pragma omp parallel for collapse(2) private(row, ori_row, ori_col, temp, temp2) default(none) shared(in2_sub_cols, in2_sub_rows, in2_pad_rows, public, private)
        for (col = 0; col < in2_sub_cols; col++) {
            for (row = 0; row < in2_sub_rows; row++) {

                ori_row = row + public.in2_pad_cumv_sel_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel_collow - 1;
                temp =
                    private.d_in2_pad[ori_col * in2_pad_rows + ori_row];

                ori_row = row + public.in2_pad_cumv_sel2_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel2_collow - 1;
                temp2 =
                    private.d_in2_pad[ori_col * in2_pad_rows + ori_row];

                private.d_in2_sub[col * in2_sub_rows + row] = temp - temp2;
            }
        }

        // row-wise cumulative sum over d_in2_sub
        for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {

            pos_ori = ei_new;

            sum = 0;
            for (position = pos_ori; position < pos_ori + in2_sub_elem;
                 position = position + in2_sub_rows) {
                private.d_in2_sub[position] =
                    private.d_in2_sub[position] + sum;
                sum = private.d_in2_sub[position];
            }
        }

        // work: d_in2_sub2_sqr and adjusted numerator
        #pragma omp parallel for collapse(2) private(row, ori_row, ori_col, temp, temp2) default(none) shared(public, private, in2_sub2_sqr_rows, in2_sub2_sqr_cols, in2_sub_rows, in_mod_elem, in_final_sum)
        for (col = 0; col < public.in2_sub2_sqr_cols; col++) {
            for (row = 0; row < public.in2_sub2_sqr_rows; row++) {

                ori_row = row + public.in2_sub_cumh_sel_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel_collow - 1;
                temp =
                    private.d_in2_sub[ori_col * in2_sub_rows + ori_row];

                ori_row = row + public.in2_sub_cumh_sel2_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel2_collow - 1;
                temp2 =
                    private.d_in2_sub[ori_col * in2_sub_rows + ori_row];

                temp2 = temp - temp2;

                private.d_in2_sub2_sqr[col * public.in2_sub2_sqr_rows + row] =
                    temp2 * temp2;

                private.d_conv[col * public.in2_sub2_sqr_rows + row] =
                    private.d_conv[col * public.in2_sub2_sqr_rows + row] -
                    temp2 * in_final_sum / in_mod_elem;
            }
        }

        // work: pad d_in2_pad with d_in2_sqr
        #pragma omp parallel for collapse(2) private(row, ori_row, ori_col) default(none) shared(in2_pad_cols, in2_pad_rows, public, private)
        for (col = 0; col < in2_pad_cols; col++) {
            for (row = 0; row < in2_pad_rows; row++) {

                if (row > (public.in2_pad_add_rows - 1) &&
                    row < (public.in2_pad_add_rows + public.in2_rows) &&
                    col > (public.in2_pad_add_cols - 1) &&
                    col < (public.in2_pad_add_cols + public.in2_cols)) {
                    ori_row = row - public.in2_pad_add_rows;
                    ori_col = col - public.in2_pad_add_cols;
                    private.d_in2_pad[col * in2_pad_rows + row] =
                        private.d_in2_sqr[ori_col * public.in2_rows + ori_row];
                } else {
                    private.d_in2_pad[col * in2_pad_rows + row] = 0;
                }
            }
        }

        // column-wise cumulative sum over squared pad
        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {

            pos_ori = ei_new * in2_pad_rows;

            sum = 0;
            for (position = pos_ori; position < pos_ori + in2_pad_rows;
                 position = position + 1) {
                private.d_in2_pad[position] =
                    private.d_in2_pad[position] + sum;
                sum = private.d_in2_pad[position];
            }
        }

        // work: second d_in2_sub from squared pad
        #pragma omp parallel for collapse(2) private(row, ori_row, ori_col, temp, temp2) default(none) shared(in2_sub_cols, in2_sub_rows, in2_pad_rows, public, private)
        for (col = 0; col < in2_sub_cols; col++) {
            for (row = 0; row < in2_sub_rows; row++) {

                ori_row = row + public.in2_pad_cumv_sel_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel_collow - 1;
                temp =
                    private.d_in2_pad[ori_col * in2_pad_rows + ori_row];

                ori_row = row + public.in2_pad_cumv_sel2_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel2_collow - 1;
                temp2 =
                    private.d_in2_pad[ori_col * in2_pad_rows + ori_row];

                private.d_in2_sub[col * in2_sub_rows + row] = temp - temp2;
            }
        }

        // row-wise cumulative sum over second d_in2_sub
        for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {

            pos_ori = ei_new;

            sum = 0;
            for (position = pos_ori; position < pos_ori + in2_sub_elem;
                 position = position + in2_sub_rows) {
                private.d_in2_sub[position] =
                    private.d_in2_sub[position] + sum;
                sum = private.d_in2_sub[position];
            }
        }

        // work: final correlation denominator and division
        #pragma omp parallel for collapse(2) private(row, ori_row, ori_col, temp, temp2) default(none) shared(conv_cols, conv_rows, in2_sub_rows, denomT, in_mod_elem, public, private)
        for (col = 0; col < conv_cols; col++) {
            for (row = 0; row < conv_rows; row++) {

                ori_row = row + public.in2_sub_cumh_sel_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel_collow - 1;
                temp =
                    private.d_in2_sub[ori_col * in2_sub_rows + ori_row];

                ori_row = row + public.in2_sub_cumh_sel2_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel2_collow - 1;
                temp2 =
                    private.d_in2_sub[ori_col * in2_sub_rows + ori_row];

                temp2 = temp - temp2;

                temp2 = temp2 -
                        (private.d_in2_sub2_sqr[col * conv_rows + row] /
                         in_mod_elem);

                if (temp2 < 0) {
                    temp2 = 0;
                }
                temp2 = sqrt(temp2);

                temp2 = denomT * temp2;

                private.d_conv[col * conv_rows + row] =
                    private.d_conv[col * conv_rows + row] / temp2;
            }
        }

        //====================================================================================================
        //	TEMPLATE MASK CREATE
        //====================================================================================================

        cent = public.sSize + public.tSize + 1;
        pointer = public.frame_no - 1 + private.point_no * public.frames;
        tMask_row = cent + private.d_tRowLoc[pointer] -
                    private.d_Row[private.point_no] - 1;
        tMask_col = cent + private.d_tColLoc[pointer] -
                    private.d_Col[private.point_no] - 1;

        // work: clear mask
        #pragma omp parallel for default(none) shared(tMask_elem, private)
        for (ei_new = 0; ei_new < tMask_elem; ei_new++) {
            private.d_tMask[ei_new] = 0;
        }
        private.d_tMask[tMask_col * tMask_rows + tMask_row] = 1;

        // work: mask convolution and multiplication with d_conv
        #pragma omp parallel for private(col, row, j, jp1, ja1, ja2, i, ip1, ia1, ia2, ja, jb, ia, ib, s) default(none) shared(mask_conv_cols, mask_conv_rows, public, private, conv_rows)
        for (col = 1; col <= mask_conv_cols; col++) {

            j = col + public.mask_conv_joffset;
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

                i = row + public.mask_conv_ioffset;
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

                for (ja = ja1; ja <= ja2; ja++) {
                    jb = jp1 - ja;
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        s += private.d_tMask[public.tMask_rows * (ja - 1) +
                                             ia - 1];
                    }
                }

                private.d_mask_conv[(col - 1) * conv_rows + (row - 1)] =
                    private.d_conv[(col - 1) * conv_rows + (row - 1)] * s;
            }
        }

        fin_max_val = 0;
        fin_max_coo = 0;
        for (i = 0; i < mask_conv_elem; i++) {
            if (private.d_mask_conv[i] > fin_max_val) {
                fin_max_val = private.d_mask_conv[i];
                fin_max_coo = i;
            }
        }

        largest_row =
            (fin_max_coo + 1) % mask_conv_rows - 1;       // (0-n) row
        largest_col = (fin_max_coo + 1) / mask_conv_rows; // (0-n) column
        if ((fin_max_coo + 1) % mask_conv_rows == 0) {
            largest_row = mask_conv_rows - 1;
            largest_col = largest_col - 1;
        }

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

        d_in = &private.d_T[private.in_pointer];

        const int in_mod_rows2 = public.in_mod_rows;
        const int in_mod_cols2 = public.in_mod_cols;
        const int frame_rows2  = public.frame_rows;
        const int base_row2    = private.d_Row[private.point_no] - 26;
        const int base_col2    = private.d_Col[private.point_no] - 26;
        const fp alpha         = public.alpha;

        #pragma omp parallel for collapse(2) private(row, ori_row, ori_col, ori_pointer, temp) default(none) shared(in_mod_cols2, in_mod_rows2, frame_rows2, base_row2, base_col2, public, d_in, alpha)
        for (col = 0; col < in_mod_cols2; col++) {
            for (row = 0; row < in_mod_rows2; row++) {

                ori_row = base_row2 + row;
                ori_col = base_col2 + col;
                ori_pointer = ori_col * frame_rows2 + ori_row;

                temp = d_in[col * in_mod_rows2 + row];
                d_in[col * in_mod_rows2 + row] =
                    alpha * temp +
                    (1.00 - alpha) * public.d_frame[ori_pointer];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    heartwall_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
