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

    if (public.frame_no == 0) {

        pointer = private.point_no * public.frames + public.frame_no;
        private.d_tRowLoc[pointer] = private.d_Row[private.point_no];
        private.d_tColLoc[pointer] = private.d_Col[private.point_no];

        d_in = &private.d_T[private.in_pointer];

#ifdef _OPENMP
#pragma omp parallel for private(row, ori_row, ori_col, ori_pointer) default(none) shared(public, private, d_in)
#endif
        for (col = 0; col < public.in_mod_cols; col++) {
            for (row = 0; row < public.in_mod_rows; row++) {
                ori_row = private.d_Row[private.point_no] - 25 + row - 1;
                ori_col = private.d_Col[private.point_no] - 25 + col - 1;
                ori_pointer = ori_col * public.frame_rows + ori_row;
                d_in[col * public.in_mod_rows + row] =
                    public.d_frame[ori_pointer];
            }
        }
    }

    //======================================================================================================================================================
    //	PROCESS POINTS
    //======================================================================================================================================================

    if (public.frame_no != 0) {
        in2_rowlow =
            private.d_Row[private.point_no] - public.sSize;
        in2_collow = private.d_Col[private.point_no] - public.sSize;

#ifdef _OPENMP
#pragma omp parallel for private(row, ori_row, ori_col, temp) default(none) shared(public, private, in2_rowlow, in2_collow)
#endif
        for (col = 0; col < public.in2_cols; col++) {
            for (row = 0; row < public.in2_rows; row++) {
                ori_row = row + in2_rowlow - 1;
                ori_col = col + in2_collow - 1;
                temp = public.d_frame[ori_col * public.frame_rows + ori_row];
                private.d_in2[col * public.in2_rows + row] = temp;
                private.d_in2_sqr[col * public.in2_rows + row] = temp * temp;
            }
        }

        d_in = &private.d_T[private.in_pointer];

#ifdef _OPENMP
#pragma omp parallel for private(row, rot_row, rot_col, pointer, temp) default(none) shared(public, private, d_in)
#endif
        for (col = 0; col < public.in_mod_cols; col++) {
            for (row = 0; row < public.in_mod_rows; row++) {
                rot_row = (public.in_mod_rows - 1) - row;
                rot_col = (public.in_mod_rows - 1) - col;
                pointer = rot_col * public.in_mod_rows + rot_row;
                temp = d_in[pointer];
                private.d_in_mod[col * public.in_mod_rows + row] = temp;
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

        mean = in_final_sum / public.in_mod_elem;
        mean_sqr = mean * mean;
        variance = (in_sqr_final_sum / public.in_mod_elem) - mean_sqr;
        deviation = sqrt(variance);

        denomT = sqrt((fp)(public.in_mod_elem - 1)) * deviation;

#ifdef _OPENMP
#pragma omp parallel for private(j,jp1,ja1,ja2,row,i,ip1,ia1,ia2,s,ja,jb,ia,ib) default(none) shared(public, private)
#endif
        for (col = 1; col <= public.conv_cols; col++) {

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

                private.d_conv[(col - 1) * public.conv_rows + (row - 1)] = s;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for private(row, ori_row, ori_col) default(none) shared(public, private)
#endif
        for (col = 0; col < public.in2_pad_cols; col++) {
            for (row = 0; row < public.in2_pad_rows; row++) {

                if (row > (public.in2_pad_add_rows - 1) &&
                    row < (public.in2_pad_add_rows + public.in2_rows) &&
                    col > (public.in2_pad_add_cols - 1) &&
                    col < (public.in2_pad_add_cols + public.in2_cols)) {
                    ori_row = row - public.in2_pad_add_rows;
                    ori_col = col - public.in2_pad_add_cols;
                    private.d_in2_pad[col * public.in2_pad_rows + row] =
                        private.d_in2[ori_col * public.in2_rows + ori_row];
                } else {
                    private.d_in2_pad[col * public.in2_pad_rows + row] = 0;
                }
            }
        }

#ifdef _OPENMP
#pragma omp parallel for private(pos_ori, position, sum) default(none) shared(public, private)
#endif
        for (ei_new = 0; ei_new < public.in2_pad_cols; ei_new++) {

            pos_ori = ei_new * public.in2_pad_rows;

            sum = 0;
            for (position = pos_ori; position < pos_ori + public.in2_pad_rows;
                 position++) {
                private.d_in2_pad[position] =
                    private.d_in2_pad[position] + sum;
                sum = private.d_in2_pad[position];
            }
        }

#ifdef _OPENMP
#pragma omp parallel for private(row, ori_row, ori_col, temp, temp2) default(none) shared(public, private)
#endif
        for (col = 0; col < public.in2_sub_cols; col++) {
            for (row = 0; row < public.in2_sub_rows; row++) {

                ori_row = row + public.in2_pad_cumv_sel_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel_collow - 1;
                temp =
                    private.d_in2_pad[ori_col * public.in2_pad_rows + ori_row];

                ori_row = row + public.in2_pad_cumv_sel2_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel2_collow - 1;
                temp2 =
                    private.d_in2_pad[ori_col * public.in2_pad_rows + ori_row];

                private.d_in2_sub[col * public.in2_sub_rows + row] =
                    temp - temp2;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for private(pos_ori, position, sum) default(none) shared(public, private)
#endif
        for (ei_new = 0; ei_new < public.in2_sub_rows; ei_new++) {

            pos_ori = ei_new;

            sum = 0;
            for (position = pos_ori; position < pos_ori + public.in2_sub_elem;
                 position += public.in2_sub_rows) {
                private.d_in2_sub[position] =
                    private.d_in2_sub[position] + sum;
                sum = private.d_in2_sub[position];
            }
        }

#ifdef _OPENMP
#pragma omp parallel for private(row, ori_row, ori_col, temp, temp2) default(none) shared(public, private, in_final_sum)
#endif
        for (col = 0; col < public.in2_sub2_sqr_cols; col++) {
            for (row = 0; row < public.in2_sub2_sqr_rows; row++) {

                ori_row = row + public.in2_sub_cumh_sel_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel_collow - 1;
                temp =
                    private.d_in2_sub[ori_col * public.in2_sub_rows + ori_row];

                ori_row = row + public.in2_sub_cumh_sel2_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel2_collow - 1;
                temp2 =
                    private.d_in2_sub[ori_col * public.in2_sub_rows + ori_row];

                temp2 = temp - temp2;

                private.d_in2_sub2_sqr[col * public.in2_sub2_sqr_rows + row] =
                    temp2 * temp2;

                private.d_conv[col * public.in2_sub2_sqr_rows + row] =
                    private.d_conv[col * public.in2_sub2_sqr_rows + row] -
                    temp2 * in_final_sum / public.in_mod_elem;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for private(row, ori_row, ori_col) default(none) shared(public, private)
#endif
        for (col = 0; col < public.in2_pad_cols; col++) {
            for (row = 0; row < public.in2_pad_rows; row++) {

                if (row > (public.in2_pad_add_rows - 1) &&
                    row < (public.in2_pad_add_rows + public.in2_rows) &&
                    col > (public.in2_pad_add_cols - 1) &&
                    col < (public.in2_pad_add_cols + public.in2_cols)) {
                    ori_row = row - public.in2_pad_add_rows;
                    ori_col = col - public.in2_pad_add_cols;
                    private.d_in2_pad[col * public.in2_pad_rows + row] =
                        private.d_in2_sqr[ori_col * public.in2_rows + ori_row];
                } else {
                    private.d_in2_pad[col * public.in2_pad_rows + row] = 0;
                }
            }
        }

#ifdef _OPENMP
#pragma omp parallel for private(pos_ori, position, sum) default(none) shared(public, private)
#endif
        for (ei_new = 0; ei_new < public.in2_pad_cols; ei_new++) {

            pos_ori = ei_new * public.in2_pad_rows;

            sum = 0;
            for (position = pos_ori; position < pos_ori + public.in2_pad_rows;
                 position++) {
                private.d_in2_pad[position] =
                    private.d_in2_pad[position] + sum;
                sum = private.d_in2_pad[position];
            }
        }

#ifdef _OPENMP
#pragma omp parallel for private(row, ori_row, ori_col, temp, temp2) default(none) shared(public, private)
#endif
        for (col = 0; col < public.in2_sub_cols; col++) {
            for (row = 0; row < public.in2_sub_rows; row++) {

                ori_row = row + public.in2_pad_cumv_sel_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel_collow - 1;
                temp =
                    private.d_in2_pad[ori_col * public.in2_pad_rows + ori_row];

                ori_row = row + public.in2_pad_cumv_sel2_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel2_collow - 1;
                temp2 =
                    private.d_in2_pad[ori_col * public.in2_pad_rows + ori_row];

                private.d_in2_sub[col * public.in2_sub_rows + row] =
                    temp - temp2;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for private(pos_ori, position, sum) default(none) shared(public, private)
#endif
        for (ei_new = 0; ei_new < public.in2_sub_rows; ei_new++) {

            pos_ori = ei_new;

            sum = 0;
            for (position = pos_ori; position < pos_ori + public.in2_sub_elem;
                 position += public.in2_sub_rows) {
                private.d_in2_sub[position] =
                    private.d_in2_sub[position] + sum;
                sum = private.d_in2_sub[position];
            }
        }

#ifdef _OPENMP
#pragma omp parallel for private(row, ori_row, ori_col, temp, temp2) default(none) shared(public, private, denomT)
#endif
        for (col = 0; col < public.conv_cols; col++) {
            for (row = 0; row < public.conv_rows; row++) {

                ori_row = row + public.in2_sub_cumh_sel_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel_collow - 1;
                temp =
                    private.d_in2_sub[ori_col * public.in2_sub_rows + ori_row];

                ori_row = row + public.in2_sub_cumh_sel2_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel2_collow - 1;
                temp2 =
                    private.d_in2_sub[ori_col * public.in2_sub_rows + ori_row];

                temp2 = temp - temp2;

                temp2 = temp2 -
                        (private.d_in2_sub2_sqr[col * public.conv_rows + row] /
                         public.in_mod_elem);

                if (temp2 < 0) {
                    temp2 = 0;
                }
                temp2 = sqrt(temp2);

                temp2 = denomT * temp2;

                private.d_conv[col * public.conv_rows + row] =
                    private.d_conv[col * public.conv_rows + row] / temp2;
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

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(public, private)
#endif
        for (ei_new = 0; ei_new < public.tMask_elem; ei_new++) {
            private.d_tMask[ei_new] = 0;
        }
        private.d_tMask[tMask_col * public.tMask_rows + tMask_row] = 1;

#ifdef _OPENMP
#pragma omp parallel for private(j,jp1,ja1,ja2,row,i,ip1,ia1,ia2,s,ja,jb,ia,ib) default(none) shared(public, private)
#endif
        for (col = 1; col <= public.mask_conv_cols; col++) {

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

            for (row = 1; row <= public.mask_conv_rows; row++) {

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
                    (void)jb;
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        (void)ib;
                        s += private.d_tMask[public.tMask_rows * (ja - 1) +
                                             ia - 1];
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

        largest_row =
            (fin_max_coo + 1) % public.mask_conv_rows - 1;
        largest_col = (fin_max_coo + 1) / public.mask_conv_rows;
        if ((fin_max_coo + 1) % public.mask_conv_rows == 0) {
            largest_row = public.mask_conv_rows - 1;
            largest_col = largest_col - 1;
        }

        largest_row = largest_row + 1;
        largest_col = largest_col + 1;
        offset_row =
            largest_row - public.in_mod_rows - (public.sSize - public.tSize);
        offset_col =
            largest_col - public.in_mod_cols - (public.sSize - public.tSize);
        pointer = private.point_no * public.frames + public.frame_no;
        private.d_tRowLoc[pointer] = private.d_Row[private.point_no] + offset_row;
        private.d_tColLoc[pointer] = private.d_Col[private.point_no] + offset_col;
    }

    if (public.frame_no != 0 && (public.frame_no) % 10 == 0) {

        loc_pointer = private.point_no * public.frames + public.frame_no;
        private.d_Row[private.point_no] = private.d_tRowLoc[loc_pointer];
        private.d_Col[private.point_no] = private.d_tColLoc[loc_pointer];

        d_in = &private.d_T[private.in_pointer];

#ifdef _OPENMP
#pragma omp parallel for private(row, ori_row, ori_col, ori_pointer, temp) default(none) shared(public, private, d_in)
#endif
        for (col = 0; col < public.in_mod_cols; col++) {
            for (row = 0; row < public.in_mod_rows; row++) {

                ori_row = private.d_Row[private.point_no] - 25 + row - 1;
                ori_col = private.d_Col[private.point_no] - 25 + col - 1;
                ori_pointer = ori_col * public.frame_rows + ori_row;

                temp = d_in[col * public.in_mod_rows + row];
                d_in[col * public.in_mod_rows + row] =
                    public.alpha * temp +
                    (1.00 - public.alpha) * public.d_frame[ori_pointer];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    heartwall_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
