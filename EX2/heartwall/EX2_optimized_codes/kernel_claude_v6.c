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
        const int base_row = private.d_Row[private.point_no] - 26;
        const int base_col = private.d_Col[private.point_no] - 26;
        const int frame_rows = public.frame_rows;
        const int in_mod_rows = public.in_mod_rows;
        
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(ori_row, ori_col, ori_pointer)
#endif
        for (col = 0; col < public.in_mod_cols; col++) {
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
        in2_rowlow = private.d_Row[private.point_no] - public.sSize;
        in2_collow = private.d_Col[private.point_no] - public.sSize;

        // work
        const int in2_rowlow_m1 = in2_rowlow - 1;
        const int in2_collow_m1 = in2_collow - 1;
        const int frame_rows = public.frame_rows;
        const int in2_rows = public.in2_rows;
        
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(ori_row, ori_col, temp)
#endif
        for (col = 0; col < public.in2_cols; col++) {
            for (row = 0; row < in2_rows; row++) {
                ori_row = row + in2_rowlow_m1;
                ori_col = col + in2_collow_m1;
                temp = public.d_frame[ori_col * frame_rows + ori_row];
                private.d_in2[col * in2_rows + row] = temp;
                private.d_in2_sqr[col * in2_rows + row] = temp * temp;
            }
        }

        // variables
        d_in = &private.d_T[private.in_pointer];

        // work
        const int in_mod_rows = public.in_mod_rows;
        const int in_mod_rows_m1 = in_mod_rows - 1;
        
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(rot_row, rot_col, pointer, temp)
#endif
        for (col = 0; col < public.in_mod_cols; col++) {
            for (row = 0; row < in_mod_rows; row++) {
                rot_row = in_mod_rows_m1 - row;
                rot_col = in_mod_rows_m1 - col;
                pointer = rot_col * in_mod_rows + rot_row;
                temp = d_in[pointer];
                private.d_in_mod[col * in_mod_rows + row] = temp;
                private.d_in_sqr[pointer] = temp * temp;
            }
        }

        in_final_sum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:in_final_sum)
#endif
        for (i = 0; i < public.in_mod_elem; i++) {
            in_final_sum += d_in[i];
        }

        in_sqr_final_sum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:in_sqr_final_sum)
#endif
        for (i = 0; i < public.in_mod_elem; i++) {
            in_sqr_final_sum += private.d_in_sqr[i];
        }

        mean = in_final_sum / public.in_mod_elem;
        mean_sqr = mean * mean;
        variance = (in_sqr_final_sum / public.in_mod_elem) - mean_sqr;
        deviation = sqrt(variance);

        denomT = sqrt((fp)(public.in_mod_elem - 1)) * deviation;

        // work
        const int conv_rows = public.conv_rows;
        const int joffset = public.joffset;
        const int ioffset = public.ioffset;
        const int in2_cols = public.in2_cols;
        const int in_mod_cols = public.in_mod_cols;
        const int in2_rows_c = public.in2_rows;
        const int in_mod_rows_c = public.in_mod_rows;
        
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(j, jp1, ja1, ja2, i, ip1, ia1, ia2, s, ja, jb, ia, ib)
#endif
        for (col = 1; col <= public.conv_cols; col++) {
            for (row = 1; row <= conv_rows; row++) {
                j = col + joffset;
                jp1 = j + 1;
                ja1 = (in2_cols < jp1) ? (jp1 - in2_cols) : 1;
                ja2 = (in_mod_cols < j) ? in_mod_cols : j;

                i = row + ioffset;
                ip1 = i + 1;
                ia1 = (in2_rows_c < ip1) ? (ip1 - in2_rows_c) : 1;
                ia2 = (in_mod_rows_c < i) ? in_mod_rows_c : i;

                s = 0;
                for (ja = ja1; ja <= ja2; ja++) {
                    jb = jp1 - ja;
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        s += private.d_in_mod[in_mod_rows_c * (ja - 1) + ia - 1] *
                             private.d_in2[in2_rows_c * (jb - 1) + ib - 1];
                    }
                }
                private.d_conv[(col - 1) * conv_rows + (row - 1)] = s;
            }
        }

        // work
        const int in2_pad_rows = public.in2_pad_rows;
        const int in2_pad_add_rows = public.in2_pad_add_rows;
        const int in2_pad_add_cols = public.in2_pad_add_cols;
        const int in2_pad_add_rows_p_in2_rows = in2_pad_add_rows + public.in2_rows;
        const int in2_pad_add_cols_p_in2_cols = in2_pad_add_cols + public.in2_cols;
        
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(ori_row, ori_col)
#endif
        for (col = 0; col < public.in2_pad_cols; col++) {
            for (row = 0; row < in2_pad_rows; row++) {
                if (row > (in2_pad_add_rows - 1) &&
                    row < in2_pad_add_rows_p_in2_rows &&
                    col > (in2_pad_add_cols - 1) &&
                    col < in2_pad_add_cols_p_in2_cols) {
                    ori_row = row - in2_pad_add_rows;
                    ori_col = col - in2_pad_add_cols;
                    private.d_in2_pad[col * in2_pad_rows + row] =
                        private.d_in2[ori_col * public.in2_rows + ori_row];
                } else {
                    private.d_in2_pad[col * in2_pad_rows + row] = 0;
                }
            }
        }

#ifdef _OPENMP
#pragma omp parallel for private(pos_ori, sum, position)
#endif
        for (ei_new = 0; ei_new < public.in2_pad_cols; ei_new++) {
            pos_ori = ei_new * in2_pad_rows;
            sum = 0;
            for (position = pos_ori; position < pos_ori + in2_pad_rows; position++) {
                private.d_in2_pad[position] += sum;
                sum = private.d_in2_pad[position];
            }
        }

        // work
        const int in2_sub_rows = public.in2_sub_rows;
        const int in2_pad_cumv_sel_rowlow_m1 = public.in2_pad_cumv_sel_rowlow - 1;
        const int in2_pad_cumv_sel_collow_m1 = public.in2_pad_cumv_sel_collow - 1;
        const int in2_pad_cumv_sel2_rowlow_m1 = public.in2_pad_cumv_sel2_rowlow - 1;
        const int in2_pad_cumv_sel2_collow_m1 = public.in2_pad_cumv_sel2_collow - 1;
        
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(ori_row, ori_col, temp, temp2)
#endif
        for (col = 0; col < public.in2_sub_cols; col++) {
            for (row = 0; row < in2_sub_rows; row++) {
                ori_row = row + in2_pad_cumv_sel_rowlow_m1;
                ori_col = col + in2_pad_cumv_sel_collow_m1;
                temp = private.d_in2_pad[ori_col * in2_pad_rows + ori_row];

                ori_row = row + in2_pad_cumv_sel2_rowlow_m1;
                ori_col = col + in2_pad_cumv_sel2_collow_m1;
                temp2 = private.d_in2_pad[ori_col * in2_pad_rows + ori_row];

                private.d_in2_sub[col * in2_sub_rows + row] = temp - temp2;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for private(pos_ori, sum, position)
#endif
        for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {
            pos_ori = ei_new;
            sum = 0;
            for (position = pos_ori; position < pos_ori + public.in2_sub_elem;
                 position += in2_sub_rows) {
                private.d_in2_sub[position] += sum;
                sum = private.d_in2_sub[position];
            }
        }

        // work
        const int in2_sub_cumh_sel_rowlow_m1 = public.in2_sub_cumh_sel_rowlow - 1;
        const int in2_sub_cumh_sel_collow_m1 = public.in2_sub_cumh_sel_collow - 1;
        const int in2_sub_cumh_sel2_rowlow_m1 = public.in2_sub_cumh_sel2_rowlow - 1;
        const int in2_sub_cumh_sel2_collow_m1 = public.in2_sub_cumh_sel2_collow - 1;
        const fp in_final_sum_div_in_mod_elem = in_final_sum / public.in_mod_elem;
        
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(ori_row, ori_col, temp, temp2)
#endif
        for (col = 0; col < public.in2_sub2_sqr_cols; col++) {
            for (row = 0; row < public.in2_sub2_sqr_rows; row++) {
                ori_row = row + in2_sub_cumh_sel_rowlow_m1;
                ori_col = col + in2_sub_cumh_sel_collow_m1;
                temp = private.d_in2_sub[ori_col * in2_sub_rows + ori_row];

                ori_row = row + in2_sub_cumh_sel2_rowlow_m1;
                ori_col = col + in2_sub_cumh_sel2_collow_m1;
                temp2 = private.d_in2_sub[ori_col * in2_sub_rows + ori_row];

                temp2 = temp - temp2;

                private.d_in2_sub2_sqr[col * public.in2_sub2_sqr_rows + row] = temp2 * temp2;
                private.d_conv[col * public.in2_sub2_sqr_rows + row] =
                    private.d_conv[col * public.in2_sub2_sqr_rows + row] -
                    temp2 * in_final
