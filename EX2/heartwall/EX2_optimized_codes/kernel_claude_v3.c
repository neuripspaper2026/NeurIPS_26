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
        int base_row = private.d_Row[private.point_no] - 26;
        int base_col = private.d_Col[private.point_no] - 26;
        
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(ori_row, ori_col, ori_pointer)
#endif
        for (col = 0; col < public.in_mod_cols; col++) {
            for (row = 0; row < public.in_mod_rows; row++) {
                ori_row = base_row + row;
                ori_col = base_col + col;
                ori_pointer = ori_col * public.frame_rows + ori_row;
                d_in[col * public.in_mod_rows + row] = public.d_frame[ori_pointer];
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
        int base_row_in2 = in2_rowlow - 1;
        int base_col_in2 = in2_collow - 1;
        
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(ori_row, ori_col, temp)
#endif
        for (col = 0; col < public.in2_cols; col++) {
            for (row = 0; row < public.in2_rows; row++) {
                ori_row = row + base_row_in2;
                ori_col = col + base_col_in2;
                temp = public.d_frame[ori_col * public.frame_rows + ori_row];
                private.d_in2[col * public.in2_rows + row] = temp;
                private.d_in2_sqr[col * public.in2_rows + row] = temp * temp;
            }
        }

        // variables
        d_in = &private.d_T[private.in_pointer];

        // work
        int in_mod_rows_m1 = public.in_mod_rows - 1;
        
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(rot_row, rot_col, pointer, temp)
#endif
        for (col = 0; col < public.in_mod_cols; col++) {
            for (row = 0; row < public.in_mod_rows; row++) {
                rot_row = in_mod_rows_m1 - row;
                rot_col = in_mod_rows_m1 - col;
                pointer = rot_col * public.in_mod_rows + rot_row;
                temp = d_in[pointer];
                private.d_in_mod[col * public.in_mod_rows + row] = temp;
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
#ifdef _OPENMP
#pragma omp parallel for private(j, jp1, ja1, ja2, row, i, ip1, ia1, ia2, s, ja, jb, ia, ib)
#endif
        for (col = 1; col <= public.conv_cols; col++) {
            j = col + public.joffset;
            jp1 = j + 1;
            ja1 = (public.in2_cols < jp1) ? (jp1 - public.in2_cols) : 1;
            ja2 = (public.in_mod_cols < j) ? public.in_mod_cols : j;

            for (row = 1; row <= public.conv_rows; row++) {
                i = row + public.ioffset;
                ip1 = i + 1;
                ia1 = (public.in2_rows < ip1) ? (ip1 - public.in2_rows) : 1;
                ia2 = (public.in_mod_rows < i) ? public.in_mod_rows : i;

                s = 0;
                for (ja = ja1; ja <= ja2; ja++) {
                    jb = jp1 - ja;
                    int ja_offset = public.in_mod_rows * (ja - 1);
                    int jb_offset = public.in2_rows * (jb - 1);
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        s += private.d_in_mod[ja_offset + ia - 1] * private.d_in2[jb_offset + ib - 1];
                    }
                }
                private.d_conv[(col - 1) * public.conv_rows + (row - 1)] = s;
            }
        }

        // work
        int in2_pad_add_rows = public.in2_pad_add_rows;
        int in2_pad_add_cols = public.in2_pad_add_cols;
        int in2_pad_add_rows_plus_in2_rows = in2_pad_add_rows + public.in2_rows;
        int in2_pad_add_cols_plus_in2_cols = in2_pad_add_cols + public.in2_cols;
        
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(ori_row, ori_col)
#endif
        for (col = 0; col < public.in2_pad_cols; col++) {
            for (row = 0; row < public.in2_pad_rows; row++) {
                if (row > (in2_pad_add_rows - 1) && row < in2_pad_add_rows_plus_in2_rows &&
                    col > (in2_pad_add_cols - 1) && col < in2_pad_add_cols_plus_in2_cols) {
                    ori_row = row - in2_pad_add_rows;
                    ori_col = col - in2_pad_add_cols;
                    private.d_in2_pad[col * public.in2_pad_rows + row] =
                        private.d_in2[ori_col * public.in2_rows + ori_row];
                } else {
                    private.d_in2_pad[col * public.in2_pad_rows + row] = 0;
                }
            }
        }

#ifdef _OPENMP
#pragma omp parallel for private(pos_ori, sum, position)
#endif
        for (ei_new = 0; ei_new < public.in2_pad_cols; ei_new++) {
            pos_ori = ei_new * public.in2_pad_rows;
            sum = 0;
            for (position = pos_ori; position < pos_ori + public.in2_pad_rows; position++) {
                private.d_in2_pad[position] += sum;
                sum = private.d_in2_pad[position];
            }
        }

        // work
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(ori_row, ori_col, temp, temp2)
#endif
        for (col = 0; col < public.in2_sub_cols; col++) {
            for (row = 0; row < public.in2_sub_rows; row++) {
                ori_row = row + public.in2_pad_cumv_sel_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel_collow - 1;
                temp = private.d_in2_pad[ori_col * public.in2_pad_rows + ori_row];

                ori_row = row + public.in2_pad_cumv_sel2_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel2_collow - 1;
                temp2 = private.d_in2_pad[ori_col * public.in2_pad_rows + ori_row];

                private.d_in2_sub[col * public.in2_sub_rows + row] = temp - temp2;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for private(pos_ori, sum, position)
#endif
        for (ei_new = 0; ei_new < public.in2_sub_rows; ei_new++) {
            pos_ori = ei_new;
            sum = 0;
            for (position = pos_ori; position < pos_ori + public.in2_sub_elem; position += public.in2_sub_rows) {
                private.d_in2_sub[position] += sum;
                sum = private.d_in2_sub[position];
            }
        }

        // work
        fp inv_in_mod_elem = 1.0 / public.in_mod_elem;
        
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(ori_row, ori_col, temp, temp2)
#endif
        for (col = 0; col < public.in2_sub2_sqr_cols; col++) {
            for (row = 0; row < public.in2_sub2_sqr_rows; row++) {
                ori_row = row + public.in2_sub_cumh_sel_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel_collow - 1;
                temp = private.d_in2_sub[ori_col * public.in2_sub_rows + ori_row];

                ori_row = row + public.in2_sub_cumh_sel2_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel2_collow - 1;
                temp2 = private.d_in2_sub[ori_col * public.in2_sub_rows + ori_row];

                temp2 = temp - temp2;
                private.d_in2_sub2_sqr[col * public.in2_sub2_sqr_rows + row] = temp2 * temp2;
                private.d_conv[col * public.in2_sub2_sqr_rows + row] -= temp2 * in_final_sum * inv_in_mod_elem;
            }
        }

        // work
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(ori_row, ori_col)
#endif
        for (col = 0; col < public.in2_pad_cols; col++) {
            for (row = 0; row < public.in2_pad_rows; row++) {
                if (row > (in2_pad_add_rows - 1) && row < in2_pad_add_rows_plus_in2_rows &&
                    col > (in2_pad_add_cols - 1) && col < in2_pad_add_cols_plus_in2_cols) {
                    ori_row = row - in2_pad_add_rows;
                    ori_col = col - in2_pad_add_cols;
                    private.d_in2_pad[col * public.in2_pad_rows + row] =
                        private.d_in2_sqr[ori_col * public.in2_rows + ori_row];
                } else {
                    private.d_in2_pad[col * public.in2_pad_rows + row] = 0;
                }
            }
        }

        // work
#ifdef _OPENMP
#pragma omp parallel for private(pos_ori, sum, position)
#endif
        for (ei_new = 0; ei_new < public.in2_pad_cols; ei_new++) {
            pos_ori = ei_new * public.in2_pad_rows;
            sum = 0;
            for (position = pos_ori; position < pos_ori + public.in2_pad_rows; position++) {
                private.d_in2_pad[position] += sum;
                sum = private.d_in2_pad[position];
            }
        }

        // work
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(ori_row, ori_col, temp, temp2)
#endif
        for (col = 0; col < public.in2_sub_cols; col++) {
