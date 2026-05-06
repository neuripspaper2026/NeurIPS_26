<<<CODE>>>
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

        // update template, limit the number of working threads to the size of
        // template
        const int in_mod_rows = public.in_mod_rows;
        const int in_mod_cols = public.in_mod_cols;
        const int frame_rows = public.frame_rows;
        const fp* d_frame = public.d_frame;
        const int point_row = private.d_Row[private.point_no];
        const int point_col = private.d_Col[private.point_no];
        
        for (col = 0; col < in_mod_cols; col++) {
            const int base_col = point_col - 26 + col;
            const int col_offset = col * in_mod_rows;
            for (row = 0; row < in_mod_rows; row++) {
                const int base_row = point_row - 26 + row;
                const int ori_pointer = base_col * frame_rows + base_row;
                d_in[col_offset + row] = d_frame[ori_pointer];
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

        const int in2_rows = public.in2_rows;
        const int in2_cols = public.in2_cols;
        const int frame_rows = public.frame_rows;
        const fp* d_frame = public.d_frame;
        fp* d_in2 = private.d_in2;
        fp* d_in2_sqr = private.d_in2_sqr;
        
        // work
        for (col = 0; col < in2_cols; col++) {
            const int base_col = col + in2_collow - 1;
            const int col_offset = col * in2_rows;
            for (row = 0; row < in2_rows; row++) {
                const int base_row = row + in2_rowlow - 1;
                const int ori_pointer = base_col * frame_rows + base_row;
                const fp temp = d_frame[ori_pointer];
                const int idx = col_offset + row;
                d_in2[idx] = temp;
                d_in2_sqr[idx] = temp * temp;
            }
        }

        // variables
        d_in = &private.d_T[private.in_pointer];

        const int in_mod_rows = public.in_mod_rows;
        const int in_mod_cols = public.in_mod_cols;
        
        // work
        for (col = 0; col < in_mod_cols; col++) {
            const int rot_col = (in_mod_rows - 1) - col;
            const int col_offset = col * in_mod_rows;
            const int rot_col_offset = rot_col * in_mod_rows;
            for (row = 0; row < in_mod_rows; row++) {
                const int rot_row = (in_mod_rows - 1) - row;
                const int pointer = rot_col_offset + rot_row;
                const fp temp = d_in[pointer];
                private.d_in_mod[col_offset + row] = temp;
                private.d_in_sqr[pointer] = temp * temp;
            }
        }

        // Calculate sums using loop unrolling
        in_final_sum = 0;
        in_sqr_final_sum = 0;
        const int in_mod_elem = public.in_mod_elem;
        
        // Process 4 elements at a time for better cache usage
        i = 0;
        for (; i < (in_mod_elem & ~3); i += 4) {
            in_final_sum += d_in[i] + d_in[i+1] + d_in[i+2] + d_in[i+3];
            in_sqr_final_sum += private.d_in_sqr[i] + private.d_in_sqr[i+1] + 
                               private.d_in_sqr[i+2] + private.d_in_sqr[i+3];
        }
        // Handle remaining elements
        for (; i < in_mod_elem; i++) {
            in_final_sum += d_in[i];
            in_sqr_final_sum += private.d_in_sqr[i];
        }

        mean = in_final_sum / in_mod_elem;
        mean_sqr = mean * mean;
        variance = (in_sqr_final_sum / in_mod_elem) - mean_sqr;
        deviation = sqrtf(variance);
        denomT = sqrtf((fp)(in_mod_elem - 1)) * deviation;

        // work
        const int conv_rows = public.conv_rows;
        const int conv_cols = public.conv_cols;
        const int joffset = public.joffset;
        const int ioffset = public.ioffset;
        const int in2_pad_rows = public.in2_pad_rows;
        const int in2_pad_cols = public.in2_pad_cols;
        fp* d_conv = private.d_conv;
        fp* d_in_mod = private.d_in_mod;
        fp* d_in2_pad = private.d_in2_pad;
        fp* d_in2_sub = private.d_in2_sub;
        fp* d_in2_sub2_sqr = private.d_in2_sub2_sqr;
        
        for (col = 1; col <= conv_cols; col++) {
            j = col + joffset;
            jp1 = j + 1;
            ja1 = (in2_cols < jp1) ? (jp1 - in2_cols) : 1;
            ja2 = (in_mod_cols < j) ? in_mod_cols : j;

            for (row = 1; row <= conv_rows; row++) {
                i = row + ioffset;
                ip1 = i + 1;
                ia1 = (in2_rows < ip1) ? (ip1 - in2_rows) : 1;
                ia2 = (in_mod_rows < i) ? in_mod_rows : i;

                s = 0;
                for (ja = ja1; ja <= ja2; ja++) {
                    jb = jp1 - ja;
                    const int ja_idx = in_mod_rows * (ja - 1);
                    const int jb_idx = in2_rows * (jb - 1);
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        s += d_in_mod[ja_idx + ia - 1] * d_in2[jb_idx + ib - 1];
                    }
                }
                d_conv[(col - 1) * conv_rows + (row - 1)] = s;
            }
        }

        // work
        const int in2_pad_add_rows = public.in2_pad_add_rows;
        const int in2_pad_add_cols = public.in2_pad_add_cols;
        const int in2_pad_cumv_sel_rowlow = public.in2_pad_cumv_sel_rowlow;
        const int in2_pad_cumv_sel_collow = public.in2_pad_cumv_sel_collow;
        const int in2_pad_cumv_sel2_rowlow = public.in2_pad_cumv_sel2_rowlow;
        const int in2_pad_cumv_sel2_collow = public.in2_pad_cumv_sel2_collow;
        const int in2_sub_cols = public.in2_sub_cols;
