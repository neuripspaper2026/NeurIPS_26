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
        const int point_row = private.d_Row[private.point_no];
        const int point_col = private.d_Col[private.point_no];

        // update template, limit the number of working threads to the size of template
        #pragma omp parallel for collapse(2) default(none) shared(in_mod_cols,in_mod_rows,frame_rows,d_frame,d_T,point_row,point_col) private(col,row,ori_row,ori_col,ori_pointer) if(in_mod_cols*in_mod_rows > 256)
        for (col = 0; col < in_mod_cols; col++) {
            for (row = 0; row < in_mod_rows; row++) {

                ori_row = point_row - 25 + row - 1;
                ori_col = point_col - 25 + col - 1;
                ori_pointer = ori_col * frame_rows + ori_row;

                d_T[col * in_mod_rows + row] = d_frame[ori_pointer];
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

        fp *restrict d_in2 = private.d_in2;
        fp *restrict d_in2_sqr = private.d_in2_sqr;
        fp *restrict d_in_mod = private.d_in_mod;
        fp *restrict d_in_sqr = private.d_in_sqr;
        fp *restrict d_conv = private.d_conv;
        fp *restrict d_in2_pad = private.d_in2_pad;
        fp *restrict d_in2_sub = private.d_in2_sub;
        fp *restrict d_in2_sub2_sqr = private.d_in2_sub2_sqr;
        fp *restrict d_tMask = private.d_tMask;
        fp *restrict d_mask_conv = private.d_mask_conv;
        fp *restrict d_tRowLoc = private.d_tRowLoc;
        fp *restrict d_tColLoc = private.d_tColLoc;
        fp *restrict d_Row = private.d_Row;
        fp *restrict d_Col = private.d_Col;

        const fp *restrict d_frame = public.d_frame;

        in2_rowlow = d_Row[private.point_no] - public.sSize; // (1 to n+1)
        in2_collow = d_Col[private.point_no] - public.sSize;

        // work: build d_in2 and d_in2_sqr
        #pragma omp parallel for collapse(2) default(none) shared(in2_cols,in2_rows,in2_rowlow,in2_collow,frame_rows,d_frame,d_in2,d_in2_sqr) private(col,row,ori_row,ori_col,temp) if(in2_cols*in2_rows > 256)
        for (col = 0; col < in2_cols; col++) {
            for (row = 0; row < in2_rows; row++) {

                ori_row = row + in2_rowlow - 1;
                ori_col = col + in2_collow - 1;
                temp = d_frame[ori_col * frame_rows + ori_row];
                d_in2[col * in2_rows + row] = temp;
                d_in2_sqr[col * in2_rows + row] = temp * temp;
            }
        }

        // variables
        d_in = &private.d_T[private.in_pointer];
        fp *restrict d_T = d_in;

        // work: build d_in_mod and d_in_sqr
        {
            const int rows = in_mod_rows;
            const int cols = in_mod_cols;
            #pragma omp parallel for collapse(2) default(none) shared(rows,cols,d_T,d_in_mod,d_in_sqr) private(col,row,rot_row,rot_col,pointer,temp) if(rows*cols > 256)
            for (col = 0; col < cols; col++) {
                for (row = 0; row < rows; row++) {

                    rot_row = (rows - 1) - row;
                    rot_col = (rows - 1) - col;
                    pointer = rot_col * rows + rot_row;

                    temp = d_T[pointer];
                    d_in_mod[col * rows + row] = temp;
                    d_in_sqr[pointer] = temp * temp;
                }
            }
        }

        in_final_sum = 0;
        #pragma omp parallel for reduction(+:in_final_sum) default(none) shared(in_mod_elem,d_T) if(in_mod_elem > 256)
        for (i = 0; i < in_mod_elem; i++) {
            in_final_sum += d_T[i];
        }

        in_sqr_final_sum = 0;
        #pragma omp parallel for reduction(+:in_sqr_final_sum) default(none) shared(in_mod_elem,d_in_sqr) if(in_mod_elem > 256)
        for (i = 0; i < in_mod_elem; i++) {
            in_sqr_final_sum += d_in_sqr[i];
        }

        mean = in_final_sum / in_mod_elem;
        mean_sqr = mean * mean;
        variance = (in_sqr_final_sum / in_mod_elem) - mean_sqr;
        deviation = sqrt(variance);
        denomT = sqrt((fp)(in_mod_elem - 1)) * deviation;

        // work: main convolution
        const int joffset = public.joffset;
        const int ioffset = public.ioffset;

        #pragma omp parallel for default(none) shared(conv_cols,conv_rows,joffset,ioffset,in2_cols,in2_rows,in_mod_cols,in_mod_rows,d_in_mod,d_in2,d_conv) private(col,j,jp1,ja1,ja2,row,i,ip1,ia1,ia2,ja,jb,ia,ib,s) if((long)conv_cols*conv_rows > 64)
        for (col = 1; col <= conv_cols; col++) {

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

                for (ja = ja1; ja <= ja2; ja++) {
                    jb = jp1 - ja;
                    for (ia = ia1; ia <= ia2; ia++) {
                        ib = ip1 - ia;
                        s += d_in_mod[in_mod_rows * (ja - 1) + ia - 1] *
                             d_in2[in2_rows * (jb - 1) + ib - 1];
                    }
                }

                d_conv[(col - 1) * conv_rows + (row - 1)] = s;
            }
        }

        // work: pad in2 into in2_pad (from d_in2)
        #pragma omp parallel for collapse(2) default(none) shared(in2_pad_cols,in2_pad_rows,public,d_in2_pad,d_in2) private(col,row,ori_row,ori_col) if(in2_pad_cols*in2_pad_rows > 256)
        for (col = 0; col < in2_pad_cols; col++) {
            for (row = 0; row < in2_pad_rows; row++) {

                if (row > (public.in2_pad_add_rows - 1) &&
                    row < (public.in2_pad_add_rows + public.in2_rows) &&
                    col > (public.in2_pad_add_cols - 1) &&
                    col < (public.in2_pad_add_cols + public.in2_cols)) {
                    ori_row = row - public.in2_pad_add_rows;
                    ori_col = col - public.in2_pad_add_cols;
                    d_in2_pad[col * in2_pad_rows + row] =
                        d_in2[ori_col * in2_rows + ori_row];
                } else {
                    d_in2_pad[col * in2_pad_rows + row] = 0;
                }
            }
        }

        // column-wise cumulative sum on d_in2_pad
        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {
            pos_ori = ei_new * in2_pad_rows;
            sum = 0;
            for (position = pos_ori;
                 position < pos_ori + in2_pad_rows;
                 position++) {
                d_in2_pad[position] = d_in2_pad[position] + sum;
                sum = d_in2_pad[position];
            }
        }

        // work: build d_in2_sub from d_in2_pad (first phase)
        #pragma omp parallel for collapse(2) default(none) shared(in2_sub_cols,in2_sub_rows,public,d_in2_pad,d_in2_sub,in2_pad_rows) private(col,row,ori_row,ori_col,temp,temp2) if(in2_sub_cols*in2_sub_rows > 256)
        for (col = 0; col < in2_sub_cols; col++) {
            for (row = 0; row < in2_sub_rows; row++) {

                ori_row = row + public.in2_pad_cumv_sel_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel_collow - 1;
                temp = d_in2_pad[ori_col * in2_pad_rows + ori_row];

                ori_row = row + public.in2_pad_cumv_sel2_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel2_collow - 1;
                temp2 = d_in2_pad[ori_col * in2_pad_rows + ori_row];

                d_in2_sub[col * in2_sub_rows + row] = temp - temp2;
            }
        }

        // row-wise cumulative sum on d_in2_sub (first phase)
        for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {
            pos_ori = ei_new;
            sum = 0;
            for (position = pos_ori;
                 position < pos_ori + in2_sub_elem;
                 position += in2_sub_rows) {
                d_in2_sub[position] = d_in2_sub[position] + sum;
                sum = d_in2_sub[position];
            }
        }

        // work: build d_in2_sub2_sqr and update numerator in d_conv
        #pragma omp parallel for collapse(2) default(none) shared(in2_sub2_sqr_cols,in2_sub2_sqr_rows,public,d_in2_sub,d_in2_sub2_sqr,d_conv,in_final_sum,in_mod_elem) private(col,row,ori_row,ori_col,temp,temp2) if(in2_sub2_sqr_cols*in2_sub2_sqr_rows > 256)
        for (col = 0; col < in2_sub2_sqr_cols; col++) {
            for (row = 0; row < in2_sub2_sqr_rows; row++) {

                ori_row = row + public.in2_sub_cumh_sel_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel_collow - 1;
                temp = d_in2_sub[ori_col * in2_sub_rows + ori_row];

                ori_row = row + public.in2_sub_cumh_sel2_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel2_collow - 1;
                temp2 = d_in2_sub[ori_col * in2_sub_rows + ori_row];

                temp2 = temp - temp2;

                d_in2_sub2_sqr[col * in2_sub2_sqr_rows + row] = temp2 * temp2;

                d_conv[col * in2_sub2_sqr_rows + row] =
                    d_conv[col * in2_sub2_sqr_rows + row] -
                    temp2 * in_final_sum / in_mod_elem;
            }
        }

        // work: pad in2_sqr into in2_pad
        #pragma omp parallel for collapse(2) default(none) shared(in2_pad_cols,in2_pad_rows,public,d_in2_pad,d_in2_sqr,in2_rows) private(col,row,ori_row,ori_col) if(in2_pad_cols*in2_pad_rows > 256)
        for (col = 0; col < in2_pad_cols; col++) {
            for (row = 0; row < in2_pad_rows; row++) {

                if (row > (public.in2_pad_add_rows - 1) &&
                    row < (public.in2_pad_add_rows + public.in2_rows) &&
                    col > (public.in2_pad_add_cols - 1) &&
                    col < (public.in2_pad_add_cols + public.in2_cols)) {
                    ori_row = row - public.in2_pad_add_rows;
                    ori_col = col - public.in2_pad_add_cols;
                    d_in2_pad[col * in2_pad_rows + row] =
                        d_in2_sqr[ori_col * in2_rows + ori_row];
                } else {
                    d_in2_pad[col * in2_pad_rows + row] = 0;
                }
            }
        }

        // column-wise cumulative sum on d_in2_pad (squared)
        for (ei_new = 0; ei_new < in2_pad_cols; ei_new++) {
            pos_ori = ei_new * in2_pad_rows;
            sum = 0;
            for (position = pos_ori;
                 position < pos_ori + in2_pad_rows;
                 position++) {
                d_in2_pad[position] = d_in2_pad[position] + sum;
                sum = d_in2_pad[position];
            }
        }

        // work: build d_in2_sub from d_in2_pad (second phase)
        #pragma omp parallel for collapse(2) default(none) shared(in2_sub_cols,in2_sub_rows,public,d_in2_pad,d_in2_sub,in2_pad_rows) private(col,row,ori_row,ori_col,temp,temp2) if(in2_sub_cols*in2_sub_rows > 256)
        for (col = 0; col < in2_sub_cols; col++) {
            for (row = 0; row < in2_sub_rows; row++) {

                ori_row = row + public.in2_pad_cumv_sel_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel_collow - 1;
                temp = d_in2_pad[ori_col * in2_pad_rows + ori_row];

                ori_row = row + public.in2_pad_cumv_sel2_rowlow - 1;
                ori_col = col + public.in2_pad_cumv_sel2_collow - 1;
                temp2 = d_in2_pad[ori_col * in2_pad_rows + ori_row];

                d_in2_sub[col * in2_sub_rows + row] = temp - temp2;
            }
        }

        // row-wise cumulative sum on d_in2_sub (second phase)
        for (ei_new = 0; ei_new < in2_sub_rows; ei_new++) {
            pos_ori = ei_new;
            sum = 0;
            for (position = pos_ori;
                 position < pos_ori + in2_sub_elem;
                 position += in2_sub_rows) {
                d_in2_sub[position] = d_in2_sub[position] + sum;
                sum = d_in2_sub[position];
            }
        }

        // work: final normalization of d_conv
        #pragma omp parallel for collapse(2) default(none) shared(conv_cols,conv_rows,public,d_in2_sub,d_in2_sub2_sqr,d_conv,denomT,in_mod_elem,in2_sub_rows) private(col,row,ori_row,ori_col,temp,temp2) if((long)conv_cols*conv_rows > 64)
        for (col = 0; col < conv_cols; col++) {
            for (row = 0; row < conv_rows; row++) {

                ori_row = row + public.in2_sub_cumh_sel_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel_collow - 1;
                temp = d_in2_sub[ori_col * in2_sub_rows + ori_row];

                ori_row = row + public.in2_sub_cumh_sel2_rowlow - 1;
                ori_col = col + public.in2_sub_cumh_sel2_collow - 1;
                temp2 = d_in2_sub[ori_col * in2_sub_rows + ori_row];

                temp2 = temp - temp2;

                temp2 = temp2 -
                        (d_in2_sub2_sqr[col * conv_rows + row] /
                         in_mod_elem);

                if (temp2 < 0) {
                    temp2 = 0;
                }
                temp2 = sqrt(temp2);

                temp2 = denomT * temp2;

                d_conv[col * conv_rows + row] =
                    d_conv[col * conv_rows + row] / temp2;
            }
        }

        //====================================================================================================
        //	TEMPLATE MASK CREATE
        //====================================================================================================

        cent = public.sSize + public.tSize + 1;
        pointer = public.frame_no - 1 + private.point_no * public.frames;
        tMask_row = cent + d_tRowLoc[pointer] -
                    d_Row[private.point_no] - 1;
        tMask_col = cent + d_tColLoc[pointer] -
                    d_Col[private.point_no] - 1;

        // work: clear tMask
        #pragma omp parallel for default(none) shared(tMask_elem,d_tMask) if(tMask_elem > 256)
        for (ei_new = 0; ei_new < tMask_elem; ei_new++) {
            d_tMask[ei_new] = 0;
        }
        d_tMask[tMask_col * tMask_rows + tMask_row] = 1;

        const int mask_cols = public.mask_cols;
        const int mask_rows = public.mask_rows;
        const int mask_conv_joffset = public.mask_conv_joffset;
        const int mask_conv_ioffset = public.mask_conv_ioffset;

        // work: masked convolution
        #pragma omp parallel for default(none) shared(mask_conv_cols,mask_conv_rows,mask_cols,mask_rows,tMask_cols,tMask_rows,mask_conv_joffset,mask_conv_ioffset,d_tMask,d_mask_conv,d_conv,conv_rows) private(col,j,jp1,ja1,ja2,row,i,ip1,ia1,ia2,ja,jb,ia,ib,s) if((long)mask_conv_cols*mask_conv_rows > 64)
        for (col = 1; col <= mask_conv_cols; col++) {

            j = col + mask_conv_joffset;
            jp1 = j + 1;
            if (mask_cols < jp1) {
                ja1 = jp1 - mask_cols;
            } else {
                ja1 = 1;
            }
            if (tMask_cols < j) {
                ja2 = tMask_cols;
            } else {
                ja2 = j;
            }

            for (row = 1; row <= mask_conv_rows; row++) {

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
        for (i = 0; i < mask_conv_elem; i++) {
            if (d_mask_conv[i] > fin_max_val) {
                fin_max_val = d_mask_conv[i];
                fin_max_coo = i;
            }
        }

        largest_row =
            (fin_max_coo + 1) % mask_conv_rows - 1;
        largest_col = (fin_max_coo + 1) / mask_conv_rows;
        if ((fin_max_coo + 1) % mask_conv_rows == 0) {
            largest_row = mask_conv_rows - 1;
            largest_col = largest_col - 1;
        }

        largest_row = largest_row + 1;
        largest_col = largest_col + 1;
        offset_row =
            largest_row - in_mod_rows - (public.sSize - public.tSize);
        offset_col =
            largest_col - in_mod_cols - (public.sSize - public.tSize);
        pointer = private.point_no * public.frames + public.frame_no;
        d_tRowLoc[pointer] = d_Row[private.point_no] + offset_row;
        d_tColLoc[pointer] = d_Col[private.point_no] + offset_col;
    }

    // if the last frame in the bath, update template
    if (public.frame_no != 0 && (public.frame_no) % 10 == 0) {

        const int in_mod_rows = public.in_mod_rows;
        const int in_mod_cols = public.in_mod_cols;
        const int frame_rows = public.frame_rows;

        fp *restrict d_T = &private.d_T[private.in_pointer];
        fp *restrict d_Row = private.d_Row;
        fp *restrict d_Col = private.d_Col;
        fp *restrict d_tRowLoc = private.d_tRowLoc;
        fp *restrict d_tColLoc = private.d_tColLoc;
        const fp *restrict d_frame = public.d_frame;

        loc_pointer = private.point_no * public.frames + public.frame_no;
        d_Row[private.point_no] = d_tRowLoc[loc_pointer];
        d_Col[private.point_no] = d_tColLoc[loc_pointer];

        const int point_row = d_Row[private.point_no];
        const int point_col = d_Col[private.point_no];
        const fp alpha = public.alpha;

        #pragma omp parallel for collapse(2) default(none) shared(in_mod_cols,in_mod_rows,frame_rows,d_frame,d_T,point_row,point_col,alpha) private(col,row,ori_row,ori_col,ori_pointer,temp) if(in_mod_cols*in_mod_rows > 256)
        for (col = 0; col < in_mod_cols; col++) {
            for (row = 0; row < in_mod_rows; row++) {

                ori_row = point_row - 25 + row - 1;
                ori_col = point_col - 25 + col - 1;
                ori_pointer = ori_col * frame_rows + ori_row;

                temp = d_T[col * in_mod_rows + row];
                d_T[col * in_mod_rows + row] =
                    alpha * temp +
                    (1.00 - alpha) * d_frame[ori_pointer];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    heartwall_kernel_time += (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
