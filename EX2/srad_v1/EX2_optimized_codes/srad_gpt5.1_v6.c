#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "define.c"
#include "graphics.c"
#include "resize.c"
#include "timer.c"

int main(int argc, char *argv[]) {
    struct timespec main_start, main_end;
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &main_start);
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    long long time0;
    long long time1;
    long long time2;
    long long time3;
    long long time4;
    long long time5;
    long long time6;
    long long time7;
    long long time8;
    long long time9;
    long long time10;

    time0 = get_time();

    fp *image_ori;
    int image_ori_rows;
    int image_ori_cols;
    long image_ori_elem;

    fp *image;
    long Nr, Nc;
    long Ne;

    int niter;
    fp lambda;

    int r1, r2, c1, c2;
    long NeROI;

    fp meanROI, varROI, q0sqr;

    int *iN, *iS, *jE, *jW;

    fp Jc;

    fp *dN, *dS, *dW, *dE;

    fp tmp, sum, sum2;
    fp G2, L, num, den, qsqr, D;

    fp *c;
    fp cN, cS, cW, cE;

    int iter;
    long i, j;
    long k;

    int threads;

    time1 = get_time();

    int arg_error = 0;
    if (argc != 7) {
        printf("ERROR: wrong number of arguments\n");
        arg_error = 1;
    } else {
        niter = atoi(argv[1]);
        lambda = atof(argv[2]);
        Nr = atoi(argv[3]);
        Nc = atoi(argv[4]);
        threads = atoi(argv[5]);
    }

    (void)threads;

    time2 = get_time();

    if (arg_error) {
        clock_gettime(CLOCK_MONOTONIC, &kernel_end);
        clock_gettime(CLOCK_MONOTONIC, &main_end);
        double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
        double main_time = (main_end.tv_sec - main_start.tv_sec) +
                           (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

        FILE *timing_file = stderr;
        const char *timing_path = getenv("TIMING_LOG_FILE");
        if (timing_path && timing_path[0] != '\0') {
            FILE *tmpf = fopen(timing_path, "w");
            if (tmpf)
                timing_file = tmpf;
        }

        fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
        fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

        if (timing_file != stderr)
            fclose(timing_file);
        return 0;
    }

    image_ori_rows = 502;
    image_ori_cols = 458;
    image_ori_elem = (long)image_ori_rows * (long)image_ori_cols;

    image_ori = (fp *)malloc(sizeof(fp) * image_ori_elem);

    read_graphics("${REPO_ROOT}/EX1/srad_v1/input_data/image.pgm", image_ori, image_ori_rows,
                  image_ori_cols, 1);

    time3 = get_time();

    Ne = Nr * Nc;

    image = (fp *)malloc(sizeof(fp) * Ne);

    resize(image_ori, image_ori_rows, image_ori_cols, image, Nr, Nc, 1);

    time4 = get_time();

    r1 = 0;
    r2 = (int)(Nr - 1);
    c1 = 0;
    c2 = (int)(Nc - 1);

    NeROI = (long)(r2 - r1 + 1) * (long)(c2 - c1 + 1);

    iN = (int *)malloc(sizeof(int) * Nr);
    iS = (int *)malloc(sizeof(int) * Nr);
    jW = (int *)malloc(sizeof(int) * Nc);
    jE = (int *)malloc(sizeof(int) * Nc);

    dN = (fp *)malloc(sizeof(fp) * Ne);
    dS = (fp *)malloc(sizeof(fp) * Ne);
    dW = (fp *)malloc(sizeof(fp) * Ne);
    dE = (fp *)malloc(sizeof(fp) * Ne);

    c = (fp *)malloc(sizeof(fp) * Ne);

    for (i = 0; i < Nr; i++) {
        int ii = (int)i;
        iN[i] = ii - 1;
        iS[i] = ii + 1;
    }
    for (j = 0; j < Nc; j++) {
        int jj = (int)j;
        jW[j] = jj - 1;
        jE[j] = jj + 1;
    }
    iN[0] = 0;
    iS[Nr - 1] = (int)(Nr - 1);
    jW[0] = 0;
    jE[Nc - 1] = (int)(Nc - 1);

    time5 = get_time();

    {
        const long n = Ne;
        fp *restrict img = image;
        #pragma omp parallel for if(n > 10000) schedule(static)
        for (long idx = 0; idx < n; idx++) {
            img[idx] = expf(img[idx] / 255.0f);
        }
    }

    time6 = get_time();

    for (iter = 0; iter < niter; iter++) {

        sum = 0.0f;
        sum2 = 0.0f;
        {
            fp sum_local = 0.0f;
            fp sum2_local = 0.0f;
            const long nr = Nr;
            const long c1l = c1;
            const long c2l = c2;
            const long r1l = r1;
            const long r2l = r2;
            fp *restrict img = image;
            #pragma omp parallel for reduction(+:sum_local,sum2_local) if(NeROI > 10000) schedule(static)
            for (long jj = c1l; jj <= c2l; jj++) {
                long base = nr * jj;
                for (long ii = r1l; ii <= r2l; ii++) {
                    fp val = img[ii + base];
                    sum_local += val;
                    sum2_local += val * val;
                }
            }
            sum = sum_local;
            sum2 = sum2_local;
        }

        meanROI = sum / (fp)NeROI;
        varROI = (sum2 / (fp)NeROI) - meanROI * meanROI;
        q0sqr = varROI / (meanROI * meanROI);

        {
            const long nr = Nr;
            const long nc = Nc;
            fp *restrict img = image;
            fp *restrict dN_local = dN;
            fp *restrict dS_local = dS;
            fp *restrict dW_local = dW;
            fp *restrict dE_local = dE;
            fp *restrict c_local = c;
            int *restrict iN_local = iN;
            int *restrict iS_local = iS;
            int *restrict jW_local = jW;
            int *restrict jE_local = jE;
            fp q0sqr_local = q0sqr;

            #pragma omp parallel for collapse(2) if(Ne > 10000) schedule(static)
            for (long jj = 0; jj < nc; jj++) {
                for (long ii = 0; ii < nr; ii++) {

                    long k_local = ii + nr * jj;
                    fp Jc_local = img[k_local];

                    long iN_idx = iN_local[ii] + nr * jj;
                    long iS_idx = iS_local[ii] + nr * jj;
                    long jW_idx = ii + nr * jW_local[jj];
                    long jE_idx = ii + nr * jE_local[jj];

                    fp dN_val = img[iN_idx] - Jc_local;
                    fp dS_val = img[iS_idx] - Jc_local;
                    fp dW_val = img[jW_idx] - Jc_local;
                    fp dE_val = img[jE_idx] - Jc_local;

                    dN_local[k_local] = dN_val;
                    dS_local[k_local] = dS_val;
                    dW_local[k_local] = dW_val;
                    dE_local[k_local] = dE_val;

                    fp Jc_inv = 1.0f / Jc_local;
                    fp G2_local = (dN_val * dN_val + dS_val * dS_val +
                                   dW_val * dW_val + dE_val * dE_val) *
                                  (Jc_inv * Jc_inv);

                    fp L_local = (dN_val + dS_val + dW_val + dE_val) * Jc_inv;

                    fp num_local = 0.5f * G2_local - (1.0f / 16.0f) * (L_local * L_local);
                    fp den_local = 1.0f + 0.25f * L_local;
                    fp qsqr_local = num_local / (den_local * den_local);

                    den_local = (qsqr_local - q0sqr_local) /
                                (q0sqr_local * (1.0f + q0sqr_local));
                    fp c_val = 1.0f / (1.0f + den_local);

                    if (c_val < 0.0f) {
                        c_val = 0.0f;
                    } else if (c_val > 1.0f) {
                        c_val = 1.0f;
                    }

                    c_local[k_local] = c_val;
                }
            }
        }

        {
            const long nr = Nr;
            const long nc = Nc;
            fp *restrict img = image;
            fp *restrict dN_local = dN;
            fp *restrict dS_local = dS;
            fp *restrict dW_local = dW;
            fp *restrict dE_local = dE;
            fp *restrict c_local = c;
            int *restrict iS_local = iS;
            int *restrict jE_local = jE;
            fp lambda_local = lambda;

            #pragma omp parallel for collapse(2) if(Ne > 10000) schedule(static)
            for (long jj = 0; jj < nc; jj++) {
                for (long ii = 0; ii < nr; ii++) {

                    long k_local = ii + nr * jj;

                    fp cN_local = c_local[k_local];
                    fp cS_local = c_local[iS_local[ii] + nr * jj];
                    fp cW_local = c_local[k_local];
                    fp cE_local = c_local[ii + nr * jE_local[jj]];

                    fp D_local = cN_local * dN_local[k_local] +
                                 cS_local * dS_local[k_local] +
                                 cW_local * dW_local[k_local] +
                                 cE_local * dE_local[k_local];

                    img[k_local] = img[k_local] + 0.25f * lambda_local * D_local;
                }
            }
        }
    }

    time7 = get_time();

    {
        const long n = Ne;
        fp *restrict img = image;
        #pragma omp parallel for if(n > 10000) schedule(static)
        for (long idx = 0; idx < n; idx++) {
            img[idx] = logf(img[idx]) * 255.0f;
        }
    }

    time8 = get_time();

    write_graphics(argv[6], image, Nr, Nc, 1, 255);

    time9 = get_time();

    free(image_ori);
    free(image);

    free(iN);
    free(iS);
    free(jW);
    free(jE);
    free(dN);
    free(dS);
    free(dW);
    free(dE);
    free(c);

    time10 = get_time();

    printf("Time spent in different stages of the application:\n");
    printf("%.12f s, %.12f % : SETUP VARIABLES\n",
           (float)(time1 - time0) / 1000000,
           (float)(time1 - time0) / (float)(time10 - time0) * 100);
    printf("%.12f s, %.12f % : READ COMMAND LINE PARAMETERS\n",
           (float)(time2 - time1) / 1000000,
           (float)(time2 - time1) / (float)(time10 - time0) * 100);
    printf("%.12f s, %.12f % : READ IMAGE FROM FILE\n",
           (float)(time3 - time2) / 1000000,
           (float)(time3 - time2) / (float)(time10 - time0) * 100);
    printf("%.12f s, %.12f % : RESIZE IMAGE\n",
           (float)(time4 - time3) / 1000000,
           (float)(time4 - time3) / (float)(time10 - time0) * 100);
    printf("%.12f s, %.12f % : SETUP, MEMORY ALLOCATION\n",
           (float)(time5 - time4) / 1000000,
           (float)(time5 - time4) / (float)(time10 - time0) * 100);
    printf("%.12f s, %.12f % : EXTRACT IMAGE\n",
           (float)(time6 - time5) / 1000000,
           (float)(time6 - time5) / (float)(time10 - time0) * 100);
    printf("%.12f s, %.12f % : COMPUTE\n", (float)(time7 - time6) / 1000000,
           (float)(time7 - time6) / (float)(time10 - time0) * 100);
    printf("%.12f s, %.12f % : COMPRESS IMAGE\n",
           (float)(time8 - time7) / 1000000,
           (float)(time8 - time7) / (float)(time10 - time0) * 100);
    printf("%.12f s, %.12f % : SAVE IMAGE INTO FILE\n",
           (float)(time9 - time8) / 1000000,
           (float)(time9 - time8) / (float)(time10 - time0) * 100);
    printf("%.12f s, %.12f % : FREE MEMORY\n",
           (float)(time10 - time9) / 1000000,
           (float)(time10 - time9) / (float)(time10 - time0) * 100);
    printf("Total time:\n");
    printf("%.12f s\n", (float)(time10 - time0) / 1000000);

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    clock_gettime(CLOCK_MONOTONIC, &main_end);
    double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                         (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    double main_time = (main_end.tv_sec - main_start.tv_sec) +
                       (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

    FILE *timing_file = stderr;
    const char *timing_path = getenv("TIMING_LOG_FILE");
    if (timing_path && timing_path[0] != '\0') {
        FILE *tmp = fopen(timing_path, "w");
        if (tmp)
            timing_file = tmp;
    }

    fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

    if (timing_file != stderr)
        fclose(timing_file);
}
