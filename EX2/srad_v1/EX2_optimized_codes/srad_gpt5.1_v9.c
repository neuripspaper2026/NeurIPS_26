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

    fp *image_ori; // original input image
    int image_ori_rows;
    int image_ori_cols;
    long image_ori_elem;

    fp *image;   // input image
    long Nr, Nc; // IMAGE nbr of rows/cols/elements
    long Ne;

    // algorithm parameters
    int niter; // nbr of iterations
    fp lambda; // update step size

    // size of IMAGE
    int r1, r2, c1, c2; // row/col coordinates of uniform ROI
    long NeROI;         // ROI nbr of elements

    // ROI statistics
    fp meanROI, varROI, q0sqr; // local region statistics

    // surrounding pixel indices
    int *iN, *iS, *jE, *jW;

    // center pixel value
    fp Jc;

    // directional derivatives
    fp *dN, *dS, *dW, *dE;

    // calculation variables
    fp tmp, sum, sum2;
    fp G2, L, num, den, qsqr, D;

    // diffusion coefficient
    fp *c;
    fp cN, cS, cW, cE;

    // counters
    int iter;  // primary loop
    long i, j; // image row/col
    long k;    // image single index

    // number of threads (retained for interface compatibility)
    int threads;

    time1 = get_time();

    int arg_ok = 1;
    if (argc != 7) {
        printf("ERROR: wrong number of arguments\n");
        arg_ok = 0;
    } else {
        niter = atoi(argv[1]);
        lambda = (fp)atof(argv[2]);
        Nr = atol(argv[3]);
        Nc = atol(argv[4]);
        threads = atoi(argv[5]);
    }

    (void)threads; // threads parameter currently unused

    time2 = get_time();

    if (!arg_ok) {
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

    image_ori = (fp *)malloc(sizeof(fp) * (size_t)image_ori_elem);
    if (!image_ori) {
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

    read_graphics("${REPO_ROOT}/EX1/srad_v1/input_data/image.pgm",
                  image_ori, image_ori_rows, image_ori_cols, 1);

    time3 = get_time();

    Ne = Nr * Nc;
    image = (fp *)malloc(sizeof(fp) * (size_t)Ne);
    if (!image) {
        free(image_ori);

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

    resize(image_ori, image_ori_rows, image_ori_cols, image, Nr, Nc, 1);

    time4 = get_time();

    r1 = 0;      // top row index of ROI
    r2 = (int)Nr - 1; // bottom row index of ROI
    c1 = 0;      // left column index of ROI
    c2 = (int)Nc - 1; // right column index of ROI

    // ROI image size
    NeROI = (long)(r2 - r1 + 1) * (long)(c2 - c1 + 1);

    // allocate variables for surrounding pixels
    iN = (int *)malloc(sizeof(int) * (size_t)Nr); // north surrounding element
    iS = (int *)malloc(sizeof(int) * (size_t)Nr); // south surrounding element
    jW = (int *)malloc(sizeof(int) * (size_t)Nc); // west surrounding element
    jE = (int *)malloc(sizeof(int) * (size_t)Nc); // east surrounding element

    // allocate variables for directional derivatives
    dN = (fp *)malloc(sizeof(fp) * (size_t)Ne); // north direction derivative
    dS = (fp *)malloc(sizeof(fp) * (size_t)Ne); // south direction derivative
    dW = (fp *)malloc(sizeof(fp) * (size_t)Ne); // west direction derivative
    dE = (fp *)malloc(sizeof(fp) * (size_t)Ne); // east direction derivative

    // allocate variable for diffusion coefficient
    c = (fp *)malloc(sizeof(fp) * (size_t)Ne); // diffusion coefficient

    if (!iN || !iS || !jW || !jE || !dN || !dS || !dW || !dE || !c) {
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

    // N/S/W/E indices of surrounding pixels (every element of IMAGE)
#ifdef _OPENMP
#pragma omp parallel for private(i) schedule(static)
#endif
    for (i = 0; i < Nr; i++) {
        iN[i] = (int)i - 1;
        iS[i] = (int)i + 1;
    }
#ifdef _OPENMP
#pragma omp parallel for private(j) schedule(static)
#endif
    for (j = 0; j < Nc; j++) {
        jW[j] = (int)j - 1;
        jE[j] = (int)j + 1;
    }

    // N/S/W/E boundary conditions
    iN[0] = 0;
    iS[Nr - 1] = (int)Nr - 1;
    jW[0] = 0;
    jE[Nc - 1] = (int)Nc - 1;

    time5 = get_time();

#ifdef _OPENMP
#pragma omp parallel for private(i) schedule(static)
#endif
    for (i = 0; i < Ne; i++) {
        image[i] = (fp)exp((double)image[i] / 255.0);
    }

    time6 = get_time();

    for (iter = 0; iter < niter; iter++) {

        sum = 0;
        sum2 = 0;
#ifdef _OPENMP
#pragma omp parallel for private(i, j, k, tmp) reduction(+ : sum, sum2) schedule(static)
#endif
        for (j = c1; j <= c2; j++) {
            long jNr = (long)j * Nr;
            for (i = r1; i <= r2; i++) {
                k = (long)i + jNr;
                tmp = image[k];
                sum += tmp;
                sum2 += tmp * tmp;
            }
        }

        meanROI = sum / (fp)NeROI;
        varROI = (sum2 / (fp)NeROI) - meanROI * meanROI;
        q0sqr = varROI / (meanROI * meanROI);

// directional derivatives, ICOV, diffusion coefficient
#ifdef _OPENMP
#pragma omp parallel for private(i, j, k, Jc, G2, L, num, den, qsqr) schedule(static)
#endif
        for (j = 0; j < Nc; j++) {
            long jNr = j * Nr;
            int jWest = jW[j];
            int jEast = jE[j];
            long jWestNr = (long)jWest * Nr;
            long jEastNr = (long)jEast * Nr;

            for (i = 0; i < Nr; i++) {
                long idx = (long)i + jNr;
                k = idx;
                Jc = image[k];

                long nIdx = (long)iN[i] + jNr;
                long sIdx = (long)iS[i] + jNr;
                long wIdx = (long)i + jWestNr;
                long eIdx = (long)i + jEastNr;

                fp dN_k = image[nIdx] - Jc;
                fp dS_k = image[sIdx] - Jc;
                fp dW_k = image[wIdx] - Jc;
                fp dE_k = image[eIdx] - Jc;

                dN[k] = dN_k;
                dS[k] = dS_k;
                dW[k] = dW_k;
                dE[k] = dE_k;

                G2 = (dN_k * dN_k + dS_k * dS_k + dW_k * dW_k + dE_k * dE_k) /
                     (Jc * Jc);

                L = (dN_k + dS_k + dW_k + dE_k) / Jc;

                num = (fp)(0.5) * G2 - (fp)(1.0 / 16.0) * (L * L);
                den = (fp)1.0 + (fp)0.25 * L;
                qsqr = num / (den * den);

                den = (qsqr - q0sqr) / (q0sqr * ((fp)1.0 + q0sqr));
                fp ck = (fp)1.0 / ((fp)1.0 + den);

                if (ck < (fp)0.0) {
                    ck = (fp)0.0;
                } else if (ck > (fp)1.0) {
                    ck = (fp)1.0;
                }
                c[k] = ck;
            }
        }

// divergence & image update
#ifdef _OPENMP
#pragma omp parallel for private(i, j, k, cN, cS, cW, cE, D) schedule(static)
#endif
        for (j = 0; j < Nc; j++) {
            long jNr = j * Nr;
            int jEast = jE[j];
            long jEastNr = (long)jEast * Nr;

            for (i = 0; i < Nr; i++) {
                long idx = (long)i + jNr;
                k = idx;

                cN = c[k];
                cS = c[(long)iS[i] + jNr];
                cW = c[k];
                cE = c[(long)i + jEastNr];

                D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];

                image[k] = image[k] + (fp)0.25 * lambda * D;
            }
        }
    }

    time7 = get_time();

#ifdef _OPENMP
#pragma omp parallel for private(i) schedule(static)
#endif
    for (i = 0; i < Ne; i++) {
        image[i] = (fp)(log((double)image[i]) * 255.0);
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
           (float)(time1 - time0) / 1000000.0f,
           (float)(time1 - time0) / (float)(time10 - time0) * 100.0f);
    printf("%.12f s, %.12f % : READ COMMAND LINE PARAMETERS\n",
           (float)(time2 - time1) / 1000000.0f,
           (float)(time2 - time1) / (float)(time10 - time0) * 100.0f);
    printf("%.12f s, %.12f % : READ IMAGE FROM FILE\n",
           (float)(time3 - time2) / 1000000.0f,
           (float)(time3 - time2) / (float)(time10 - time0) * 100.0f);
    printf("%.12f s, %.12f % : RESIZE IMAGE\n",
           (float)(time4 - time3) / 1000000.0f,
           (float)(time4 - time3) / (float)(time10 - time0) * 100.0f);
    printf("%.12f s, %.12f % : SETUP, MEMORY ALLOCATION\n",
           (float)(time5 - time4) / 1000000.0f,
           (float)(time5 - time4) / (float)(time10 - time0) * 100.0f);
    printf("%.12f s, %.12f % : EXTRACT IMAGE\n",
           (float)(time6 - time5) / 1000000.0f,
           (float)(time6 - time5) / (float)(time10 - time0) * 100.0f);
    printf("%.12f s, %.12f % : COMPUTE\n",
           (float)(time7 - time6) / 1000000.0f,
           (float)(time7 - time6) / (float)(time10 - time0) * 100.0f);
    printf("%.12f s, %.12f % : COMPRESS IMAGE\n",
           (float)(time8 - time7) / 1000000.0f,
           (float)(time8 - time7) / (float)(time10 - time0) * 100.0f);
    printf("%.12f s, %.12f % : SAVE IMAGE INTO FILE\n",
           (float)(time9 - time8) / 1000000.0f,
           (float)(time9 - time8) / (float)(time10 - time0) * 100.0f);
    printf("%.12f s, %.12f % : FREE MEMORY\n",
           (float)(time10 - time9) / 1000000.0f,
           (float)(time10 - time9) / (float)(time10 - time0) * 100.0f);
    printf("Total time:\n");
    printf("%.12f s\n", (float)(time10 - time0) / 1000000.0f);

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

    return 0;
}
