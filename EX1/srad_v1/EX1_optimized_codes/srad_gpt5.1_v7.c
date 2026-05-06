#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

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

    if (argc != 7) {
        printf("ERROR: wrong number of arguments\n");
        return 0;
    } else {
        niter = atoi(argv[1]);
        lambda = (fp)atof(argv[2]);
        Nr = atol(argv[3]);
        Nc = atol(argv[4]);
        threads = atoi(argv[5]);
    }

    (void)threads;

    time2 = get_time();

    image_ori_rows = 502;
    image_ori_cols = 458;
    image_ori_elem = (long)image_ori_rows * (long)image_ori_cols;

    image_ori = (fp *)malloc(sizeof(fp) * (size_t)image_ori_elem);
    if (!image_ori) {
        return 1;
    }

    read_graphics("${REPO_ROOT}/EX1/srad_v1/input_data/image.pgm",
                  image_ori, image_ori_rows, image_ori_cols, 1);

    time3 = get_time();

    Ne = Nr * Nc;

    image = (fp *)malloc(sizeof(fp) * (size_t)Ne);
    if (!image) {
        free(image_ori);
        return 1;
    }

    resize(image_ori, image_ori_rows, image_ori_cols, image, (int)Nr, (int)Nc, 1);

    time4 = get_time();

    r1 = 0;
    r2 = (int)Nr - 1;
    c1 = 0;
    c2 = (int)Nc - 1;

    NeROI = (long)(r2 - r1 + 1) * (long)(c2 - c1 + 1);

    iN = (int *)malloc(sizeof(int) * (size_t)Nr);
    iS = (int *)malloc(sizeof(int) * (size_t)Nr);
    jW = (int *)malloc(sizeof(int) * (size_t)Nc);
    jE = (int *)malloc(sizeof(int) * (size_t)Nc);

    dN = (fp *)malloc(sizeof(fp) * (size_t)Ne);
    dS = (fp *)malloc(sizeof(fp) * (size_t)Ne);
    dW = (fp *)malloc(sizeof(fp) * (size_t)Ne);
    dE = (fp *)malloc(sizeof(fp) * (size_t)Ne);

    c = (fp *)malloc(sizeof(fp) * (size_t)Ne);

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
        return 1;
    }

    for (i = 0; i < Nr; i++) {
        iN[i] = (int)(i - 1);
        iS[i] = (int)(i + 1);
    }
    for (j = 0; j < Nc; j++) {
        jW[j] = (int)(j - 1);
        jE[j] = (int)(j + 1);
    }

    iN[0] = 0;
    iS[Nr - 1] = (int)(Nr - 1);
    jW[0] = 0;
    jE[Nc - 1] = (int)(Nc - 1);

    time5 = get_time();

    const fp inv255 = (fp)(1.0 / 255.0);
    for (k = 0; k < Ne; k++) {
        image[k] = (fp)exp((double)(image[k] * inv255));
    }

    time6 = get_time();

    const fp half = (fp)0.5;
    const fp one_sixteenth = (fp)(1.0 / 16.0);
    const fp quarter = (fp)0.25;
    const fp one = (fp)1.0;
    const fp zero = (fp)0.0;
    const fp lambda_quarter = lambda * quarter;

    for (iter = 0; iter < niter; iter++) {
        sum = zero;
        sum2 = zero;

        for (j = c1; j <= c2; j++) {
            long jNr = (long)Nr * (long)j;
            for (i = r1; i <= r2; i++) {
                tmp = image[i + jNr];
                sum += tmp;
                sum2 += tmp * tmp;
            }
        }

        meanROI = sum / (fp)NeROI;
        varROI = (sum2 / (fp)NeROI) - meanROI * meanROI;
        q0sqr = varROI / (meanROI * meanROI);

        for (j = 0; j < Nc; j++) {
            long jNr = (long)Nr * (long)j;
            int jWj = jW[j];
            int jEj = jE[j];
            long jWNr = (long)Nr * (long)jWj;
            long jENr = (long)Nr * (long)jEj;

            for (i = 0; i < Nr; i++) {
                int iNi = iN[i];
                int iSi = iS[i];

                long k_center = i + jNr;
                long kN = iNi + jNr;
                long kS = iSi + jNr;
                long kW = i + jWNr;
                long kE = i + jENr;

                Jc = image[k_center];

                fp dNk = image[kN] - Jc;
                fp dSk = image[kS] - Jc;
                fp dWk = image[kW] - Jc;
                fp dEk = image[kE] - Jc;

                dN[k_center] = dNk;
                dS[k_center] = dSk;
                dW[k_center] = dWk;
                dE[k_center] = dEk;

                fp Jc2 = Jc * Jc;
                G2 = (dNk * dNk + dSk * dSk + dWk * dWk + dEk * dEk) / Jc2;

                fp sumD = dNk + dSk + dWk + dEk;
                L = sumD / Jc;

                fp L2 = L * L;
                num = half * G2 - one_sixteenth * L2;
                den = one + quarter * L;
                fp den2 = den * den;
                qsqr = num / den2;

                den = (qsqr - q0sqr) / (q0sqr * (one + q0sqr));
                fp ck = one / (one + den);

                if (ck < zero) {
                    ck = zero;
                } else if (ck > one) {
                    ck = one;
                }

                c[k_center] = ck;
            }
        }

        for (j = 0; j < Nc; j++) {
            long jNr = (long)Nr * (long)j;
            int jEj = jE[j];
            long jENr = (long)Nr * (long)jEj;

            for (i = 0; i < Nr; i++) {
                int iSi = iS[i];

                long k_center = i + jNr;
                long kS = iSi + jNr;
                long kE = i + jENr;

                cN = c[k_center];
                cS = c[kS];
                cW = cN;
                cE = c[kE];

                D = cN * dN[k_center] + cS * dS[k_center] +
                    cW * dW[k_center] + cE * dE[k_center];

                image[k_center] = image[k_center] + lambda_quarter * D;
            }
        }
    }

    time7 = get_time();

    for (k = 0; k < Ne; k++) {
        image[k] = (fp)log((double)image[k]) * (fp)255.0;
    }

    time8 = get_time();

    write_graphics(argv[6], image, (int)Nr, (int)Nc, 1, 255);

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
    double kernel_time = (double)(kernel_end.tv_sec - kernel_start.tv_sec) +
                         (double)(kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    double main_time = (double)(main_end.tv_sec - main_start.tv_sec) +
                       (double)(main_end.tv_nsec - main_start.tv_nsec) / 1e9;

    FILE *timing_file = stderr;
    const char *timing_path = getenv("TIMING_LOG_FILE");
    if (timing_path && timing_path[0] != '\0') {
        FILE *tmp = fopen(timing_path, "w");
        if (tmp) {
            timing_file = tmp;
        }
    }

    fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

    if (timing_file != stderr) {
        fclose(timing_file);
    }

    return 0;
}
