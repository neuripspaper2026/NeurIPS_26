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
        lambda = atof(argv[2]);
        Nr = atoi(argv[3]);
        Nc = atoi(argv[4]);
        threads = atoi(argv[5]);
    }

    (void)threads;

    time2 = get_time();

    image_ori_rows = 502;
    image_ori_cols = 458;
    image_ori_elem = image_ori_rows * image_ori_cols;

    image_ori = (fp *)malloc(sizeof(fp) * image_ori_elem);

    read_graphics("${REPO_ROOT}/EX1/srad_v1/input_data/image.pgm", image_ori, image_ori_rows,
                  image_ori_cols, 1);

    time3 = get_time();

    Ne = Nr * Nc;

    image = (fp *)malloc(sizeof(fp) * Ne);

    resize(image_ori, image_ori_rows, image_ori_cols, image, Nr, Nc, 1);

    time4 = get_time();

    r1 = 0;
    r2 = Nr - 1;
    c1 = 0;
    c2 = Nc - 1;

    NeROI = (r2 - r1 + 1) * (c2 - c1 + 1);

    iN = malloc(sizeof(int) * Nr);
    iS = malloc(sizeof(int) * Nr);
    jW = malloc(sizeof(int) * Nc);
    jE = malloc(sizeof(int) * Nc);

    dN = malloc(sizeof(fp) * Ne);
    dS = malloc(sizeof(fp) * Ne);
    dW = malloc(sizeof(fp) * Ne);
    dE = malloc(sizeof(fp) * Ne);

    c = malloc(sizeof(fp) * Ne);

    iN[0] = 0;
    for (i = 1; i < Nr - 1; i++) {
        iN[i] = i - 1;
        iS[i] = i + 1;
    }
    iN[Nr - 1] = Nr - 2;
    iS[0] = 1;
    iS[Nr - 1] = Nr - 1;

    jW[0] = 0;
    for (j = 1; j < Nc - 1; j++) {
        jW[j] = j - 1;
        jE[j] = j + 1;
    }
    jW[Nc - 1] = Nc - 2;
    jE[0] = 1;
    jE[Nc - 1] = Nc - 1;

    time5 = get_time();

    for (i = 0; i < Ne; i++) {
        image[i] = exp(image[i] / 255);
    }

    time6 = get_time();

    const fp inv_NeROI = 1.0 / NeROI;
    const fp quarter_lambda = 0.25 * lambda;

    for (iter = 0; iter < niter; iter++) {

        sum = 0;
        sum2 = 0;
        for (i = r1; i <= r2; i++) {
            for (j = c1; j <= c2; j++) {
                tmp = image[i + Nr * j];
                sum += tmp;
                sum2 += tmp * tmp;
            }
        }
        meanROI = sum * inv_NeROI;
        varROI = (sum2 * inv_NeROI) - meanROI * meanROI;
        q0sqr = varROI / (meanROI * meanROI);

        for (j = 0; j < Nc; j++) {
            long base_j = Nr * j;
            long base_jW = Nr * jW[j];
            long base_jE = Nr * jE[j];

            for (i = 0; i < Nr; i++) {
                k = i + base_j;
                Jc = image[k];

                long iN_idx = iN[i] + base_j;
                long iS_idx = iS[i] + base_j;
                long jW_idx = i + base_jW;
                long jE_idx = i + base_jE;

                fp dN_val = image[iN_idx] - Jc;
                fp dS_val = image[iS_idx] - Jc;
                fp dW_val = image[jW_idx] - Jc;
                fp dE_val = image[jE_idx] - Jc;

                dN[k] = dN_val;
                dS[k] = dS_val;
                dW[k] = dW_val;
                dE[k] = dE_val;

                fp inv_Jc = 1.0 / Jc;
                G2 = (dN_val * dN_val + dS_val * dS_val + dW_val * dW_val + dE_val * dE_val) * inv_Jc * inv_Jc;

                L = (dN_val + dS_val + dW_val + dE_val) * inv_Jc;

                num = (0.5 * G2) - ((1.0 / 16.0) * (L * L));
                den = 1 + (.25 * L);
                qsqr = num / (den * den);

                den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));
                fp c_val = 1.0 / (1.0 + den);

                if (c_val < 0) {
                    c_val = 0;
                } else if (c_val > 1) {
                    c_val = 1;
                }
                c[k] = c_val;
            }
        }

        for (j = 0; j < Nc; j++) {
            long base_j = Nr * j;
            long base_jE = Nr * jE[j];

            for (i = 0; i < Nr; i++) {
                k = i + base_j;

                cN = c[k];
                cS = c[iS[i] + base_j];
                cW = c[k];
                cE = c[i + base_jE];

                D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];

                image[k] = image[k] + quarter_lambda * D;
            }
        }
    }

    time7 = get_time();

    for (i = 0; i < Ne; i++) {
        image[i] = log(image[i]) * 255;
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
