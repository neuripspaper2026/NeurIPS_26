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

    // inputs image, input paramenters
    fp *image_ori; // originalinput image
    int image_ori_rows;
    int image_ori_cols;
    long image_ori_elem;

    // inputs image, input paramenters
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

    // surrounding pixel indicies
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

    if (argc != 7) {
        printf("ERROR: wrong number of arguments\n");
        return 0;
    } else {
        niter = atoi(argv[1]);
        lambda = (fp)atof(argv[2]);
        Nr = atol(argv[3]); // it is 502 in the original image
        Nc = atol(argv[4]); // it is 458 in the original image
        threads = atoi(argv[5]);
    }

    (void)threads; // threads parameter currently unused

    time2 = get_time();

    image_ori_rows = 502;
    image_ori_cols = 458;
    image_ori_elem = (long)image_ori_rows * (long)image_ori_cols;

    image_ori = (fp *)malloc(sizeof(fp) * (size_t)image_ori_elem);

    read_graphics("${REPO_ROOT}/EX1/srad_v1/input_data/image.pgm", image_ori, image_ori_rows,
                  image_ori_cols, 1);

    time3 = get_time();

    Ne = Nr * Nc;

    image = (fp *)malloc(sizeof(fp) * (size_t)Ne);

    resize(image_ori, image_ori_rows, image_ori_cols, image, (int)Nr, (int)Nc, 1);

    time4 = get_time();

    r1 = 0;              // top row index of ROI
    r2 = (int)Nr - 1;    // bottom row index of ROI
    c1 = 0;              // left column index of ROI
    c2 = (int)Nc - 1;    // right column index of ROI

    // ROI image size
    NeROI = (long)(r2 - r1 + 1) * (long)(c2 - c1 + 1); // number of elements in ROI, ROI size

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

    // N/S/W/E indices of surrounding pixels (every element of IMAGE)
    for (i = 0; i < Nr; i++) {
        iN[i] = (int)i - 1;        // holds index of IMAGE row above
        iS[i] = (int)(i + 1);      // holds index of IMAGE row below
    }
    for (j = 0; j < Nc; j++) {
        jW[j] = (int)j - 1;        // holds index of IMAGE column on the left
        jE[j] = (int)(j + 1);      // holds index of IMAGE column on the right
    }
    // N/S/W/E boundary conditions, fix surrounding indices outside boundary of IMAGE
    iN[0] = 0;                 // changes IMAGE top row index from -1 to 0
    iS[Nr - 1] = (int)Nr - 1;  // changes IMAGE bottom row index from Nr to Nr-1
    jW[0] = 0;                 // changes IMAGE leftmost column index from -1 to 0
    jE[Nc - 1] = (int)Nc - 1;  // changes IMAGE rightmost column index from Nc to Nc-1

    time5 = get_time();

    const fp inv255 = (fp)(1.0 / 255.0);
    for (i = 0; i < Ne; i++) { // do for the number of elements in input IMAGE
        image[i] = (fp)exp((double)(image[i] * inv255)); // exponentiate input IMAGE and copy to output image
    }

    time6 = get_time();

    const fp half = (fp)0.5;
    const fp one_sixteenth = (fp)(1.0 / 16.0);
    const fp quarter = (fp)0.25;
    const fp one = (fp)1.0;

    for (iter = 0; iter < niter; iter++) { // do for the number of iterations input parameter

        sum = (fp)0.0;
        sum2 = (fp)0.0;
        for (j = c1; j <= c2; j++) { // do for the range of columns in ROI
            const long jNr = (long)j * Nr;
            for (i = r1; i <= r2; i++) { // do for the range of rows in ROI
                tmp = image[i + jNr]; // get corresponding value in IMAGE
                sum += tmp;           // take corresponding value and add to sum
                sum2 += tmp * tmp;    // take square of corresponding value and add to sum2
            }
        }
        meanROI = sum / (fp)NeROI;                  // gets mean (average) value of element in ROI
        varROI = (sum2 / (fp)NeROI) - meanROI * meanROI; // gets variance of ROI
        q0sqr = varROI / (meanROI * meanROI);       // gets standard deviation of ROI

        // directional derivatives, ICOV, diffusion coefficent
        for (j = 0; j < Nc; j++) { // do for the range of columns in IMAGE
            const long jNr = j * Nr;

            for (i = 0; i < Nr; i++) { // do for the range of rows in IMAGE

                // current index/pixel
                k = i + jNr;  // get position of current element
                Jc = image[k];  // get value of the current element

                const long north_idx = (long)iN[i] + jNr;
                const long south_idx = (long)iS[i] + jNr;
                const long west_idx  = i + (long)Nr * (long)jW[j];
                const long east_idx  = i + (long)Nr * (long)jE[j];

                // directional derivates (every element of IMAGE)
                const fp dN_val = image[north_idx] - Jc; // north direction derivative
                const fp dS_val = image[south_idx] - Jc; // south direction derivative
                const fp dW_val = image[west_idx]  - Jc; // west direction derivative
                const fp dE_val = image[east_idx]  - Jc; // east direction derivative

                dN[k] = dN_val;
                dS[k] = dS_val;
                dW[k] = dW_val;
                dE[k] = dE_val;

                // normalized discrete gradient mag squared (equ 52,53)
                const fp Jc2 = Jc * Jc;
                const fp grad_sum = dN_val * dN_val + dS_val * dS_val + dW_val * dW_val + dE_val * dE_val;
                G2 = grad_sum / Jc2;

                // normalized discrete laplacian (equ 54)
                L = (dN_val + dS_val + dW_val + dE_val) / Jc; // laplacian (based on derivatives)

                // ICOV (equ 31/35)
                const fp L2 = L * L;
                num = (half * G2) - (one_sixteenth * L2); // num (based on gradient and laplacian)
                den = one + (quarter * L);               // den (based on laplacian)
                const fp den2 = den * den;
                qsqr = num / den2;                        // qsqr (based on num and den)

                // diffusion coefficent (equ 33) (every element of IMAGE)
                const fp denom_q = q0sqr * (one + q0sqr);
                den = (qsqr - q0sqr) / denom_q; // den (based on qsqr and q0sqr)
                fp cval = one / (one + den);    // diffusion coefficient (based on den)

                // saturate diffusion coefficent to 0-1 range
                if (cval < (fp)0.0) {
                    cval = (fp)0.0;
                } else if (cval > one) {
                    cval = one;
                }
                c[k] = cval;
            }
        }

        // divergence & image update
        for (j = 0; j < Nc; j++) { // do for the range of columns in IMAGE
            const long jNr = j * Nr;

            for (i = 0; i < Nr; i++) { // do for the range of rows in IMAGE

                // current index
                k = i + jNr; // get position of current element

                // diffusion coefficent
                cN = c[k];                             // north diffusion coefficient
                cS = c[(long)iS[i] + jNr];             // south diffusion coefficient
                cW = c[k];                             // west diffusion coefficient
                cE = c[i + (long)Nr * (long)jE[j]];    // east diffusion coefficient

                // divergence (equ 58)
                D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k]; // divergence

                // image update (equ 61) (every element of IMAGE)
                image[k] = image[k] + quarter * lambda * D; // updates image (based on input time step and divergence)
            }
        }
    }

    time7 = get_time();

    for (i = 0; i < Ne; i++) { // do for the number of elements in IMAGE
        image[i] = (fp)log((double)image[i]) * (fp)255.0; // take logarithm of image, log compress
    }

    time8 = get_time();

    write_graphics(argv[6], image, (int)Nr, (int)Nc, 1, 255);

    time9 = get_time();

    free(image_ori);
    free(image);

    free(iN);
    free(iS);
    free(jW);
    free(jE); // deallocate surrounding pixel memory
    free(dN);
    free(dS);
    free(dW);
    free(dE); // deallocate directional derivative memory
    free(c);  // deallocate diffusion coefficient memory

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
}
