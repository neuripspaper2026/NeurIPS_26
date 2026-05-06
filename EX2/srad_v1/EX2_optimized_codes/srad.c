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
        lambda = atof(argv[2]);
        Nr = atoi(argv[3]); // it is 502 in the original image
        Nc = atoi(argv[4]); // it is 458 in the original image
        threads = atoi(argv[5]);
    }

    (void)threads; // threads parameter currently unused

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

    r1 = 0;      // top row index of ROI
    r2 = Nr - 1; // bottom row index of ROI
    c1 = 0;      // left column index of ROI
    c2 = Nc - 1; // right column index of ROI

    // ROI image size
    NeROI =
        (r2 - r1 + 1) * (c2 - c1 + 1); // number of elements in ROI, ROI size

    // allocate variables for surrounding pixels
    iN = malloc(sizeof(int *) * Nr); // north surrounding element
    iS = malloc(sizeof(int *) * Nr); // south surrounding element
    jW = malloc(sizeof(int *) * Nc); // west surrounding element
    jE = malloc(sizeof(int *) * Nc); // east surrounding element

    // allocate variables for directional derivatives
    dN = malloc(sizeof(fp) * Ne); // north direction derivative
    dS = malloc(sizeof(fp) * Ne); // south direction derivative
    dW = malloc(sizeof(fp) * Ne); // west direction derivative
    dE = malloc(sizeof(fp) * Ne); // east direction derivative

    // allocate variable for diffusion coefficient
    c = malloc(sizeof(fp) * Ne); // diffusion coefficient

    // N/S/W/E indices of surrounding pixels (every element of IMAGE)
    for (i = 0; i < Nr; i++) {
        iN[i] = i - 1; // holds index of IMAGE row above
        iS[i] = i + 1; // holds index of IMAGE row below
    }
    for (j = 0; j < Nc; j++) {
        jW[j] = j - 1; // holds index of IMAGE column on the left
        jE[j] = j + 1; // holds index of IMAGE column on the right
    }
    // N/S/W/E boundary conditions, fix surrounding indices outside boundary of
    // IMAGE
    iN[0] = 0;           // changes IMAGE top row index from -1 to 0
    iS[Nr - 1] = Nr - 1; // changes IMAGE bottom row index from Nr to Nr-1
    jW[0] = 0;           // changes IMAGE leftmost column index from -1 to 0
    jE[Nc - 1] = Nc - 1; // changes IMAGE rightmost column index from Nc to Nc-1

    time5 = get_time();

    for (i = 0; i < Ne; i++) { // do for the number of elements in input IMAGE
        image[i] =
            exp(image[i] /
                255); // exponentiate input IMAGE and copy to output image
    }

    time6 = get_time();

    for (iter = 0; iter < niter;
         iter++) { // do for the number of iterations input parameter

        sum = 0;
        sum2 = 0;
        for (i = r1; i <= r2; i++) {     // do for the range of rows in ROI
            for (j = c1; j <= c2; j++) { // do for the range of columns in ROI
                tmp = image[i + Nr * j]; // get coresponding value in IMAGE
                sum += tmp; // take corresponding value and add to sum
                sum2 +=
                    tmp *
                    tmp; // take square of corresponding value and add to sum2
            }
        }
        meanROI = sum / NeROI; // gets mean (average) value of element in ROI
        varROI = (sum2 / NeROI) - meanROI * meanROI; // gets variance of ROI
        q0sqr = varROI / (meanROI * meanROI); // gets standard deviation of ROI

// directional derivatives, ICOV, diffusion coefficent
        for (j = 0; j < Nc; j++) { // do for the range of columns in IMAGE

            for (i = 0; i < Nr; i++) { // do for the range of rows in IMAGE

                // current index/pixel
                k = i + Nr * j; // get position of current element
                Jc = image[k];  // get value of the current element

                // directional derivates (every element of IMAGE)
                dN[k] =
                    image[iN[i] + Nr * j] - Jc; // north direction derivative
                dS[k] =
                    image[iS[i] + Nr * j] - Jc; // south direction derivative
                dW[k] = image[i + Nr * jW[j]] - Jc; // west direction derivative
                dE[k] = image[i + Nr * jE[j]] - Jc; // east direction derivative

                // normalized discrete gradient mag squared (equ 52,53)
                G2 = (dN[k] * dN[k] +
                      dS[k] * dS[k] // gradient (based on derivatives)
                      + dW[k] * dW[k] + dE[k] * dE[k]) /
                     (Jc * Jc);

                // normalized discrete laplacian (equ 54)
                L = (dN[k] + dS[k] + dW[k] + dE[k]) /
                    Jc; // laplacian (based on derivatives)

                // ICOV (equ 31/35)
                num = (0.5 * G2) -
                      ((1.0 / 16.0) *
                       (L * L));     // num (based on gradient and laplacian)
                den = 1 + (.25 * L); // den (based on laplacian)
                qsqr = num / (den * den); // qsqr (based on num and den)

                // diffusion coefficent (equ 33) (every element of IMAGE)
                den = (qsqr - q0sqr) /
                      (q0sqr * (1 + q0sqr)); // den (based on qsqr and q0sqr)
                c[k] =
                    1.0 / (1.0 + den); // diffusion coefficient (based on den)

                // saturate diffusion coefficent to 0-1 range
                if (c[k] < 0) // if diffusion coefficient < 0
                {
                    c[k] = 0;
                }                  // ... set to 0
                else if (c[k] > 1) // if diffusion coefficient > 1
                {
                    c[k] = 1;
                } // ... set to 1
            }
        }

// divergence & image update
        for (j = 0; j < Nc; j++) { // do for the range of columns in IMAGE


            for (i = 0; i < Nr; i++) { // do for the range of rows in IMAGE

                // current index
                k = i + Nr * j; // get position of current element

                // diffusion coefficent
                cN = c[k];              // north diffusion coefficient
                cS = c[iS[i] + Nr * j]; // south diffusion coefficient
                cW = c[k];              // west diffusion coefficient
                cE = c[i + Nr * jE[j]]; // east diffusion coefficient

                // divergence (equ 58)
                D = cN * dN[k] + cS * dS[k] + cW * dW[k] +
                    cE * dE[k]; // divergence

                // image update (equ 61) (every element of IMAGE)
                image[k] = image[k] + 0.25 * lambda * D; // updates image (based
                                                         // on input time step
                                                         // and divergence)
            }
        }
    }

    // printf("\n");

    time7 = get_time();

    for (i = 0; i < Ne; i++) { // do for the number of elements in IMAGE
        image[i] = log(image[i]) * 255; // take logarithm of image, log compress
    }

    time8 = get_time();

    write_graphics(argv[6], image, Nr, Nc, 1, 255);

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