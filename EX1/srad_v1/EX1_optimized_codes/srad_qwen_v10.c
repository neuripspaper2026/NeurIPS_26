<<<CODE>>>
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
    iN = malloc(sizeof(int) * Nr); // north surrounding element
    iS = malloc(sizeof(int) * Nr); // south surrounding element
    jW = malloc(sizeof(int) * Nc); // west surrounding element
    jE = malloc(sizeof(int) * Nc); // east surrounding element

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

    // Vectorized exponentiation
    for (i = 0; i < Ne; i++) {
        image[i] = exp(image[i] / 255);
    }

    time6 = get_time();

    for (iter = 0; iter < niter; iter++) {
        sum = 0;
        sum2 = 0;
        
        // Optimized ROI statistics calculation with strength reduction
        const fp* image_ptr = image + r1;
        for (i = r1; i <= r2; i++) {
            for (j = c1; j <= c2; j++) {
                tmp = image_ptr[Nr * j];
                sum += tmp;
                sum2 += tmp * tmp;
            }
            image_ptr++;
        }
        meanROI = sum / NeROI;
        varROI = (sum2 / NeROI) - meanROI * meanROI;
        q0sqr = varROI / (meanROI * meanROI);

        // directional derivatives, ICOV, diffusion coefficent
        for (j = 0; j < Nc; j++) {
            const long jNr = Nr * j;
            const long jW_j = jW[j];
            const long jE_j = jE[j];
            
            for (i = 0; i < Nr; i++) {
                // current index/pixel
                k = i + jNr;
                Jc = image[k];

                // directional derivates with precomputed indices
                const long iN_i_Nr_j = iN[i] + jNr;
                const long iS_i_Nr_j = iS[i] + jNr;
                const long i_Nr_jW_j = i + Nr * jW_j;
                const long i_Nr_jE_j = i + Nr * jE_j;
                
                dN[k] = image[iN_i_Nr_j] - Jc;
                dS[k] = image[iS_i_Nr_j] - Jc;
                dW[k] = image[i_Nr_jW_j] - Jc;
                dE[k] = image[i_Nr_jE_j] - Jc;

                // normalized discrete gradient mag squared
                G2 = (dN[k] * dN[k] + dS[k] * dS[k] + dW[k] * dW[k] + dE[k] * dE[k]) / (Jc * Jc);

                // normalized discrete laplacian
                L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

                // ICOV
                num = (0.5 * G2) - ((1.0 / 16.0) * (L * L));
                den = 1 + (.25 * L);
                qsqr = num / (den * den);

                // diffusion coefficent
                den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));
                c[k] = 1.0 / (1.0 + den);

                // saturate diffusion coefficent to 0-1 range
                if (c[k] < 0) {
                    c[k] = 0;
                } else if (c[k] > 1) {
                    c[k] = 1;
                }
            }
        }
