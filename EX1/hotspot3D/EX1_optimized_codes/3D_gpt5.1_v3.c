#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#define STR_SIZE (256)
#define MAX_PD (3.0e6)
/* required precision in degrees	*/
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP 0.5


/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

void fatal(char *s) { fprintf(stderr, "Error: %s\n", s); }

void readinput(float *vect, int grid_rows, int grid_cols, int layers,
               char *file) {
    int i, j, k;
    FILE *fp;
    char str[STR_SIZE];
    float val;

    if ((fp = fopen(file, "r")) == 0)
        fatal("The file was not opened");


    for (i = 0; i <= grid_rows - 1; i++)
        for (j = 0; j <= grid_cols - 1; j++)
            for (k = 0; k <= layers - 1; k++) {
                if (fgets(str, STR_SIZE, fp) == NULL)
                    fatal("Error reading file\n");
                if (feof(fp))
                    fatal("not enough lines in file");
                if ((sscanf(str, "%f", &val) != 1))
                    fatal("invalid file format");
                vect[i * grid_cols + j + k * grid_rows * grid_cols] = val;
            }

    fclose(fp);
}


void writeoutput(float *vect, int grid_rows, int grid_cols, int layers,
                 char *file) {

    int i, j, k, index = 0;
    FILE *fp;
    char str[STR_SIZE];

    if ((fp = fopen(file, "w")) == 0)
        printf("The file was not opened\n");

    for (i = 0; i < grid_rows; i++)
        for (j = 0; j < grid_cols; j++)
            for (k = 0; k < layers; k++) {
                sprintf(str, "%d\t%g\n", index,
                        vect[i * grid_cols + j + k * grid_rows * grid_cols]);
                fputs(str, fp);
                index++;
            }

    fclose(fp);
}


void computeTempCPU(float *pIn, float *tIn, float *tOut, int nx, int ny, int nz,
                    float Cap, float Rx, float Ry, float Rz, float dt,
                    int numiter) {
    float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw = stepDivCap / Rx;
    cn = cs = stepDivCap / Ry;
    ct = cb = stepDivCap / Rz;

    cc = 1.0 - (2.0 * ce + 2.0 * cn + 3.0 * ct);

    int c, w, e, n, s, b, t;
    int x, y, z;
    int i = 0;
    do {
        for (z = 0; z < nz; z++)
            for (y = 0; y < ny; y++)
                for (x = 0; x < nx; x++) {
                    c = x + y * nx + z * nx * ny;

                    w = (x == 0) ? c : c - 1;
                    e = (x == nx - 1) ? c : c + 1;
                    n = (y == 0) ? c : c - nx;
                    s = (y == ny - 1) ? c : c + nx;
                    b = (z == 0) ? c : c - nx * ny;
                    t = (z == nz - 1) ? c : c + nx * ny;


                    tOut[c] = tIn[c] * cc + tIn[n] * cn + tIn[s] * cs +
                              tIn[e] * ce + tIn[w] * cw + tIn[t] * ct +
                              tIn[b] * cb + (dt / Cap) * pIn[c] + ct * amb_temp;
                }
        float *temp = tIn;
        tIn = tOut;
        tOut = temp;
        i++;
    } while (i < numiter);
}

float accuracy(float *arr1, float *arr2, int len) {
    float err = 0.0;
    int i;
    for (i = 0; i < len; i++) {
        err += (arr1[i] - arr2[i]) * (arr1[i] - arr2[i]);
    }

    return (float)sqrt(err / len);
}

void computeTempOMP(float *restrict pIn, float *restrict tIn, float *restrict tOut,
                    const int nx, const int ny, const int nz,
                    const float Cap, const float Rx, const float Ry, const float Rz,
                    const float dt, const int numiter) {

    float ce, cw, cn, cs, ct, cb, cc;

    const float stepDivCap = dt / Cap;
    ce = cw = stepDivCap / Rx;
    cn = cs = stepDivCap / Ry;
    ct = cb = stepDivCap / Rz;

    cc = 1.0f - (2.0f * ce + 2.0f * cn + 3.0f * ct);

    int count = 0;
    float *tIn_t  = tIn;
    float *tOut_t = tOut;

    const int nxny   = nx * ny;
    const int nxyz   = nxny * nz;
    const int last_x = nx - 1;
    const int last_y = ny - 1;
    const int last_z = nz - 1;
    const float ct_amb = ct * amb_temp;

    printf("1 thread running\n");

    do {
        int z, y, x;

        /* interior points (no boundary checks) */
        for (z = 1; z < last_z; ++z) {
            const int zoff   = z * nxny;
            const int zoff_m = zoff - nxny;
            const int zoff_p = zoff + nxny;
            for (y = 1; y < last_y; ++y) {
                const int base   = zoff + y * nx;
                const int base_n = base - nx;
                const int base_s = base + nx;
                int c = base + 1;  /* start from x = 1 */
                int w = c - 1;
                int e = c + 1;
                int n = base_n + 1;
                int s = base_s + 1;
                int b = zoff_m + y * nx + 1;
                int t = zoff_p + y * nx + 1;

                const int end_c = base + last_x; /* x = last_x - 1 will be last iter */

                for (; c < end_c; ++c, ++w, ++e, ++n, ++s, ++b, ++t) {
                    tOut_t[c] =
                        cc * tIn_t[c] +
                        cw * tIn_t[w] + ce * tIn_t[e] +
                        cs * tIn_t[s] + cn * tIn_t[n] +
                        cb * tIn_t[b] + ct * tIn_t[t] +
                        stepDivCap * pIn[c] + ct_amb;
                }
            }
        }

        /* boundary planes and edges */

        /* z = 0 and z = last_z */
        for (z = 0; z < nz; z += (nz - 1)) {
            const int zoff = z * nxny;
            const int zoff_b = (z == 0) ? zoff : zoff - nxny;
            const int zoff_t = (z == last_z) ? zoff : zoff + nxny;
            for (y = 0; y < ny; ++y) {
                const int base   = zoff + y * nx;
                const int base_n = (y == 0)       ? base : base - nx;
                const int base_s = (y == last_y) ? base : base + nx;
                for (x = 0; x < nx; ++x) {
                    const int c = base + x;
                    const int w = (x == 0)       ? c : c - 1;
                    const int e = (x == last_x) ? c : c + 1;
                    const int n = base_n + x;
                    const int s = base_s + x;
                    const int b = (z == 0)       ? c : zoff_b + y * nx + x;
                    const int t = (z == last_z) ? c : zoff_t + y * nx + x;

                    tOut_t[c] =
                        cc * tIn_t[c] +
                        cw * tIn_t[w] + ce * tIn_t[e] +
                        cs * tIn_t[s] + cn * tIn_t[n] +
                        cb * tIn_t[b] + ct * tIn_t[t] +
                        stepDivCap * pIn[c] + ct_amb;
                }
            }
        }

        /* y = 0 and y = last_y (excluding already processed z planes) */
        for (z = 1; z < last_z; ++z) {
            const int zoff   = z * nxny;
            const int zoff_b = zoff - nxny;
            const int zoff_t = zoff + nxny;

            for (y = 0; y < ny; y += (ny - 1)) {
                const int base   = zoff + y * nx;
                const int base_n = (y == 0)       ? base : base - nx;
                const int base_s = (y == last_y) ? base : base + nx;
                for (x = 0; x < nx; ++x) {
                    const int c = base + x;
                    const int w = (x == 0)       ? c : c - 1;
                    const int e = (x == last_x) ? c : c + 1;
                    const int n = base_n + x;
                    const int s = base_s + x;
                    const int b = zoff_b + y * nx + x;
                    const int t = zoff_t + y * nx + x;

                    tOut_t[c] =
                        cc * tIn_t[c] +
                        cw * tIn_t[w] + ce * tIn_t[e] +
                        cs * tIn_t[s] + cn * tIn_t[n] +
                        cb * tIn_t[b] + ct * tIn_t[t] +
                        stepDivCap * pIn[c] + ct_amb;
                }
            }
        }

        /* x = 0 and x = last_x (excluding already processed faces) */
        for (z = 1; z < last_z; ++z) {
            const int zoff   = z * nxny;
            const int zoff_b = zoff - nxny;
            const int zoff_t = zoff + nxny;

            for (y = 1; y < last_y; ++y) {
                const int base   = zoff + y * nx;
                const int base_n = base - nx;
                const int base_s = base + nx;

                for (x = 0; x < nx; x += (nx - 1)) {
                    const int c = base + x;
                    const int w = (x == 0)       ? c : c - 1;
                    const int e = (x == last_x) ? c : c + 1;
                    const int n = base_n + x;
                    const int s = base_s + x;
                    const int b = zoff_b + y * nx + x;
                    const int t = zoff_t + y * nx + x;

                    tOut_t[c] =
                        cc * tIn_t[c] +
                        cw * tIn_t[w] + ce * tIn_t[e] +
                        cs * tIn_t[s] + cn * tIn_t[n] +
                        cb * tIn_t[b] + ct * tIn_t[t] +
                        stepDivCap * pIn[c] + ct_amb;
                }
            }
        }

        float *tmp = tIn_t;
        tIn_t  = tOut_t;
        tOut_t = tmp;
        ++count;
    } while (count < numiter);

    (void)nxyz; /* silence unused warning if nxyz not used elsewhere */
    return;
}

void usage(int argc, char **argv) {
    fprintf(stderr, "Usage: %s <rows/cols> <layers> <iterations> <powerFile> "
                    "<tempFile> <outputFile>\n",
            argv[0]);
    fprintf(stderr, "\t<rows/cols>  - number of rows/cols in the grid "
                    "(positive integer)\n");
    fprintf(stderr,
            "\t<layers>  - number of layers in the grid (positive integer)\n");

    fprintf(stderr, "\t<iteration> - number of iterations\n");
    fprintf(stderr, "\t<powerFile>  - name of the file containing the initial "
                    "power values of each cell\n");
    fprintf(stderr, "\t<tempFile>  - name of the file containing the initial "
                    "temperature values of each cell\n");
    fprintf(stderr, "\t<outputFile - output file\n");
    exit(1);
}


int main(int argc, char **argv) {
    if (argc != 7) {
        usage(argc, argv);
    }

    struct timespec main_start, main_end;
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &main_start);

    FILE *timing_file = stderr;
    const char *timing_path = getenv("TIMING_LOG_FILE");
    if (timing_path && timing_path[0] != '\0') {
        FILE *tmp = fopen(timing_path, "w");
        if (tmp)
            timing_file = tmp;
    }

    char *pfile, *tfile, *ofile; // *testFile;
    int iterations = atoi(argv[3]);

    pfile = argv[4];
    tfile = argv[5];
    ofile = argv[6];
    // testFile = argv[7];
    int numCols = atoi(argv[1]);
    int numRows = atoi(argv[1]);
    int layers = atoi(argv[2]);

    /* calculating parameters*/

    float dx = chip_height / numRows;
    float dy = chip_width / numCols;
    float dz = t_chip / layers;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
    float Rx = dy / (2.0 * K_SI * t_chip * dx);
    float Ry = dx / (2.0 * K_SI * t_chip * dy);
    float Rz = dz / (K_SI * dx * dy);

    // cout << Rx << " " << Ry << " " << Rz << endl;
    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float dt = PRECISION / max_slope;


    float *powerIn, *tempOut, *tempIn, *tempCopy; // *pCopy;
    //    float *d_powerIn, *d_tempIn, *d_tempOut;
    int size = numCols * numRows * layers;

    powerIn = (float *)calloc(size, sizeof(float));
    tempCopy = (float *)malloc(size * sizeof(float));
    tempIn = (float *)calloc(size, sizeof(float));
    tempOut = (float *)calloc(size, sizeof(float));
    // pCopy = (float*)calloc(size,sizeof(float));
    float *answer = (float *)calloc(size, sizeof(float));

    // outCopy = (float*)calloc(size, sizeof(float));
    readinput(powerIn, numRows, numCols, layers, pfile);
    readinput(tempIn, numRows, numCols, layers, tfile);

    memcpy(tempCopy, tempIn, size * sizeof(float));

    struct timeval start, stop;
    float time;
    gettimeofday(&start, NULL);
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);
    computeTempOMP(powerIn, tempIn, tempOut, numCols, numRows, layers, Cap, Rx,
                   Ry, Rz, dt, iterations);
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gettimeofday(&stop, NULL);
    time = (stop.tv_usec - start.tv_usec) * 1.0e-6 + stop.tv_sec - start.tv_sec;
    computeTempCPU(powerIn, tempCopy, answer, numCols, numRows, layers, Cap, Rx,
                   Ry, Rz, dt, iterations);

    float acc = accuracy(tempOut, answer, numRows * numCols * layers);
    printf("Time: %.3f (s)\n", time);
    printf("Accuracy: %e\n", acc);
    writeoutput(tempOut, numRows, numCols, layers, ofile);
    free(tempIn);
    free(tempOut);
    free(powerIn);
    clock_gettime(CLOCK_MONOTONIC, &main_end);

    double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                         (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    double main_time = (main_end.tv_sec - main_start.tv_sec) +
                       (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

    fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

    if (timing_file != stderr)
        fclose(timing_file);

    return 0;
}