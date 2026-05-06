#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// Returns the current system time in microseconds
long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

using namespace std;

#define BLOCK_SIZE 16
#define BLOCK_SIZE_C BLOCK_SIZE
#define BLOCK_SIZE_R BLOCK_SIZE

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip) */
#define MAX_PD (3.0e6)
/* required precision in degrees    */
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor   */
#define FACTOR_CHIP 0.5


typedef float FLOAT;

/* chip parameters  */
const FLOAT t_chip = 0.0005;
const FLOAT chip_height = 0.016;
const FLOAT chip_width = 0.016;

/* ambient temperature, assuming no package at all  */
const FLOAT amb_temp = 80.0;

int num_omp_threads;

/* Single iteration of the transient solver in the grid model.
 * advances the solution of the discretized difference equations
 * by one time step
 */
void single_iteration(FLOAT *restrict result, FLOAT *restrict temp, FLOAT *restrict power, int row,
                      int col, FLOAT Cap_1, FLOAT Rx_1, FLOAT Ry_1, FLOAT Rz_1,
                      FLOAT step) {
    FLOAT delta;
    int r, c;
    int chunk;
    const int blk_r = BLOCK_SIZE_R;
    const int blk_c = BLOCK_SIZE_C;
    const int num_chunk = (row / blk_r) * (col / blk_c);
    const int chunks_in_row = col / blk_c;
    const int chunks_in_col = row / blk_r;

    const FLOAT amb = amb_temp; /* cache frequently used globals locally */

    /* Parallelize outermost loop over chunks; each chunk works on disjoint region */
    #ifdef _OPENMP
    #pragma omp parallel for private(chunk, r, c, delta) schedule(static)
    #endif
    for (chunk = 0; chunk < num_chunk; ++chunk) {
        const int r_start = blk_r * (chunk / chunks_in_col);
        const int c_start = blk_c * (chunk % chunks_in_row);
        const int r_end = r_start + blk_r > row ? row : r_start + blk_r;
        const int c_end = c_start + blk_c > col ? col : c_start + blk_c;

        /* Boundary blocks: handle all special cases */
        if (r_start == 0 || c_start == 0 || r_end == row || c_end == col) {
            const int r_block_end = r_start + blk_r;
            const int c_block_end = c_start + blk_c;

            for (r = r_start; r < r_block_end; ++r) {
                const int r_off = r * col;
                const int r_off_up   = (r > 0)        ? (r - 1) * col : 0;
                const int r_off_down = (r < row - 1)  ? (r + 1) * col : 0;

                for (c = c_start; c < c_block_end; ++c) {
                    const int idx = r_off + c;

                    /* Corner 1: top-left */
                    if (r == 0 && c == 0) {
                        delta = Cap_1 *
                                (power[0] +
                                 (temp[1] - temp[0]) * Rx_1 +
                                 (temp[col] - temp[0]) * Ry_1 +
                                 (amb - temp[0]) * Rz_1);
                    }
                    /* Corner 2: top-right */
                    else if (r == 0 && c == col - 1) {
                        delta = Cap_1 *
                                (power[c] +
                                 (temp[c - 1] - temp[c]) * Rx_1 +
                                 (temp[c + col] - temp[c]) * Ry_1 +
                                 (amb - temp[c]) * Rz_1);
                    }
                    /* Corner 3: bottom-right */
                    else if (r == row - 1 && c == col - 1) {
                        delta = Cap_1 *
                                (power[idx] +
                                 (temp[idx - 1] - temp[idx]) * Rx_1 +
                                 (temp[(r - 1) * col + c] - temp[idx]) * Ry_1 +
                                 (amb - temp[idx]) * Rz_1);
                    }
                    /* Corner 4: bottom-left */
                    else if (r == row - 1 && c == 0) {
                        delta = Cap_1 *
                                (power[r_off] +
                                 (temp[r_off + 1] - temp[r_off]) * Rx_1 +
                                 (temp[(r - 1) * col] - temp[r_off]) * Ry_1 +
                                 (amb - temp[r_off]) * Rz_1);
                    }
                    /* Edge 1: top row (excluding corners) */
                    else if (r == 0) {
                        delta = Cap_1 *
                                (power[c] +
                                 (temp[c + 1] + temp[c - 1] - (FLOAT)2.0 * temp[c]) * Rx_1 +
                                 (temp[col + c] - temp[c]) * Ry_1 +
                                 (amb - temp[c]) * Rz_1);
                    }
                    /* Edge 2: right column (excluding corners) */
                    else if (c == col - 1) {
                        delta = Cap_1 *
                                (power[idx] +
                                 (temp[r_off_down + c] + temp[r_off_up + c] -
                                  (FLOAT)2.0 * temp[idx]) * Ry_1 +
                                 (temp[idx - 1] - temp[idx]) * Rx_1 +
                                 (amb - temp[idx]) * Rz_1);
                    }
                    /* Edge 3: bottom row (excluding corners) */
                    else if (r == row - 1) {
                        delta = Cap_1 *
                                (power[idx] +
                                 (temp[idx + 1] + temp[idx - 1] -
                                  (FLOAT)2.0 * temp[idx]) * Rx_1 +
                                 (temp[r_off_up + c] - temp[idx]) * Ry_1 +
                                 (amb - temp[idx]) * Rz_1);
                    }
                    /* Edge 4: left column (excluding corners) */
                    else if (c == 0) {
                        delta = Cap_1 *
                                (power[r_off] +
                                 (temp[r_off_down] + temp[r_off_up] -
                                  (FLOAT)2.0 * temp[r_off]) * Ry_1 +
                                 (temp[r_off + 1] - temp[r_off]) * Rx_1 +
                                 (amb - temp[r_off]) * Rz_1);
                    }
                    else {
                        /* Interior cell inside boundary block - use interior stencil */
                        const FLOAT t_center = temp[idx];
                        delta = Cap_1 *
                                (power[idx] +
                                 (temp[r_off_down + c] + temp[r_off_up + c] -
                                  (FLOAT)2.0 * t_center) * Ry_1 +
                                 (temp[idx + 1] + temp[idx - 1] -
                                  (FLOAT)2.0 * t_center) * Rx_1 +
                                 (amb - t_center) * Rz_1);
                    }

                    result[idx] = temp[idx] + delta;
                }
            }
            continue;
        }

        /* Interior blocks: all points are interior, use fast stencil */
        for (r = r_start; r < r_end; ++r) {
            const int r_off = r * col;
            const int r_off_up   = (r - 1) * col;
            const int r_off_down = (r + 1) * col;

            for (c = c_start; c < c_end; ++c) {
                const int idx = r_off + c;
                const FLOAT t_center = temp[idx];

                result[idx] =
                    t_center +
                    Cap_1 *
                    (power[idx] +
                     (temp[r_off_down + c] + temp[r_off_up + c] -
                      (FLOAT)2.0 * t_center) * Ry_1 +
                     (temp[idx + 1] + temp[idx - 1] -
                      (FLOAT)2.0 * t_center) * Rx_1 +
                     (amb - t_center) * Rz_1);
            }
        }
    }
}

/* Transient solver driver routine: simply converts the heat
 * transfer differential equations to difference equations
 * and solves the difference equations by iterating
 */
void compute_tran_temp(FLOAT *result, int num_iterations, FLOAT *temp,
                       FLOAT *power, int row, int col) {
#ifdef VERBOSE
    int i = 0;
#endif

    FLOAT grid_height = chip_height / row;
    FLOAT grid_width = chip_width / col;

    FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    FLOAT Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    FLOAT Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    FLOAT Rz = t_chip / (K_SI * grid_height * grid_width);

    FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    FLOAT step = PRECISION / max_slope / 1000.0;

    FLOAT Rx_1 = 1.f / Rx;
    FLOAT Ry_1 = 1.f / Ry;
    FLOAT Rz_1 = 1.f / Rz;
    FLOAT Cap_1 = step / Cap;
#ifdef VERBOSE
    fprintf(stdout, "total iterations: %d s\tstep size: %g s\n", num_iterations,
            step);
    fprintf(stdout, "Rx: %g\tRy: %g\tRz: %g\tCap: %g\n", Rx, Ry, Rz, Cap);
#endif

    {
        FLOAT *r = result;
        FLOAT *t = temp;
        for (int i = 0; i < num_iterations; i++) {
#ifdef VERBOSE
            fprintf(stdout, "iteration %d\n", i++);
#endif
            single_iteration(r, t, power, row, col, Cap_1, Rx_1, Ry_1, Rz_1,
                             step);
            FLOAT *tmp = t;
            t = r;
            r = tmp;
        }
    }
#ifdef VERBOSE
    fprintf(stdout, "iteration %d\n", i++);
#endif
}

void fatal(char *s) {
    fprintf(stderr, "error: %s\n", s);
    exit(1);
}

void writeoutput(FLOAT *vect, int grid_rows, int grid_cols, char *file) {

    int i, j, index = 0;
    FILE *fp;
    char str[STR_SIZE];

    if ((fp = fopen(file, "w")) == 0)
        printf("The file was not opened\n");


    for (i = 0; i < grid_rows; i++)
        for (j = 0; j < grid_cols; j++) {

            sprintf(str, "%d\t%g\n", index, vect[i * grid_cols + j]);
            fputs(str, fp);
            index++;
        }

    fclose(fp);
}

void read_input(FLOAT *vect, int grid_rows, int grid_cols, char *file) {
    int i, index;
    FILE *fp;
    char str[STR_SIZE];
    FLOAT val;

    fp = fopen(file, "r");
    if (!fp)
        fatal("file could not be opened for reading");

    for (i = 0; i < grid_rows * grid_cols; i++) {
        fgets(str, STR_SIZE, fp);
        if (feof(fp))
            fatal("not enough lines in file");
        if ((sscanf(str, "%f", &val) != 1))
            fatal("invalid file format");
        vect[i] = val;
    }

    fclose(fp);
}

void usage(int argc, char **argv) {
    fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <no. of "
                    "threads><temp_file> <power_file>\n",
            argv[0]);
    fprintf(stderr,
            "\t<grid_rows>  - number of rows in the grid (positive integer)\n");
    fprintf(
        stderr,
        "\t<grid_cols>  - number of columns in the grid (positive integer)\n");
    fprintf(stderr, "\t<sim_time>   - number of iterations\n");
    fprintf(stderr, "\t<no. of threads>   - number of threads\n");
    fprintf(stderr, "\t<temp_file>  - name of the file containing the initial "
                    "temperature values of each cell\n");
    fprintf(stderr, "\t<power_file> - name of the file containing the "
                    "dissipated power values of each cell\n");
    fprintf(stderr, "\t<output_file> - name of the output file\n");
    exit(1);
}

int main(int argc, char **argv) {
    int grid_rows, grid_cols, sim_time, i;
    FLOAT *temp, *power, *result;
    char *tfile, *pfile, *ofile;

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

    /* check validity of inputs */
    if (argc != 8)
        usage(argc, argv);
    if ((grid_rows = atoi(argv[1])) <= 0 || (grid_cols = atoi(argv[2])) <= 0 ||
        (sim_time = atoi(argv[3])) <= 0 ||
        (num_omp_threads = atoi(argv[4])) <= 0)
        usage(argc, argv);

    /* allocate memory for the temperature and power arrays */
    temp = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
    power = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
    result = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
    if (!temp || !power)
        fatal("unable to allocate memory");

    /* read initial temperatures and input power    */
    tfile = argv[5];
    pfile = argv[6];
    ofile = argv[7];

    read_input(temp, grid_rows, grid_cols, tfile);
    read_input(power, grid_rows, grid_cols, pfile);

    printf("Start computing the transient temperature\n");
    long long start_time = get_time();
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);
    compute_tran_temp(result, sim_time, temp, power, grid_rows, grid_cols);
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);

    long long end_time = get_time();

    printf("Ending simulation\n");

    /* output results to the file specified by command line */
    /* The latest temperatures are in temp for even sim_time, otherwise in result */
    writeoutput((sim_time % 2 == 0) ? temp : result, grid_rows, grid_cols, ofile);

    /* cleanup  */
    free(temp);
    free(power);

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