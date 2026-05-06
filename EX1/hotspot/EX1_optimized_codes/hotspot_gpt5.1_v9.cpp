#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

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
void single_iteration(FLOAT * __restrict result, FLOAT * __restrict temp, FLOAT * __restrict power, int row,
                      int col, const FLOAT Cap_1, const FLOAT Rx_1, const FLOAT Ry_1, const FLOAT Rz_1,
                      const FLOAT step) {
    (void)step;
    const int blkR = BLOCK_SIZE_R;
    const int blkC = BLOCK_SIZE_C;
    const int chunks_in_row = col / blkC;
    const int chunks_in_col = row / blkR;
    const int num_chunk = chunks_in_row * chunks_in_col;

    for (int chunk = 0; chunk < num_chunk; ++chunk) {
        const int r_block = chunk / chunks_in_col;
        const int c_block = chunk % chunks_in_row;

        const int r_start = blkR * r_block;
        const int c_start = blkC * c_block;
        const int r_end = r_start + blkR;
        const int c_end = c_start + blkC;

        const int is_top_block    = (r_start == 0);
        const int is_bottom_block = (r_end == row);
        const int is_left_block   = (c_start == 0);
        const int is_right_block  = (c_end == col);

        /* Fast interior-block path: no boundary handling needed */
        if (!is_top_block && !is_bottom_block && !is_left_block && !is_right_block) {
            for (int r = r_start; r < r_end; ++r) {
                const int base = r * col;
                for (int c = c_start; c < c_end; ++c) {
                    const int idx = base + c;
                    const FLOAT t_center = temp[idx];
                    const FLOAT t_up     = temp[idx - col];
                    const FLOAT t_down   = temp[idx + col];
                    const FLOAT t_left   = temp[idx - 1];
                    const FLOAT t_right  = temp[idx + 1];
                    const FLOAT p        = power[idx];

                    const FLOAT dRy = (t_down + t_up   - (FLOAT)2.0 * t_center) * Ry_1;
                    const FLOAT dRx = (t_right + t_left - (FLOAT)2.0 * t_center) * Rx_1;
                    const FLOAT dRz = (amb_temp - t_center) * Rz_1;

                    result[idx] = t_center + Cap_1 * (p + dRy + dRx + dRz);
                }
            }
            continue;
        }

        /* Block includes at least one boundary: handle with original logic, but hoist common terms */
        for (int r = r_start; r < r_end; ++r) {
            const int base = r * col;
            const int is_top    = (r == 0);
            const int is_bottom = (r == row - 1);

            for (int c = c_start; c < c_end; ++c) {
                const int idx = base + c;
                const int is_left  = (c == 0);
                const int is_right = (c == col - 1);

                FLOAT t_center = temp[idx];
                FLOAT p_center;
                FLOAT delta;

                /* Corners */
                if (is_top && is_left) {
                    /* (r == 0, c == 0) */
                    p_center = power[0];
                    const FLOAT t_right = temp[1];
                    const FLOAT t_down  = temp[col];
                    const FLOAT dRx = (t_right - t_center) * Rx_1;
                    const FLOAT dRy = (t_down  - t_center) * Ry_1;
                    const FLOAT dRz = (amb_temp - t_center) * Rz_1;
                    delta = Cap_1 * (p_center + dRx + dRy + dRz);
                } else if (is_top && is_right) {
                    /* (r == 0, c == col-1) */
                    p_center = power[c];
                    const FLOAT t_left  = temp[c - 1];
                    const FLOAT t_down  = temp[c + col];
                    const FLOAT dRx = (t_left - t_center) * Rx_1;
                    const FLOAT dRy = (t_down  - t_center) * Ry_1;
                    const FLOAT dRz = (amb_temp - t_center) * Rz_1;
                    delta = Cap_1 * (p_center + dRx + dRy + dRz);
                } else if (is_bottom && is_right) {
                    /* (r == row-1, c == col-1) */
                    p_center = power[idx];
                    const FLOAT t_left = temp[idx - 1];
                    const FLOAT t_up   = temp[idx - col];
                    const FLOAT dRx = (t_left - t_center) * Rx_1;
                    const FLOAT dRy = (t_up   - t_center) * Ry_1;
                    const FLOAT dRz = (amb_temp - t_center) * Rz_1;
                    delta = Cap_1 * (p_center + dRx + dRy + dRz);
                } else if (is_bottom && is_left) {
                    /* (r == row-1, c == 0) */
                    p_center = power[base];
                    const FLOAT t_right = temp[base + 1];
                    const FLOAT t_up    = temp[base - col];
                    const FLOAT dRx = (t_right - t_center) * Rx_1;
                    const FLOAT dRy = (t_up    - t_center) * Ry_1;
                    const FLOAT dRz = (amb_temp - t_center) * Rz_1;
                    delta = Cap_1 * (p_center + dRx + dRy + dRz);
                }
                /* Edges (but not corners) */
                else if (is_top) {
                    /* Edge top row (excluding corners) */
                    p_center = power[c];
                    const FLOAT t_left  = temp[c - 1];
                    const FLOAT t_right = temp[c + 1];
                    const FLOAT t_down  = temp[col + c];
                    const FLOAT dRx = (t_right + t_left - (FLOAT)2.0 * t_center) * Rx_1;
                    const FLOAT dRy = (t_down  - t_center) * Ry_1;
                    const FLOAT dRz = (amb_temp - t_center) * Rz_1;
                    delta = Cap_1 * (p_center + dRx + dRy + dRz);
                } else if (is_right) {
                    /* Right column (excluding corners) */
                    p_center = power[idx];
                    const FLOAT t_up   = temp[idx - col];
                    const FLOAT t_down = temp[idx + col];
                    const FLOAT t_left = temp[idx - 1];
                    const FLOAT dRy = (t_down + t_up - (FLOAT)2.0 * t_center) * Ry_1;
                    const FLOAT dRx = (t_left - t_center) * Rx_1;
                    const FLOAT dRz = (amb_temp - t_center) * Rz_1;
                    delta = Cap_1 * (p_center + dRy + dRx + dRz);
                } else if (is_bottom) {
                    /* Bottom row (excluding corners) */
                    p_center = power[idx];
                    const FLOAT t_left  = temp[idx - 1];
                    const FLOAT t_right = temp[idx + 1];
                    const FLOAT t_up    = temp[idx - col];
                    const FLOAT dRx = (t_right + t_left - (FLOAT)2.0 * t_center) * Rx_1;
                    const FLOAT dRy = (t_up    - t_center) * Ry_1;
                    const FLOAT dRz = (amb_temp - t_center) * Rz_1;
                    delta = Cap_1 * (p_center + dRx + dRy + dRz);
                } else if (is_left) {
                    /* Left column (excluding corners) */
                    p_center = power[base];
                    const FLOAT t_up   = temp[base - col];
                    const FLOAT t_down = temp[base + col];
                    const FLOAT t_right= temp[base + 1];
                    const FLOAT dRy = (t_down + t_up - (FLOAT)2.0 * t_center) * Ry_1;
                    const FLOAT dRx = (t_right - t_center) * Rx_1;
                    const FLOAT dRz = (amb_temp - t_center) * Rz_1;
                    delta = Cap_1 * (p_center + dRy + dRx + dRz);
                } else {
                    /* Interior cell reached only when block touches boundary but cell itself is internal */
                    const FLOAT t_up    = temp[idx - col];
                    const FLOAT t_down  = temp[idx + col];
                    const FLOAT t_left  = temp[idx - 1];
                    const FLOAT t_right = temp[idx + 1];
                    p_center = power[idx];
                    const FLOAT dRy = (t_down + t_up   - (FLOAT)2.0 * t_center) * Ry_1;
                    const FLOAT dRx = (t_right + t_left - (FLOAT)2.0 * t_center) * Rx_1;
                    const FLOAT dRz = (amb_temp - t_center) * Rz_1;
                    delta = Cap_1 * (p_center + dRy + dRx + dRz);
                }

                result[idx] = t_center + delta;
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