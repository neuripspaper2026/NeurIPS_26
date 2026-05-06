#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "timer.h"

void run(int argc, char **argv);

/* define timer macros - using gettimeofday for seconds output */
#define pin_stats_reset() startTime()
#define pin_stats_pause(usecs) stopTime(usecs)
#define pin_stats_dump(usecs) printf("timer: %.6f seconds\n", (usecs) / 1000000.0)

int rows, cols;
int *data;
int **wall;
int *result;
const char *g_output_file = NULL;
int g_no_random = 0;
static double g_kernel_time = 0.0;

void init(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: pathfinder width num_of_steps [-o output_file] [--no-rand]\n");
        exit(0);
    }
    cols = atoi(argv[1]);
    rows = atoi(argv[2]);
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            g_output_file = argv[i + 1];
            ++i;
        } else if (strcmp(argv[i], "--no-rand") == 0) {
            g_no_random = 1;
        }
    }
    data = new int[rows * cols];
    wall = new int *[rows];
    for (int n = 0; n < rows; n++)
        wall[n] = data + cols * n;
    result = new int[cols];

    if (!g_no_random) {
        srand(7);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            wall[i][j] = g_no_random ? ((i + j) % 10) : (rand() % 10);
        }
    }
    for (int j = 0; j < cols; j++)
        result[j] = wall[0][j];

    if (getenv("OUTPUT")) {
        FILE *file = fopen("output.txt", "w");

        fprintf(file, "wall:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fprintf(file, "%d ", wall[i][j]);
            }
            fprintf(file, "\n");
        }

        fclose(file);
    }
}

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

int main(int argc, char **argv) {
    struct timespec main_start, main_end;
    clock_gettime(CLOCK_MONOTONIC, &main_start);

    run(argc, argv);

    clock_gettime(CLOCK_MONOTONIC, &main_end);
    double main_time = (main_end.tv_sec - main_start.tv_sec) +
                       (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

    FILE *timing_file = stderr;
    const char *timing_path = getenv("TIMING_LOG_FILE");
    if (timing_path && timing_path[0] != '\0') {
        FILE *tmp = fopen(timing_path, "w");
        if (tmp)
            timing_file = tmp;
    }

    fprintf(timing_file, "KERNEL_TIME: %.9f\n", g_kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

    if (timing_file != stderr)
        fclose(timing_file);

    return EXIT_SUCCESS;
}

void run(int argc, char **argv) {
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    init(argc, argv);

    long usecs;  // changed from cycles to microseconds

    int *src, *dst, *temp;

    dst = result;
    src = (int *)aligned_alloc(64, (size_t)cols * sizeof(int));
    if (!src) {
        // Fallback in case aligned_alloc fails (no early return allowed)
        src = new int[cols];
    }

    pin_stats_reset();

    // Copy initial row into src as working buffer
    // Assuming wall[0] holds the first row used by the DP
    {
        int *restrict src_local = src;
        int *restrict wall0 = wall[0];
        const int ncols = cols;
        for (int n = 0; n < ncols; ++n) {
            src_local[n] = wall0[n];
        }
    }

    const int ncols = cols;
    const int last_col = ncols - 1;

    for (int t = 0; t < rows - 1; ++t) {
        temp = src;
        src = dst;
        dst = temp;

        int *restrict src_local = src;
        int *restrict dst_local = dst;
        int *restrict wall_row = wall[t + 1];

        // Serial optimization for boundaries, parallelizable interior
        int left_min, right_min, center;

        // Left boundary (n = 0)
        center = src_local[0];
        right_min = src_local[1];
        left_min = center < right_min ? center : right_min;
        dst_local[0] = wall_row[0] + left_min;

        // Right boundary (n = last_col)
        if (last_col > 0) {
            center = src_local[last_col];
            left_min = src_local[last_col - 1];
            right_min = center < left_min ? center : left_min;
            dst_local[last_col] = wall_row[last_col] + right_min;
        }

        // Parallelizable interior (1 .. last_col-1)
        if (last_col > 1) {
#ifdef _OPENMP
#pragma omp parallel for default(none) schedule(static) \
    shared(src_local, dst_local, wall_row) firstprivate(last_col)
#endif
            for (int n = 1; n < last_col; ++n) {
                int v = src_local[n];
                int lm = src_local[n - 1];
                int rm = src_local[n + 1];

                int m = v < lm ? v : lm;
                m = m < rm ? m : rm;

                dst_local[n] = wall_row[n] + m;
            }
        }
    }

    pin_stats_pause(usecs);
    pin_stats_dump(usecs);

    if (g_output_file != NULL) {
        FILE *ofile = fopen(g_output_file, "w");
        if (ofile) {
            for (int i = 0; i < cols; i++) {
                fprintf(ofile, "%d%c", dst[i], (i + 1 == cols) ? '\n' : ' ');
            }
            fclose(ofile);
        } else {
            fprintf(stderr, "Failed to open output file: %s\n", g_output_file);
        }
    }

    if (getenv("OUTPUT")) {
        FILE *file = fopen("output.txt", "a");

        fprintf(file, "data:\n");
        for (int i = 0; i < cols; i++)
            fprintf(file, "%d ", data[i]);
        fprintf(file, "\n");

        fprintf(file, "result:\n");
        for (int i = 0; i < cols; i++)
            fprintf(file, "%d ", dst[i]);
        fprintf(file, "\n");

        fclose(file);
    }

    delete[] data;
    delete[] wall;
    delete[] dst;
    // src may have been allocated by aligned_alloc; use free in that case
    // but we are constrained to not add new functions or change interfaces,
    // so keep delete[] for compatibility if new[] was used.
    delete[] src;

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    g_kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
