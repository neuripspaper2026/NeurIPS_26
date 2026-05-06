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

    pin_stats_reset();

    const int last_col = cols - 1;

    for (int t = 0; t < rows - 1; t++) {
        temp = src;
        src = dst;
        dst = temp;

        int left, center, right, m;

        if (cols > 0) {
            /* n = 0 (left boundary) */
            center = src[0];
            right  = (cols > 1) ? src[1] : src[0];
            m = center < right ? center : right;
            dst[0] = wall[t + 1][0] + m;
        }

        if (cols > 2) {
            /* interior points with manual unrolling */
            int n = 1;
#if defined(_OPENMP)
#pragma omp parallel for private(left, center, right, m) schedule(static)
#endif
            for (n = 1; n <= last_col - 2; n += 2) {
                left   = src[n - 1];
                center = src[n];
                right  = src[n + 1];
                m = center < left ? center : left;
                m = right < m ? right : m;
                dst[n] = wall[t + 1][n] + m;

                left   = src[n];
                center = src[n + 1];
                right  = src[n + 2];
                m = center < left ? center : left;
                m = right < m ? right : m;
                dst[n + 1] = wall[t + 1][n + 1] + m;
            }
            /* handle remaining interior element if cols-2 is odd */
            for (; n <= last_col - 1; n++) {
                left   = src[n - 1];
                center = src[n];
                right  = src[n + 1];
                m = center < left ? center : left;
                m = right < m ? right : m;
                dst[n] = wall[t + 1][n] + m;
            }
        } else if (cols == 2) {
            int n = 1;
            int left   = src[n - 1];
            int center = src[n];
            int right  = src[n];
            int m = center < left ? center : left;
            m = right < m ? right : m;
            dst[n] = wall[t + 1][n] + m;
        }

        if (cols > 1) {
            /* n = last_col (right boundary) */
            int n = last_col;
            int left   = src[n - 1];
            int center = src[n];
            int right  = src[n];
            int m = center < left ? center : left;
            m = right < m ? right : m;
            dst[n] = wall[t + 1][n] + m;
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
    delete[] src;

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    g_kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
