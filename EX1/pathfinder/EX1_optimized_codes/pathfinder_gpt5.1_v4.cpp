#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

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
    src = (int *)malloc((size_t)cols * sizeof(int));
    if (!src) {
        fprintf(stderr, "Memory allocation failed for src\n");
        return;
    }

    pin_stats_reset();

    const int last_row = rows - 1;
    const int last_col_index = cols - 1;

    for (int t = 0; t < last_row; ++t) {
        temp = src;
        src = dst;
        dst = temp;

        int *restrict dst_row = dst;
        int *restrict src_row = src;
        int *restrict wall_row = wall[t + 1];

        if (cols > 0) {
            int min_val = src_row[0];
            if (cols > 1) {
                int right_val = src_row[1];
                min_val = (min_val < right_val) ? min_val : right_val;
            }
            dst_row[0] = wall_row[0] + min_val;
        }

        int n = 1;
        const int limit = last_col_index;

        for (; n + 1 < limit; n += 2) {
            int left0   = src_row[n - 1];
            int center0 = src_row[n];
            int right0  = src_row[n + 1];

            int min0 = center0 < left0 ? center0 : left0;
            min0 = (min0 < right0) ? min0 : right0;
            dst_row[n] = wall_row[n] + min0;

            int left1   = src_row[n];
            int center1 = src_row[n + 1];
            int right1  = src_row[n + 2];

            int min1 = center1 < left1 ? center1 : left1;
            min1 = (min1 < right1) ? min1 : right1;
            dst_row[n + 1] = wall_row[n + 1] + min1;
        }

        for (; n < limit; ++n) {
            int left   = src_row[n - 1];
            int center = src_row[n];
            int right  = src_row[n + 1];

            int minv = center < left ? center : left;
            minv = (minv < right) ? minv : right;
            dst_row[n] = wall_row[n] + minv;
        }

        if (cols > 1) {
            int left_val  = src_row[last_col_index - 1];
            int center_val = src_row[last_col_index];
            int min_val = center_val < left_val ? center_val : left_val;
            dst_row[last_col_index] = wall_row[last_col_index] + min_val;
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
        if (file) {
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
    }

    free(data);
    free(wall);
    free(dst);
    free(src);

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    g_kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
