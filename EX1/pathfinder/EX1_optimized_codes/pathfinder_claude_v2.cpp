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
    int min;

    dst = result;
    src = new int[cols];

    pin_stats_reset();
    
    // Prefetch wall data
    for (int t = 0; t < rows - 1; t++) {
        __builtin_prefetch(&wall[t + 1][0], 0, 3);
    }
    
    for (int t = 0; t < rows - 1; t++) {
        temp = src;
        src = dst;
        dst = temp;
        
        // Prefetch next iteration's wall data
        if (t + 2 < rows) {
            __builtin_prefetch(&wall[t + 2][0], 0, 3);
        }
        
        // Handle first element
        min = src[0];
        if (cols > 1)
            min = MIN(min, src[1]);
        dst[0] = wall[t + 1][0] + min;
        
        // Main loop with loop unrolling
        int n;
        for (n = 1; n < cols - 1 - 3; n += 4) {
            // Unroll 4 iterations
            int min0 = src[n];
            min0 = MIN(min0, src[n - 1]);
            min0 = MIN(min0, src[n + 1]);
            dst[n] = wall[t + 1][n] + min0;
            
            int min1 = src[n + 1];
            min1 = MIN(min1, src[n]);
            min1 = MIN(min1, src[n + 2]);
            dst[n + 1] = wall[t + 1][n + 1] + min1;
            
            int min2 = src[n + 2];
            min2 = MIN(min2, src[n + 1]);
            min2 = MIN(min2, src[n + 3]);
            dst[n + 2] = wall[t + 1][n + 2] + min2;
            
            int min3 = src[n + 3];
            min3 = MIN(min3, src[n + 2]);
            min3 = MIN(min3, src[n + 4]);
            dst[n + 3] = wall[t + 1][n + 3] + min3;
        }
        
        // Handle remaining elements
        for (; n < cols - 1; n++) {
            min = src[n];
            min = MIN(min, src[n - 1]);
            min = MIN(min, src[n + 1]);
            dst[n] = wall[t + 1][n] + min;
        }
        
        // Handle last element
        if (cols > 1) {
            min = src[cols - 1];
            min = MIN(min, src[cols - 2]);
            dst[cols - 1] = wall[t + 1][cols - 1] + min;
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
