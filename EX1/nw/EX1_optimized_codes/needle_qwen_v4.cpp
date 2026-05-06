#define LIMIT -999
//#define TRACE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
//#define NUM_THREAD 4

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);
static const char *g_output_file = NULL;
int maximum(int a, int b, int c) {

    int k;
    if (a <= b)
        k = b;
    else
        k = a;

    if (k <= c)
        return (c);
    else
        return (k);
}


int blosum62[24][24] = {{4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2,
                         -1, 1, 0, -3, -2, 0, -2, -1, 0, -4},
                        {-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2,
                         -1, -1, -3, -2, -3, -1, 0, -1, -4},
                        {-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1,
                         0, -4, -2, -3, 3, 0, -1, -4},
                        {-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1,
                         0, -1, -4, -3, -3, 4, 1, -1, -4},
                        {0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2,
                         -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
                        {-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0,
                         -1, -2, -1, -2, 0, 3, -1, -4},
                        {-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0,
                         -1, -3, -2, -2, 1, 4, -1, -4},
                        {0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3,
                         -2, 0, -2, -2, -3, -3, -1, -2, -1, -4},
                        {-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2,
                         -1, -2, -2, 2, -3, 0, 0, -1, -4},
                        {-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3,
                         -2, -1, -3, -1, 3, -3, -3, -1, -4},
                        {-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3,
                         -2, -1, -2, -1, 1, -4, -3, -1, -4},
                        {-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1,
                         0, -1, -3, -2, -2, 0, 1, -1, -4},
                        {-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2,
                         -1, -1, -1, -1, 1, -3, -1, -1, -4},
                        {-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4,
                         -2, -2, 1, 3, -1, -3, -3, -1, -4},
                        {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,
                         7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
                        {1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4,
                         1, -3, -2, -2, 0, 0, 0, -4},
                        {0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2,
                         -1, 1, 5, -2, -2, 0, -1, -1, 0, -4},
                        {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,
                         -4, -3, -2, 11, 2, -3, -4, -3, -2, -4},
                        {-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3,
                         -3, -2, -2, 2, 7, -1, -3, -2, -1, -4},
                        {0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2,
                         -2, 0, -3, -1, 4, -3, -2, -1, -4},
                        {-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2,
                         0, -1, -4, -3, -3, 4, 1, -1, -4},
                        {-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0,
                         -1, -3, -2, -2, 1, 4, -1, -4},
                        {0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -2, 0, 0, -2, -1, -1, -1, -1, -1, -4},
                        {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
                         -4, -4, -4, -4, -4, -4, -4, -4, -4, 1}};

double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
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

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);
    runTest(argc, argv);
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);

    clock_gettime(CLOCK_MONOTONIC, &main_end);

    double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                         (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    double main_time = (main_end.tv_sec - main_start.tv_sec) +
                       (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

    fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

    if (timing_file != stderr)
        fclose(timing_file);

    return EXIT_SUCCESS;
}

void usage(int argc, char **argv) {
    fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> <num_threads>\n",
            argv[0]);
    fprintf(stderr, "\t<dimension>      - x and y dimensions\n");
    fprintf(stderr, "\t<penalty>        - penalty(positive integer)\n");
    fprintf(stderr, "\t<num_threads>    - no. of threads\n");
    exit(1);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
    int max_rows, max_cols, penalty, idx, index;
    int *input_itemsets, *output_itemsets, *referrence;
    int *matrix_cuda, *matrix_cuda_out, *referrence_cuda;
    int size;
    int omp_num_threads;


    // the lengths of the two sequences should be able to divided by 16.
    // And at current stage  max_rows needs to equal max_cols
    if (argc == 4 || argc == 6) {
        max_rows = atoi(argv[1]);
        max_cols = atoi(argv[1]);
        penalty = atoi(argv[2]);
        omp_num_threads = atoi(argv[3]);
        if (argc == 6) {
            if (strcmp(argv[4], "-o") == 0) {
                g_output_file = argv[5];
            } else {
                usage(argc, argv);
            }
        }
    } else {
        usage(argc, argv);
    }

    max_rows = max_rows + 1;
    max_cols = max_cols + 1;
    referrence = (int *)malloc(max_rows * max_cols * sizeof(int));
    input_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));
    output_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));


    if (!output_itemsets)
        fprintf(stderr, "error: can not allocate memory");

    srand(7);

    // Initialize input_itemsets to zero
    memset(input_itemsets, 0, max_rows * max_cols * sizeof(int));

    printf("Start Needleman-Wunsch\n");

    // Initialize first column
    for (int i = 1; i < max_rows; i++) {
        input_itemsets[i * max_cols] = rand() % 10 + 1;
    }
    // Initialize first row
    for (int j = 1; j < max_cols; j++) {
        input_itemsets[j] = rand() % 10 + 1;
    }

    // Compute reference matrix
    for (int i = 1; i < max_cols; i++) {
        const int i_offset = i * max_cols;
        const int input_i = input_itemsets[i_offset];
        for (int j = 1; j < max_rows; j++) {
            referrence[i_offset + j] = blosum62[input_i][input_itemsets[j]];
        }
    }

    // Initialize penalties
    for (int i = 1; i < max_rows; i++)
        input_itemsets[i * max_cols] = -i * penalty;
    for (int j = 1; j < max_cols; j++)
        input_itemsets[j] = -j * penalty;


    // Compute top-left matrix
    printf("Num of threads: %d\n", omp_num_threads);
    printf("Processing top-left matrix\n");

    for (int i = 0; i < max_cols - 2; i++) {
        for (idx = 0; idx <= i; idx++) {
            index = (idx + 1) * max_cols + (i + 1 - idx);
            input_itemsets[index] = maximum(
                input_itemsets[index - 1 - max_cols] + referrence[index],
                input_itemsets[index - 1] - penalty,
                input_itemsets[index - max_cols] - penalty);
        }
    }
    printf("Processing bottom-right matrix\n");
    // Compute bottom-right matrix
    for (int i = max_cols - 4; i >= 0; i--) {
        for (idx = 0; idx <= i; idx++) {
            index = (max_cols - idx - 2) * max_cols + idx + max_cols - i - 2;
            input_itemsets[index] = maximum(
                input_itemsets[index - 1 - max_cols] + referrence[index],
                input_itemsets[index - 1] - penalty,
                input_itemsets[index - max_cols] - penalty);
        }
    }

    // Write final alignment score to file if requested
    if (g_output_file != NULL) {
        int final_score = input_itemsets[(max_rows - 1) * max_cols + (max_cols - 1)];
        FILE *fout = fopen(g_output_file, "w");
        if (fout) {
            fprintf(fout, "%d\n", final_score);
            fclose(fout);
        } else {
            fprintf(stderr, "Failed to open output file: %s\n", g_output_file);
        }
    }

//#define TRACEBACK
#ifdef TRACEBACK

    FILE *fpo = fopen("result.txt", "w");
    fprintf(fpo, "print traceback value GPU:\n");

    for (int i = max_rows - 2, j = max_rows - 2; i >= 0 && j >= 0;) {
        int nw, n, w, traceback;
        if (i == max_rows - 2 && j == max_rows - 2)
            fprintf(
                fpo, "%d ",
                input_itemsets[i * max_cols + j]); // print the first element
        if (i == 0 && j == 0)
            break;
        if (i > 0 && j > 0) {
            nw = input_itemsets[(i - 1) * max_cols + j - 1];
            w = input_itemsets[i * max_cols + j - 1];
            n = input_itemsets[(i - 1) * max_cols + j];
        } else if (i == 0) {
            nw = n = LIMIT;
            w = input_itemsets[i * max_cols + j - 1];
        } else if (j == 0) {
            nw = w = LIMIT;
            n = input_itemsets[(i - 1) * max_cols + j];
        } else {
        }

        // traceback = maximum(nw, w, n);
        int new_nw, new_w, new_n;
        new_nw = nw + referrence[i * max_cols + j];
        new_w = w - penalty;
        new_n = n - penalty;

        traceback = maximum(new_nw, new_w, new_n);
        if (traceback == new_nw)
            traceback = nw;
        if (traceback == new_w)
            traceback = w;
        if (traceback == new_n)
            traceback = n;

        fprintf(fpo, "%d ", traceback);

        if (traceback == nw) {
            i--;
            j--;
            continue;
        }

        else if (traceback == w) {
            j--;
            continue;
        }

        else if (traceback == n) {
            i--;
            continue;
        }

        else
            ;
    }

    fclose(fpo);

#endif

    free(referrence);
    free(input_itemsets);
    free(output_itemsets);
}
