/*
 * =====================================================================================
 *
 *       Filename:  suite.c
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include "common.h"

static int do_verify = 0;
int omp_num_threads = 1;

static struct option long_options[] = {
    /* name, has_arg, flag, val */
    {"input", 1, NULL, 'i'},
    {"size", 1, NULL, 's'},
    {"verify", 0, NULL, 'v'},
    {"output", 1, NULL, 'o'},
    {0, 0, 0, 0}};

extern void lud_omp(float *m, int matrix_dim);

int main(int argc, char *argv[]) {
    int matrix_dim = 32; /* default size */
    int opt, option_index = 0;
    func_ret_t ret;
    const char *input_file = NULL;
    const char *output_file = NULL;
    FILE *output_fp = NULL;
    float *m, *mm;
    stopwatch sw;
    int i, j;
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


    while ((opt = getopt_long(argc, argv, "::vs:i:o:", long_options,
                              &option_index)) != -1) {
        switch (opt) {
        case 'i':
            input_file = optarg;
            break;
        case 'v':
            do_verify = 1;
            break;
        case 's':
            matrix_dim = atoi(optarg);
            printf("Generate input matrix internally, size =%d\n", matrix_dim);
            // fprintf(stderr, "Currently not supported, use -i instead\n");
            // fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i
            // input_file]\n", argv[0]);
            // exit(EXIT_FAILURE);
            break;
        case 'o':
            output_file = optarg;
            break;
        case '?':
            fprintf(stderr, "invalid option\n");
            break;
        case ':':
            fprintf(stderr, "missing argument\n");
            break;
        default:
            fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file] [-o output_file]\n",
                    argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if ((optind < argc) || (optind == 1)) {
        fprintf(stderr, "Usage: %s [-v] [-n no. of threads] [-s matrix_size|-i input_file] [-o output_file]\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

    if (input_file) {
        printf("Reading matrix from file %s\n", input_file);
        ret = create_matrix_from_file(&m, input_file, &matrix_dim);
        if (ret != RET_SUCCESS) {
            m = NULL;
            fprintf(stderr, "error create matrix from file %s\n", input_file);
            exit(EXIT_FAILURE);
        }
    } else if (matrix_dim) {
        printf("Creating matrix internally size=%d\n", matrix_dim);
        ret = create_matrix(&m, matrix_dim);
        if (ret != RET_SUCCESS) {
            m = NULL;
            fprintf(stderr, "error create matrix internally size=%d\n",
                    matrix_dim);
            exit(EXIT_FAILURE);
        }
    }

    else {
        printf("No input file specified!\n");
        exit(EXIT_FAILURE);
    }

    if (do_verify) {
        printf("Before LUD\n");
        /* print_matrix(m, matrix_dim); */
        matrix_duplicate(m, &mm, matrix_dim);
    }


    clock_gettime(CLOCK_MONOTONIC, &kernel_start);
    stopwatch_start(&sw);
    lud_omp(m, matrix_dim);
    stopwatch_stop(&sw);
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    if (output_file != NULL) {
        output_fp = fopen(output_file, "w");
        if (output_fp == NULL) {
            fprintf(stderr, "Failed to open output file %s\n", output_file);
            exit(EXIT_FAILURE);
        }
        for (i = 0; i < matrix_dim; i++) {
            for (j = 0; j < matrix_dim; j++) {
                fprintf(output_fp, "%f ", m[i * matrix_dim + j]);
            }
            fprintf(output_fp, "\n");
        }
        fclose(output_fp);
    }
    printf("Time consumed(s): %lf\n", get_interval_by_sec(&sw));

    if (do_verify) {
        printf("After LUD\n");
        /* print_matrix(m, matrix_dim); */
        printf(">>>Verify<<<<\n");
        lud_verify(mm, m, matrix_dim);
        free(mm);
    }

    free(m);

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
} /* ----------  end of function main  ---------- */
