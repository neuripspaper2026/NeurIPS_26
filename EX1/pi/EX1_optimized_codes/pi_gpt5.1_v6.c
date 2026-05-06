#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

double getClock();

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        printf("Usage: %s <steps> [output_file]\n", argv[0]);
        printf("  <steps> controls the precision of the approximation.\n");
        printf("  [output_file] (optional) is the output file for result and error.\n");
        return 0;
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

    unsigned long N = atol(argv[1]);
    const char *output_file = (argc == 3) ? argv[2] : NULL;
    printf("- Input parameters\n");
    printf("steps\t= %lu\n", N);

    printf("- Executing test...\n");
    double time_start = getClock();

    double out_result;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    const double invN = 1.0 / (double)N;
    const double halfInvN = 0.5 * invN;
    const double fourInvN = 4.0 * invN;

    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;

    unsigned long i = 0;
    unsigned long limit = (N / 4UL) * 4UL;

    for (; i < limit; i += 4UL) {
        double x0 = ((double)i + 0.5) * invN;
        double x1 = ((double)(i + 1UL) + 0.5) * invN;
        double x2 = ((double)(i + 2UL) + 0.5) * invN;
        double x3 = ((double)(i + 3UL) + 0.5) * invN;

        sum0 += sqrt(1.0 - x0 * x0);
        sum1 += sqrt(1.0 - x1 * x1);
        sum2 += sqrt(1.0 - x2 * x2);
        sum3 += sqrt(1.0 - x3 * x3);
    }

    for (; i < N; ++i) {
        double x = ((double)i + 0.5) * invN;
        sum0 += sqrt(1.0 - x * x);
    }

    double sum = (sum0 + sum1) + (sum2 + sum3);

    out_result = fourInvN * sum;
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);

    double time_finish = getClock();

    printf("time (s)= %.6f\n", time_finish - time_start);
    printf("result\t= %.8f\n", out_result);
    const double realPiValue = 3.141592653589793238;
    double error = fabs(out_result - realPiValue);
    printf("error\t= %.1e\n", error);

    if (output_file != NULL) {
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            fprintf(fp, "result = %.15f\n", out_result);
            fprintf(fp, "error = %.15e\n", error);
            fclose(fp);
            fprintf(stderr, "- Output written to: %s\n", output_file);
        } else {
            fprintf(stderr, "Error: cannot write to %s\n", output_file);
        }
    }

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

double getClock() {
#ifdef _OPENMP
    return omp_get_wtime();
#elif __linux__ || __APPLE__
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1.0e9;
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}
