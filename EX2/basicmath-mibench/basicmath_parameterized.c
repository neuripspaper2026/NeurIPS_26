#include "snipmath.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Parameterized version of basicmath benchmark */

typedef struct {
    const char* name;
    // Cubic equation parameters
    int cubic_a_max;
    double cubic_b_step;
    double cubic_c_step;
    double cubic_d_step;
    int cubic_d_min;
    // Integer square root parameters
    int sqrt_max_1;
    int sqrt_step_1;
    unsigned long sqrt_start_2;
    unsigned long sqrt_end_2;
    // Angle conversion parameters
    double angle_deg_step;
    double angle_rad_divisor;  // PI/divisor
} BenchmarkConfig;

// Predefined configurations for different sizes
BenchmarkConfig configs[] = {
    {
        "mini",
        10,           // cubic_a_max: 1 to <10 = 9 values
        0.25,         // cubic_b_step: 10 to >0, step 0.25 = 40 values
        0.61,         // cubic_c_step: 5 to <15, step 0.61 = ~17 values
        0.451,        // cubic_d_step: -1 to >-5, step 0.451 = ~9 values
        -5,           // cubic_d_min: stop at -5
        100000,       // sqrt_max_1: 0 to <100000, step 2 = 50000 values
        2,            // sqrt_step_1
        0x3fed0169L,  // sqrt_start_2
        0x3fed4169L,  // sqrt_end_2: range = 16384 values
        0.001,        // angle_deg_step: 0 to 360, step 0.001 = 360001 values
        5760.0        // angle_rad_divisor: PI/5760 = ~36481 values
    },
    {
        "small",
        10,           // cubic: 9 * 80 * 34 * 18 = ~440,640
        0.125,        // b: 10 to >0, step 0.125 = 80 values
        0.3,          // c: 5 to <15, step 0.3 = ~34 values
        0.225,        // d: -1 to >-5, step 0.225 = ~18 values
        -5,
        200000,       // sqrt_1: 200000 / 2 = 100,000 values
        2,
        0x3fed0169L,
        0x3fed8169L,  // sqrt_2: range = 32768 values
        0.0005,       // angle_deg: 360 / 0.0005 = 720,000 values
        11520.0       // angle_rad: ~72,962 values
    },
    {
        "medium",
        12,           // cubic: 11 * 160 * 67 * 36 = ~4,236,480
        0.0625,       // b: 10 / 0.0625 = 160 values
        0.15,         // c: 10 / 0.15 = ~67 values
        0.1125,       // d: 4 / 0.1125 = ~36 values
        -5,
        500000,       // sqrt_1: 250,000 values
        2,
        0x3fed0169L,
        0x3fedc169L,  // sqrt_2: range = 49152 values
        0.00025,      // angle_deg: 1,440,000 values
        23040.0       // angle_rad: ~145,924 values
    },
    {
        "large",
        15,           // cubic: 14 * 320 * 134 * 72 = ~43,161,600
        0.03125,      // b: 10 / 0.03125 = 320 values
        0.075,        // c: 10 / 0.075 = ~134 values
        0.05625,      // d: 4 / 0.05625 = ~72 values
        -5,
        1000000,      // sqrt_1: 500,000 values
        2,
        0x3fed0169L,
        0x3fee0169L,  // sqrt_2: range = 65536 values
        0.000125,     // angle_deg: 2,880,000 values
        46080.0       // angle_rad: ~291,848 values
    },
    {
        "extra-large",
        18,           // cubic: 17 * 480 * 200 * 108 = ~176,256,000
        0.0208333,    // b: 10 / 0.0208333 = 480 values
        0.05,         // c: 10 / 0.05 = 200 values
        0.0375,       // d: 4 / 0.0375 = ~108 values
        -5,
        3000000,      // sqrt_1: 1,500,000 values
        2,
        0x3fed0169L,
        0x3fee3169L,  // sqrt_2: range = 77824 values
        0.00008,      // angle_deg: 4,500,000 values
        69120.0       // angle_rad: ~437,772 values
    }
};

BenchmarkConfig* get_config(const char* size_name) {
    int num_configs = sizeof(configs) / sizeof(configs[0]);
    for (int i = 0; i < num_configs; i++) {
        if (strcmp(configs[i].name, size_name) == 0) {
            return &configs[i];
        }
    }
    return NULL;
}

void print_usage(const char* prog_name) {
    printf("Usage: %s [size]\n", prog_name);
    printf("Available sizes:\n");
    int num_configs = sizeof(configs) / sizeof(configs[0]);
    for (int i = 0; i < num_configs; i++) {
        printf("  - %s\n", configs[i].name);
    }
    printf("Default: mini\n");
}

int main(int argc, char* argv[])
{
    int ret = 0;
    struct timespec main_start, main_end;
    clock_gettime(CLOCK_MONOTONIC, &main_start);
    FILE *timing_file = stderr;
    const char *timing_path = getenv("TIMING_LOG_FILE");
    if (timing_path && timing_path[0] != '\0') {
        FILE *tmp = fopen(timing_path, "w");
        if (tmp) {
            timing_file = tmp;
        }
    }
    reset_solve_cubic_kernel_time();

    const char* size_name = "mini";  // default
    
    // Parse command line arguments
    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            print_usage(argv[0]);
            goto timing_cleanup;
        }
        size_name = argv[1];
    }
    
    // Get configuration
    BenchmarkConfig* cfg = get_config(size_name);
    if (cfg == NULL) {
        fprintf(stderr, "Error: Unknown size '%s'\n", size_name);
        print_usage(argv[0]);
        ret = 1;
        goto timing_cleanup;
    }
    
    // printf("Running basicmath benchmark with size: %s\n", cfg->name);
    
    double  a1 = 1.0, b1 = -10.5, c1 = 32.0, d1 = -30.0;
    double  x[3];
    double X;
    int     solutions;
    int i;
    unsigned long l = 0x3fed0169L;
    struct int_sqrt q;
    long n = 0;

    /* solve some cubic functions */
    printf("********* CUBIC FUNCTIONS ***********\n");
    /* should get 3 solutions: 2, 6 & 2.5   */
    SolveCubic(a1, b1, c1, d1, &solutions, x);  
    printf("Solutions:");
    for(i=0;i<solutions;i++)
        printf(" %f",x[i]);
    printf("\n");

    a1 = 1.0; b1 = -4.5; c1 = 17.0; d1 = -30.0;
    /* should get 1 solution: 2.5           */
    SolveCubic(a1, b1, c1, d1, &solutions, x);  
    printf("Solutions:");
    for(i=0;i<solutions;i++)
        printf(" %f",x[i]);
    printf("\n");

    a1 = 1.0; b1 = -3.5; c1 = 22.0; d1 = -31.0;
    SolveCubic(a1, b1, c1, d1, &solutions, x);
    printf("Solutions:");
    for(i=0;i<solutions;i++)
        printf(" %f",x[i]);
    printf("\n");

    a1 = 1.0; b1 = -13.7; c1 = 1.0; d1 = -35.0;
    SolveCubic(a1, b1, c1, d1, &solutions, x);
    printf("Solutions:");
    for(i=0;i<solutions;i++)
        printf(" %f",x[i]);
    printf("\n");

    a1 = 3.0; b1 = 12.34; c1 = 5.0; d1 = 12.0;
    SolveCubic(a1, b1, c1, d1, &solutions, x);
    printf("Solutions:");
    for(i=0;i<solutions;i++)
        printf(" %f",x[i]);
    printf("\n");

    a1 = -8.0; b1 = -67.89; c1 = 6.0; d1 = -23.6;
    SolveCubic(a1, b1, c1, d1, &solutions, x);
    printf("Solutions:");
    for(i=0;i<solutions;i++)
        printf(" %f",x[i]);
    printf("\n");

    a1 = 45.0; b1 = 8.67; c1 = 7.5; d1 = 34.0;
    SolveCubic(a1, b1, c1, d1, &solutions, x);
    printf("Solutions:");
    for(i=0;i<solutions;i++)
        printf(" %f",x[i]);
    printf("\n");

    a1 = -12.0; b1 = -1.7; c1 = 5.3; d1 = 16.0;
    SolveCubic(a1, b1, c1, d1, &solutions, x);
    printf("Solutions:");
    for(i=0;i<solutions;i++)
        printf(" %f",x[i]);
    printf("\n");

    /* Now solve some random equations using config parameters */
    for(a1=1; a1<cfg->cubic_a_max; a1+=1) {
        for(b1=10; b1>0; b1-=cfg->cubic_b_step) {
            for(c1=5; c1<15; c1+=cfg->cubic_c_step) {
                for(d1=-1; d1>cfg->cubic_d_min; d1-=cfg->cubic_d_step) {
                    SolveCubic(a1, b1, c1, d1, &solutions, x);  
                    printf("Solutions:");
                    for(i=0;i<solutions;i++)
                        printf(" %f",x[i]);
                    printf("\n");
                }
            }
        }
    }

    printf("********* INTEGER SQR ROOTS ***********\n");
    /* perform some integer square roots */
    for (i = 0; i < cfg->sqrt_max_1; i+=cfg->sqrt_step_1)
    {
        usqrt(i, &q);
        printf("sqrt(%3d) = %2d\n", i, q.sqrt);
    }
    printf("\n");
    
    for (l = cfg->sqrt_start_2; l < cfg->sqrt_end_2; l++)
    {
        usqrt(l, &q);
        printf("sqrt(%lX) = %X\n", l, q.sqrt);
    }

    printf("********* ANGLE CONVERSION ***********\n");
    /* convert some rads to degrees */
    for (X = 0.0; X <= 360.0; X += cfg->angle_deg_step)
        printf("%3.0f degrees = %.12f radians\n", X, deg2rad(X));
    puts("");
    
    for (X = 0.0; X <= (2 * PI + 1e-6); X += (PI / cfg->angle_rad_divisor))
        printf("%.12f radians = %3.0f degrees\n", X, rad2deg(X));
    
    ret = 0;

timing_cleanup:
    clock_gettime(CLOCK_MONOTONIC, &main_end);
    double kernel_time = get_solve_cubic_kernel_time();
    double main_time = (main_end.tv_sec - main_start.tv_sec) +
                       (main_end.tv_nsec - main_start.tv_nsec) / 1e9;
    fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);
    fflush(timing_file);
    if (timing_file != stderr) {
        fclose(timing_file);
    }
    return ret;
}

