#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "define.c"
#include "graphics.c"
#include "resize.c"
#include "timer.c"

int main(int argc, char *argv[]) {
    struct timespec main_start, main_end;
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &main_start);
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    long long time0;
    long long time1;
    long long time2;
    long long time3;
    long long time4;
    long long time5;
    long long time6;
    long long time7;
    long long time8;
    long long time9;
    long long time10;

    time0 = get_time();

    fp *image_ori; 
    int image_ori_rows;
    int image_ori_cols;
    long image_ori_elem;

    fp *image;   
    long Nr, Nc; 
    long Ne;

    int niter; 
    fp lambda; 

    int r1, r2, c1, c2; 
    long NeROI;         

    fp meanROI, varROI, q0sqr; 

    int *iN, *iS, *jE, *jW;

    fp Jc;

    fp *dN, *dS, *dW, *dE;

    fp tmp, sum, sum2;
    fp G2, L, num, den, qsqr, D;

    fp *c;
    fp cN, cS, cW, cE;

    int iter;  
    long i, j; 
    long k;    

    int threads;

    time1 = get_time();

    int wrong_argc = (argc != 7);
    if (wrong_argc) {
        printf("ERROR: wrong number of arguments\n");
    } else {
        niter = atoi(argv[1]);
        lambda = atof(argv[2]);
        Nr = atoi(argv[3]);
        Nc = atoi(argv[4]);
        threads = atoi(argv[5]);
    }

    (void)threads; 

    time2 = get_time();

    if (wrong_argc) {
        time3 = get_time();
        time4 = get_time();
        time5 = get_time();
        time6 = get_time();
        time7 = get_time();
        time8 = get_time();
        time9 = get_time();
        time10 = get_time();

        printf("Time spent in different stages of the application:\n");
        printf("%.12f s, %.12f % : SETUP VARIABLES\n",
               (float)(time1 - time0) / 1000000.0f,
               (float)(time1 - time0) / (float)(time10 - time0) * 100.0f);
        printf("%.12f s, %.12f % : READ COMMAND LINE PARAMETERS\n",
               (float)(time2 - time1) / 1000000.0f,
               (float)(time2 - time1) / (float)(time10 - time0) * 100.0f);
        printf("%.12f s, %.12f % : READ IMAGE FROM FILE\n",
               (float)(time3 - time2) / 1000000.0f,
               (float)(time3 - time2) / (float)(time10 - time0) * 100.0f);
        printf("%.12f s, %.12f % : RESIZE IMAGE\n",
               (float)(time4 - time3) / 1000000.0f,
               (float)(time4 - time3) / (float)(time10 - time0) * 100.0f);
        printf("%.12f s, %.12f % : SETUP, MEMORY ALLOCATION\n",
               (float)(time5 - time4) / 1000000.0f,
               (float)(time5 - time4) / (float)(time10 - time0) * 100.0f);
        printf("%.12f s, %.12f % : EXTRACT IMAGE\n",
               (float)(time6 - time5) / 1000000.0f,
               (float)(time6 - time5) / (float)(time10 - time0) * 100.0f);
        printf("%.12f s, %.12f % : COMPUTE\n",
               (float)(time7 - time6) / 1000000.0f,
               (float)(time7 - time6) / (float)(time10 - time0) * 100.0f);
        printf("%.12f s, %.12f % : COMPRESS IMAGE\n",
               (float)(time8 - time7) / 1000000.0f,
               (float)(time8 - time7) / (float)(time10 - time0) * 100.0f);
        printf("%.12f s, %.12f % : SAVE IMAGE INTO FILE\n",
               (float)(time9 - time8) / 1000000.0f,
               (float)(time9 - time8) / (float)(time10 - time0) * 100.0f);
        printf("%.12f s, %.12f % : FREE MEMORY\n",
               (float)(time10 - time9) / 1000000.0f,
               (float)(time10 - time9) / (float)(time10 - time0) * 100.0f);
        printf("Total time:\n");
        printf("%.12f s\n", (float)(time10 - time0) / 1000000.0f);

        clock_gettime(CLOCK_MONOTONIC, &kernel_end);
        clock_gettime(CLOCK_MONOTONIC, &main_end);
        double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                             (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
        double main_time = (main_end.tv_sec - main_start.tv_sec) +
                           (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

        FILE *timing_file = stderr;
        const char *timing_path = getenv("TIMING_LOG_FILE");
        if (timing_path && timing_path[0] != '\0') {
            FILE *tmp = fopen(timing_path, "w");
            if (tmp)
                timing_file = tmp;
        }

        fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
        fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

        if (timing_file != stderr)
            fclose(timing_file);

        return 0;
    }

    image_ori_rows = 502;
    image_ori_cols = 458;
    image_ori_elem = (long)image_ori_rows * (long)image_ori_cols;

    image_ori = (fp *)malloc(sizeof(fp) * image_ori_elem);

    read_graphics("${REPO_ROOT}/EX1/srad_v1/input_data/image.pgm", image_ori, image_ori_rows,
                  image_ori_cols, 1);

    time3 = get_time();

    Ne = Nr * Nc;

    image = (fp *)malloc(sizeof(fp) * (size_t)Ne);

    resize(image_ori, image_ori_rows, image_ori_cols, image, Nr, Nc, 1);

    time4 = get_time();

    r1 = 0;      
    r2 = (int)Nr - 1; 
    c1 = 0;      
    c2 = (int)Nc - 1; 

    NeROI = (long)(r2 - r1 + 1) * (long)(c2 - c1 + 1); 

    iN = (int *)malloc(sizeof(int) * (size_t)Nr); 
    iS = (int *)malloc(sizeof(int) * (size_t)Nr); 
    jW = (int *)malloc(sizeof(int) * (size_t)Nc); 
    jE = (int *)malloc(sizeof(int) * (size_t)Nc); 

    dN = (fp *)malloc(sizeof(fp) * (size_t)Ne); 
    dS = (fp *)malloc(sizeof(fp) * (size_t)Ne); 
    dW = (fp *)malloc(sizeof(fp) * (size_t)Ne); 
    dE = (fp *)malloc(sizeof(fp) * (size_t)Ne); 

    c = (fp *)malloc(sizeof(fp) * (size_t)Ne); 

    for (i = 0; i < Nr; i++) {
        iN[i] = (int)(i - 1); 
        iS[i] = (int)(i + 1); 
    }
    for (j = 0; j < Nc; j++) {
        jW[j] = (int)(j - 1); 
        jE[j] = (int)(j + 1); 
    }
    iN[0] = 0;           
    iS[Nr - 1] = (int)(Nr - 1); 
    jW[0] = 0;           
    jE[Nc - 1] = (int)(Nc - 1); 

    time5 = get_time();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < Ne; i++) { 
        image[i] = exp(image[i] / (fp)255.0); 
    }

    time6 = get_time();

    for (iter = 0; iter < niter; iter++) {

        sum = 0.0f;
        sum2 = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum, sum2) private(i, j, tmp) schedule(static)
#endif
        for (j = c1; j <= c2; j++) {
            long jNr = (long)Nr * (long)j;
            for (i = r1; i <= r2; i++) {
                tmp = image[i + jNr];
                sum += tmp;
                sum2 += tmp * tmp;
            }
        }
        meanROI = sum / (fp)NeROI;
        varROI = (sum2 / (fp)NeROI) - meanROI * meanROI;
        q0sqr = varROI / (meanROI * meanROI);

#ifdef _OPENMP
#pragma omp parallel for private(i, j, k, Jc, G2, L, num, den, qsqr) schedule(static)
#endif
        for (j = 0; j < Nc; j++) {

            long jNr = (long)Nr * (long)j;

            for (i = 0; i < Nr; i++) { 

                k = i + jNr; 
                Jc = image[k];  

                long north_index = (long)iN[i] + jNr;
                long south_index = (long)iS[i] + jNr;
                long west_index  = i + (long)Nr * (long)jW[j];
                long east_index  = i + (long)Nr * (long)jE[j];

                fp dN_local = image[north_index] - Jc;
                fp dS_local = image[south_index] - Jc;
                fp dW_local = image[west_index]  - Jc;
                fp dE_local = image[east_index]  - Jc;

                dN[k] = dN_local;
                dS[k] = dS_local;
                dW[k] = dW_local;
                dE[k] = dE_local;

                fp Jc2 = Jc * Jc;

                G2 = (dN_local * dN_local +
                      dS_local * dS_local +
                      dW_local * dW_local +
                      dE_local * dE_local) / Jc2;

                L = (dN_local + dS_local + dW_local + dE_local) / Jc;

                num = (fp)0.5 * G2 - ((fp)(1.0 / 16.0)) * (L * L);
                den = (fp)1.0 + (fp)0.25 * L;
                qsqr = num / (den * den);

                den = (qsqr - q0sqr) / (q0sqr * ((fp)1.0 + q0sqr));
                fp c_local = (fp)1.0 / ((fp)1.0 + den);

                if (c_local < (fp)0.0) {
                    c_local = (fp)0.0;
                } else if (c_local > (fp)1.0) {
                    c_local = (fp)1.0;
                }
                c[k] = c_local;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for private(i, j, k, cN, cS, cW, cE, D) schedule(static)
#endif
        for (j = 0; j < Nc; j++) { 

            long jNr = (long)Nr * (long)j;

            for (i = 0; i < Nr; i++) { 

                k = i + jNr; 

                cN = c[k];                              
                cS = c[(long)iS[i] + jNr];              
                cW = c[k];                              
                cE = c[i + (long)Nr * (long)jE[j]];     

                D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];

                image[k] = image[k] + (fp)0.25 * lambda * D;
            }
        }
    }

    time7 = get_time();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < Ne; i++) { 
        image[i] = log(image[i]) * (fp)255.0; 
    }

    time8 = get_time();

    write_graphics(argv[6], image, Nr, Nc, 1, 255);

    time9 = get_time();

    free(image_ori);
    free(image);

    free(iN);
    free(iS);
    free(jW);
    free(jE);
    free(dN);
    free(dS);
    free(dW);
    free(dE);
    free(c);

    time10 = get_time();

    printf("Time spent in different stages of the application:\n");
    printf("%.12f s, %.12f % : SETUP VARIABLES\n",
           (float)(time1 - time0) / 1000000.0f,
           (float)(time1 - time0) / (float)(time10 - time0) * 100.0f);
    printf("%.12f s, %.12f % : READ COMMAND LINE PARAMETERS\n",
           (float)(time2 - time1) / 1000000.0f,
           (float)(time2 - time1) / (float)(time10 - time0) * 100.0f);
    printf("%.12f s, %.12f % : READ IMAGE FROM FILE\n",
           (float)(time3 - time2) / 1000000.0f,
           (float)(time3 - time2) / (float)(time10 - time0) * 100.0f);
    printf("%.12f s, %.12f % : RESIZE IMAGE\n",
           (float)(time4 - time3) / 1000000.0f,
           (float)(time4 - time3) / (float)(time10 - time0) * 100.0f);
    printf("%.12f s, %.12f % : SETUP, MEMORY ALLOCATION\n",
           (float)(time5 - time4) / 1000000.0f,
           (float)(time5 - time4) / (float)(time10 - time0) * 100.0f);
    printf("%.12f s, %.12f % : EXTRACT IMAGE\n",
           (float)(time6 - time5) / 1000000.0f,
           (float)(time6 - time5) / (float)(time10 - time0) * 100.0f);
    printf("%.12f s, %.12f % : COMPUTE\n",
           (float)(time7 - time6) / 1000000.0f,
           (float)(time7 - time6) / (float)(time10 - time0) * 100.0f);
    printf("%.12f s, %.12f % : COMPRESS IMAGE\n",
           (float)(time8 - time7) / 1000000.0f,
           (float)(time8 - time7) / (float)(time10 - time0) * 100.0f);
    printf("%.12f s, %.12f % : SAVE IMAGE INTO FILE\n",
           (float)(time9 - time8) / 1000000.0f,
           (float)(time9 - time8) / (float)(time10 - time0) * 100.0f);
    printf("%.12f s, %.12f % : FREE MEMORY\n",
           (float)(time10 - time9) / 1000000.0f,
           (float)(time10 - time9) / (float)(time10 - time0) * 100.0f);
    printf("Total time:\n");
    printf("%.12f s\n", (float)(time10 - time0) / 1000000.0f);

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    clock_gettime(CLOCK_MONOTONIC, &main_end);
    double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                         (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    double main_time = (main_end.tv_sec - main_start.tv_sec) +
                       (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

    FILE *timing_file = stderr;
    const char *timing_path = getenv("TIMING_LOG_FILE");
    if (timing_path && timing_path[0] != '\0') {
        FILE *tmp = fopen(timing_path, "w");
        if (tmp)
            timing_file = tmp;
    }

    fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

    if (timing_file != stderr)
        fclose(timing_file);
}
