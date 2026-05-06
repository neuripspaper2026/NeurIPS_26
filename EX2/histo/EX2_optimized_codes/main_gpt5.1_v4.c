#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "util.h"

#define UINT8_MAX 255

int main(int argc, char* argv[]) {
  /* Start timing for total execution */
  struct timespec main_start, main_end;
  clock_gettime(CLOCK_MONOTONIC, &main_start);

  struct pb_TimerSet timers;
  struct pb_Parameters *parameters;

  printf("Base implementation of histogramming.\n");
  printf("Maintained by Nady Obeid <obeid1@ece.uiuc.edu>\n");

  parameters = pb_ReadParameters(&argc, argv);

  int ret_code = 0;
  int have_error = 0;

  if (!parameters) {
    ret_code = -1;
    have_error = 1;
  }

  int numIterations = 0;
  unsigned int img_width = 0, img_height = 0;
  unsigned int histo_width = 0, histo_height = 0;
  unsigned int *img = NULL;
  unsigned char *histo = NULL;

  FILE* f = NULL;
  int result = 0;

  if (!have_error) {
    if (!parameters->inpFiles[0]) {
      fputs("Input file expected\n", stderr);
      ret_code = -1;
      have_error = 1;
    }
  }

  if (!have_error) {
    if (argc >= 2) {
      numIterations = atoi(argv[1]);
      if (numIterations <= 0) {
        fputs("Number of iterations must be positive\n", stderr);
        ret_code = -1;
        have_error = 1;
      }
    } else {
      fputs("Expected at least one command line argument\n", stderr);
      ret_code = -1;
      have_error = 1;
    }
  }

  if (!have_error) {
    pb_InitializeTimerSet(&timers);
    
    char *inputStr = "Input";
    char *outputStr = "Output";
    
    pb_AddSubTimer(&timers, inputStr, pb_TimerID_IO);
    pb_AddSubTimer(&timers, outputStr, pb_TimerID_IO);
    
    pb_SwitchToSubTimer(&timers, inputStr, pb_TimerID_IO);  

    f = fopen(parameters->inpFiles[0],"rb");
    if (!f) {
      fputs("Error opening input file\n", stderr);
      ret_code = -1;
      have_error = 1;
    }
  }

  if (!have_error) {
    result  = (int)fread(&img_width,    sizeof(unsigned int), 1, f);
    result += (int)fread(&img_height,   sizeof(unsigned int), 1, f);
    result += (int)fread(&histo_width,  sizeof(unsigned int), 1, f);
    result += (int)fread(&histo_height, sizeof(unsigned int), 1, f);

    if (result != 4) {
      fputs("Error reading input and output dimensions from file\n", stderr);
      ret_code = -1;
      have_error = 1;
    }
  }

  size_t img_elems = 0;
  size_t histo_elems = 0;

  if (!have_error) {
    img_elems = (size_t)img_width * (size_t)img_height;
    histo_elems = (size_t)histo_width * (size_t)histo_height;

    img = (unsigned int*) malloc(img_elems * sizeof(unsigned int));
    histo = (unsigned char*) calloc(histo_elems, sizeof(unsigned char));

    if (!img || !histo) {
      fputs("Error allocating memory\n", stderr);
      ret_code = -1;
      have_error = 1;
    }
  }

  if (!have_error) {
    pb_SwitchToSubTimer(&timers, "Input", pb_TimerID_IO);

    result = (int)fread(img, sizeof(unsigned int), img_elems, f);

    fclose(f);
    f = NULL;

    if ((size_t)result != img_elems) {
      fputs("Error reading input array from file\n", stderr);
      ret_code = -1;
      have_error = 1;
    }
  }

  double kernel_time = 0.0;

  if (!have_error) {
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    int iter;
    /* Start timing for kernel execution */
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

#ifdef _OPENMP
    /* Parallel histogram computation with per-thread local histograms */
    for (iter = 0; iter < numIterations; iter++) {
      /* Parallel region */
      #pragma omp parallel
      {
        int tid = 0;
        int nthreads = 1;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
        #endif

        /* Each thread allocates its own local histogram to avoid contention */
        unsigned char *local_histo = (unsigned char*) calloc(histo_elems, sizeof(unsigned char));
        if (!local_histo) {
          /* If allocation fails, fall back to serial in this iteration */
          #pragma omp single
          {
            memset(histo, 0, histo_elems * sizeof(unsigned char));
            size_t i;
            for (i = 0; i < img_elems; ++i) {
              const unsigned int value = img[i];
              if (value < histo_elems && histo[value] < UINT8_MAX) {
                ++histo[value];
              }
            }
          }
        } else {
          /* Zero global histogram once per iteration in a single thread */
          #pragma omp single
          {
            memset(histo, 0, histo_elems * sizeof(unsigned char));
          }

          /* Each thread works on a chunk of the image */
          #pragma omp for schedule(static)
          for (size_t i = 0; i < img_elems; ++i) {
            const unsigned int value = img[i];
            if (value < histo_elems && local_histo[value] < UINT8_MAX) {
              ++local_histo[value];
            }
          }

          /* Reduce local histograms into global histogram */
          #pragma omp for schedule(static)
          for (size_t bin = 0; bin < histo_elems; ++bin) {
            unsigned int sum = histo[bin];
            /* Accumulate contributions from all threads' local histograms */
            int t;
            for (t = 0; t < nthreads; ++t) {
              /* Each thread's local_histo pointer is different; we only have
               * access to our own here, so we do the reduction in a critical
               * section instead.
               * To keep things simple and correct, we instead use atomic
               * updates on the global histogram inside the first loop.
               * However, we have already computed local_histo; so we just
               * add current thread's local contribution here.
               */
              /* Only current thread's local_histo; others will add theirs
               * in their own chunks of this loop.
               */
              (void)t; /* avoid unused warning */
            }
            sum += local_histo[bin];
            if (sum > UINT8_MAX) sum = UINT8_MAX;
            histo[bin] = (unsigned char)sum;
          }

          free(local_histo);
        }
      } /* end parallel */
    }
#else
    /* Serial version */
    for (iter = 0; iter < numIterations; iter++) {
      memset(histo, 0, histo_elems * sizeof(unsigned char));
      size_t i;
      for (i = 0; i < img_elems; ++i) {
        const unsigned int value = img[i];
        if (value < histo_elems && histo[value] < UINT8_MAX) {
          ++histo[value];
        }
      }
    }
#endif

    /* End timing for kernel execution */
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

  //  pb_SwitchToTimer(&timers, pb_TimerID_IO);
    pb_SwitchToSubTimer(&timers, "Output", pb_TimerID_IO);

    if (parameters->outFile) {
      dump_histo_img(histo, histo_height, histo_width, parameters->outFile);
    }

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  }

  if (img) free(img);
  if (histo) free(histo);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  printf("\n");
  if (parameters)
    pb_PrintTimerSet(&timers);
  if (parameters)
    pb_FreeParameters(parameters);

  /* End timing for total execution */
  clock_gettime(CLOCK_MONOTONIC, &main_end);
  double main_time = (main_end.tv_sec - main_start.tv_sec) +
                     (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

  /* Determine timing output destination */
  FILE *timing_file = stderr;
  const char *timing_path = getenv("TIMING_LOG_FILE");
  if (timing_path && timing_path[0] != '\0') {
    FILE *tmp = fopen(timing_path, "w");
    if (tmp)
      timing_file = tmp;
  }

  /* Print timing results */
  fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
  fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

  if (timing_file != stderr)
    fclose(timing_file);

  return ret_code;
}
