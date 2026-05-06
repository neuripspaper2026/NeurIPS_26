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
  int exit_code = 0; /* defer returns to end for correct timing */

  printf("Base implementation of histogramming.\n");
  printf("Maintained by Nady Obeid <obeid1@ece.uiuc.edu>\n");

  parameters = pb_ReadParameters(&argc, argv);
  if (!parameters) {
    exit_code = -1;
    goto cleanup_noparams;
  }

  if (!parameters->inpFiles[0]) {
    fputs("Input file expected\n", stderr);
    exit_code = -1;
    goto cleanup_params;
  }

  int numIterations;
  if (argc >= 2) {
    numIterations = atoi(argv[1]);
    if (numIterations <= 0) {
      fputs("Number of iterations must be positive\n", stderr);
      exit_code = -1;
      goto cleanup_params;
    }
  } else {
    fputs("Expected at least one command line argument\n", stderr);
    exit_code = -1;
    goto cleanup_params;
  }

  pb_InitializeTimerSet(&timers);

  char *inputStr = "Input";
  char *outputStr = "Output";

  pb_AddSubTimer(&timers, inputStr, pb_TimerID_IO);
  pb_AddSubTimer(&timers, outputStr, pb_TimerID_IO);

  pb_SwitchToSubTimer(&timers, inputStr, pb_TimerID_IO);

  unsigned int img_width = 0, img_height = 0;
  unsigned int histo_width = 0, histo_height = 0;

  FILE* f = fopen(parameters->inpFiles[0], "rb");
  if (!f) {
    fputs("Error opening input file\n", stderr);
    exit_code = -1;
    goto cleanup_params;
  }

  size_t result = 0;

  result += fread(&img_width,    sizeof(unsigned int), 1, f);
  result += fread(&img_height,   sizeof(unsigned int), 1, f);
  result += fread(&histo_width,  sizeof(unsigned int), 1, f);
  result += fread(&histo_height, sizeof(unsigned int), 1, f);

  if (result != 4 || img_width == 0 || img_height == 0 ||
      histo_width == 0 || histo_height == 0) {
    fputs("Error reading input and output dimensions from file\n", stderr);
    fclose(f);
    exit_code = -1;
    goto cleanup_params;
  }

  /* Ensure image values do not index out of histogram range */
  if ((unsigned long long)img_width * (unsigned long long)img_height >
      (unsigned long long)SIZE_MAX / sizeof(unsigned int)) {
    fputs("Input image too large\n", stderr);
    fclose(f);
    exit_code = -1;
    goto cleanup_params;
  }

  unsigned int *img = (unsigned int*)malloc((size_t)img_width * img_height *
                                            sizeof(unsigned int));
  if (!img) {
    fputs("Failed to allocate image buffer\n", stderr);
    fclose(f);
    exit_code = -1;
    goto cleanup_params;
  }

  if ((unsigned long long)histo_width * (unsigned long long)histo_height >
      (unsigned long long)SIZE_MAX) {
    fputs("Histogram too large\n", stderr);
    fclose(f);
    free(img);
    exit_code = -1;
    goto cleanup_params;
  }

  unsigned int histo_size = histo_width * histo_height;
  unsigned char *histo = (unsigned char*)calloc((size_t)histo_size,
                                                sizeof(unsigned char));
  if (!histo) {
    fputs("Failed to allocate histogram buffer\n", stderr);
    fclose(f);
    free(img);
    exit_code = -1;
    goto cleanup_params;
  }

  pb_SwitchToSubTimer(&timers, "Input", pb_TimerID_IO);

  result = fread(img, sizeof(unsigned int),
                 (size_t)img_width * img_height, f);

  fclose(f);

  if (result != (size_t)img_width * img_height) {
    fputs("Error reading input array from file\n", stderr);
    free(img);
    free(histo);
    exit_code = -1;
    goto cleanup_params;
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  int iter;
  /* Start timing for kernel execution */
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Parallel-friendly histogram computation */
#ifdef _OPENMP
  const size_t npixels = (size_t)img_width * img_height;
  const unsigned int bins = histo_size;

  /* Precompute to avoid overflow checks in inner loop */
  for (iter = 0; iter < numIterations; iter++) {

    /* zero main histogram */
    memset(histo, 0, (size_t)bins * sizeof(unsigned char));

    int nthreads = 1;
    #pragma omp parallel
    {
      #pragma omp single
      { nthreads = omp_get_num_threads(); }
    }

    if (nthreads <= 1 || bins > 32768U) {
      /* Fallback to efficient serial kernel for large histograms or 1 thread */
      memset(histo, 0, (size_t)bins * sizeof(unsigned char));

      unsigned int local_hist[256];
      memset(local_hist, 0, sizeof(local_hist));

      unsigned int i;
      for (i = 0; i < npixels; ++i) {
        unsigned int value = img[i];
        if (value < bins) {
          ++local_hist[value];
        }
      }
      for (unsigned int b = 0; b < 256 && b < bins; ++b) {
        unsigned int count = local_hist[b];
        histo[b] = (unsigned char)(count > UINT8_MAX ? UINT8_MAX : count);
      }
    } else {
      /* Parallel histogram with per-thread private counters */
      unsigned int **thread_hists = (unsigned int**)malloc(
          (size_t)nthreads * sizeof(unsigned int*));
      if (!thread_hists) {
        /* If allocation fails, fall back to simple serial kernel */
        memset(histo, 0, (size_t)bins * sizeof(unsigned char));
        unsigned int i;
        for (i = 0; i < npixels; ++i) {
          unsigned int value = img[i];
          if (value < bins && histo[value] < UINT8_MAX) {
            ++histo[value];
          }
        }
      } else {
        int t;
        for (t = 0; t < nthreads; ++t) {
          thread_hists[t] = (unsigned int*)calloc((size_t)bins,
                                                  sizeof(unsigned int));
          if (!thread_hists[t]) {
            int k;
            for (k = 0; k < t; ++k)
              free(thread_hists[k]);
            free(thread_hists);
            memset(histo, 0, (size_t)bins * sizeof(unsigned char));
            unsigned int i;
            for (i = 0; i < npixels; ++i) {
              unsigned int value = img[i];
              if (value < bins && histo[value] < UINT8_MAX) {
                ++histo[value];
              }
            }
            goto end_parallel_iter;
          }
        }

        #pragma omp parallel
        {
          int tid = omp_get_thread_num();
          unsigned int *local = thread_hists[tid];

          #pragma omp for schedule(static)
          for (size_t i = 0; i < npixels; ++i) {
            unsigned int value = img[i];
            if (value < bins) {
              ++local[value];
            }
          }
        }

        /* combine and clamp */
        for (unsigned int b = 0; b < bins; ++b) {
          unsigned int sum = 0;
          for (int t = 0; t < nthreads; ++t) {
            sum += thread_hists[t][b];
          }
          histo[b] = (unsigned char)(sum > UINT8_MAX ? UINT8_MAX : sum);
        }

        for (t = 0; t < nthreads; ++t)
          free(thread_hists[t]);
        free(thread_hists);
      }
    }
end_parallel_iter:
    ;
  }

#else
  /* Original serial kernel with minor micro-optimizations */
  const size_t npixels = (size_t)img_width * img_height;
  const unsigned int bins = histo_size;

  for (iter = 0; iter < numIterations; iter++) {
    memset(histo, 0, (size_t)bins * sizeof(unsigned char));

    unsigned int i;
    for (i = 0; i < npixels; ++i) {
      const unsigned int value = img[i];
      if (value < bins) {
        unsigned char h = histo[value];
        if (h < UINT8_MAX) {
          histo[value] = (unsigned char)(h + 1);
        }
      }
    }
  }
#endif

  /* End timing for kernel execution */
  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                       (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

//  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  pb_SwitchToSubTimer(&timers, outputStr, pb_TimerID_IO);

  if (parameters->outFile) {
    dump_histo_img(histo, histo_height, histo_width, parameters->outFile);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  free(img);
  free(histo);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  printf("\n");
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);

  /* End timing for total execution */
cleanup_params:
  clock_gettime(CLOCK_MONOTONIC, &main_end);
  {
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
  }

cleanup_noparams:
  return exit_code;
}
