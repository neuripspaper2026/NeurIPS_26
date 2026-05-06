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

  int exit_code = 0;
  unsigned int *img = NULL;
  unsigned char *histo = NULL;
  FILE *f = NULL;
  FILE *timing_file = stderr;

  printf("Base implementation of histogramming.\n");
  printf("Maintained by Nady Obeid <obeid1@ece.uiuc.edu>\n");

  parameters = pb_ReadParameters(&argc, argv);
  if (!parameters) {
    exit_code = -1;
    goto cleanup_final;
  }

  if (!parameters->inpFiles[0]) {
    fputs("Input file expected\n", stderr);
    exit_code = -1;
    goto cleanup_parameters;
  }

  int numIterations;
  if (argc >= 2) {
    numIterations = atoi(argv[1]);
  } else {
    fputs("Expected at least one command line argument\n", stderr);
    exit_code = -1;
    goto cleanup_parameters;
  }

  pb_InitializeTimerSet(&timers);
  
  char *inputStr = "Input";
  char *outputStr = "Output";
  
  pb_AddSubTimer(&timers, inputStr, pb_TimerID_IO);
  pb_AddSubTimer(&timers, outputStr, pb_TimerID_IO);
  
  pb_SwitchToSubTimer(&timers, inputStr, pb_TimerID_IO);  

  unsigned int img_width = 0, img_height = 0;
  unsigned int histo_width = 0, histo_height = 0;

  f = fopen(parameters->inpFiles[0], "rb");
  if (!f) {
    fputs("Error opening input file\n", stderr);
    exit_code = -1;
    goto cleanup_parameters;
  }

  int result = 0;

  result += (int)fread(&img_width,    sizeof(unsigned int), 1, f);
  result += (int)fread(&img_height,   sizeof(unsigned int), 1, f);
  result += (int)fread(&histo_width,  sizeof(unsigned int), 1, f);
  result += (int)fread(&histo_height, sizeof(unsigned int), 1, f);

  if (result != 4) {
    fputs("Error reading input and output dimensions from file\n", stderr);
    exit_code = -1;
    goto cleanup_file;
  }

  size_t num_pixels = (size_t)img_width * (size_t)img_height;
  size_t histo_size = (size_t)histo_width * (size_t)histo_height;

  img = (unsigned int*)malloc(num_pixels * sizeof(unsigned int));
  if (!img) {
    fputs("Error allocating memory for image\n", stderr);
    exit_code = -1;
    goto cleanup_file;
  }

  histo = (unsigned char*)calloc(histo_size, sizeof(unsigned char));
  if (!histo) {
    fputs("Error allocating memory for histogram\n", stderr);
    exit_code = -1;
    goto cleanup_img;
  }
  
  pb_SwitchToSubTimer(&timers, "Input", pb_TimerID_IO);

  result = (int)fread(img, sizeof(unsigned int), num_pixels, f);

  if ((size_t)result != num_pixels) {
    fputs("Error reading input array from file\n", stderr);
    exit_code = -1;
    goto cleanup_all;
  }

  fclose(f);
  f = NULL;

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  int iter;
  /* Start timing for kernel execution */
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  for (iter = 0; iter < numIterations; iter++) {
    memset(histo, 0, histo_size * sizeof(unsigned char));

    /* Parallel histogram computation */
#ifdef _OPENMP
    int nthreads = 1;
#pragma omp parallel
    {
#ifdef _OPENMP
      nthreads = omp_get_num_threads();
#endif
    }

    /* Allocate per-thread histograms to avoid contention */
    unsigned char **thread_histos = (unsigned char **)malloc((size_t)nthreads * sizeof(unsigned char *));
    if (!thread_histos) {
      /* Fallback to serial if allocation fails */
      size_t i;
      for (i = 0; i < num_pixels; ++i) {
        const unsigned int value = img[i];
        if (value < histo_size && histo[value] < UINT8_MAX) {
          ++histo[value];
        }
      }
    } else {
      int t;
      for (t = 0; t < nthreads; ++t) {
        thread_histos[t] = (unsigned char *)calloc(histo_size, sizeof(unsigned char));
        if (!thread_histos[t]) {
          /* If any allocation fails, free what we have and fallback to serial */
          int k;
          for (k = 0; k < t; ++k) {
            free(thread_histos[k]);
          }
          free(thread_histos);
          size_t i;
          for (i = 0; i < num_pixels; ++i) {
            const unsigned int value = img[i];
            if (value < histo_size && histo[value] < UINT8_MAX) {
              ++histo[value];
            }
          }
          goto skip_parallel_merge;
        }
      }

#pragma omp parallel
      {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        unsigned char *local_histo = thread_histos[tid];

#pragma omp for schedule(static)
        for (long long i = 0; i < (long long)num_pixels; ++i) {
          const unsigned int value = img[i];
          if (value < histo_size) {
            unsigned char hv = local_histo[value];
            if (hv < UINT8_MAX) {
              local_histo[value] = (unsigned char)(hv + 1);
            }
          }
        }
      }

      /* Merge per-thread histograms */
#pragma omp parallel for schedule(static)
      for (long long v = 0; v < (long long)histo_size; ++v) {
        unsigned int sum = 0;
        int t;
        for (t = 0; t < nthreads; ++t) {
          sum += thread_histos[t][v];
          if (sum >= UINT8_MAX) {
            sum = UINT8_MAX;
            break;
          }
        }
        histo[v] = (unsigned char)sum;
      }

      /* Free temporary per-thread histograms */
      int t2;
      for (t2 = 0; t2 < nthreads; ++t2) {
        free(thread_histos[t2]);
      }
      free(thread_histos);
    }

skip_parallel_merge:
#else
    /* Serial fallback if OpenMP is not available */
    size_t i;
    for (i = 0; i < num_pixels; ++i) {
      const unsigned int value = img[i];
      if (value < histo_size && histo[value] < UINT8_MAX) {
        ++histo[value];
      }
    }
#endif
  }

  /* End timing for kernel execution */
  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                       (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

//  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  pb_SwitchToSubTimer(&timers, outputStr, pb_TimerID_IO);

  if (parameters->outFile && histo) {
    dump_histo_img(histo, histo_height, histo_width, parameters->outFile);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

cleanup_all:
  if (img) {
    free(img);
    img = NULL;
  }
  if (histo) {
    free(histo);
    histo = NULL;
  }

cleanup_img:
  /* img already freed above if allocated */

cleanup_file:
  if (f) {
    fclose(f);
    f = NULL;
  }

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  printf("\n");
  pb_PrintTimerSet(&timers);

cleanup_parameters:
  if (parameters) {
    pb_FreeParameters(parameters);
    parameters = NULL;
  }

  /* End timing for total execution */
  clock_gettime(CLOCK_MONOTONIC, &main_end);
  double main_time = (main_end.tv_sec - main_start.tv_sec) +
                     (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

  /* Determine timing output destination */
  {
    const char *timing_path = getenv("TIMING_LOG_FILE");
    if (timing_path && timing_path[0] != '\0') {
      FILE *tmp = fopen(timing_path, "w");
      if (tmp)
        timing_file = tmp;
    }
  }

  /* Print timing results */
  fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
  fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

  if (timing_file != stderr)
    fclose(timing_file);

cleanup_final:
  return exit_code;
}
