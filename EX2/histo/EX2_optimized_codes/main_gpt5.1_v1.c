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

  int exit_code = 0;
  unsigned int *img = NULL;
  unsigned char *histo = NULL;
  FILE *f = NULL;
  int numIterations = 0;
  unsigned int img_width = 0, img_height = 0;
  unsigned int histo_width = 0, histo_height = 0;
  int result = 0;

  if (!parameters) {
    exit_code = -1;
  }

  if (exit_code == 0 && !parameters->inpFiles[0]) {
    fputs("Input file expected\n", stderr);
    exit_code = -1;
  }

  if (exit_code == 0) {
    if (argc >= 2) {
      numIterations = atoi(argv[1]);
      if (numIterations <= 0) {
        fputs("Number of iterations must be positive\n", stderr);
        exit_code = -1;
      }
    } else {
      fputs("Expected at least one command line argument\n", stderr);
      exit_code = -1;
    }
  }

  pb_InitializeTimerSet(&timers);
  
  char *inputStr = "Input";
  char *outputStr = "Output";
  
  pb_AddSubTimer(&timers, inputStr, pb_TimerID_IO);
  pb_AddSubTimer(&timers, outputStr, pb_TimerID_IO);
  
  pb_SwitchToSubTimer(&timers, inputStr, pb_TimerID_IO);  

  if (exit_code == 0) {
    f = fopen(parameters->inpFiles[0],"rb");
    if (!f) {
      fputs("Error opening input file\n", stderr);
      exit_code = -1;
    }
  }

  if (exit_code == 0) {
    result  = (int)fread(&img_width,    sizeof(unsigned int), 1, f);
    result += (int)fread(&img_height,   sizeof(unsigned int), 1, f);
    result += (int)fread(&histo_width,  sizeof(unsigned int), 1, f);
    result += (int)fread(&histo_height, sizeof(unsigned int), 1, f);

    if (result != 4) {
      fputs("Error reading input and output dimensions from file\n", stderr);
      exit_code = -1;
    }
  }

  if (exit_code == 0) {
    if (histo_width * histo_height == 0 || img_width * img_height == 0) {
      fputs("Invalid image or histogram dimensions\n", stderr);
      exit_code = -1;
    }
  }

  if (exit_code == 0) {
    img = (unsigned int*) malloc ((size_t)img_width * (size_t)img_height * sizeof(unsigned int));
    histo = (unsigned char*) calloc ((size_t)histo_width * (size_t)histo_height, sizeof(unsigned char));
    if (!img || !histo) {
      fputs("Error allocating memory\n", stderr);
      exit_code = -1;
    }
  }

  pb_SwitchToSubTimer(&timers, "Input", pb_TimerID_IO);

  if (exit_code == 0) {
    size_t elems = (size_t)img_width * (size_t)img_height;
    result = (int)fread(img, sizeof(unsigned int), elems, f);

    if (result != (int)elems) {
      fputs("Error reading input array from file\n", stderr);
      exit_code = -1;
    }
  }

  if (f) {
    fclose(f);
    f = NULL;
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  int iter;
  /* Start timing for kernel execution */
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  if (exit_code == 0) {
    const size_t img_size = (size_t)img_width * (size_t)img_height;
    const size_t histo_size = (size_t)histo_width * (size_t)histo_height;

    for (iter = 0; iter < numIterations; iter++){
      memset(histo, 0, histo_size * sizeof(unsigned char));

      /* Parallel histogram with per-thread private histograms to avoid contention */
#ifdef _OPENMP
      int nthreads = 1;
#pragma omp parallel
      {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
        int local_nthreads = omp_get_num_threads();
#pragma omp single
        {
          nthreads = local_nthreads;
        }
        size_t local_histo_size = histo_size;
        /* Allocate per-thread private histogram */
        unsigned int *local_hist = (unsigned int*)calloc(local_histo_size, sizeof(unsigned int));
        if (!local_hist) {
          /* Fallback: if allocation fails, do serial work in single thread */
#pragma omp single
          {
            size_t i;
            for (i = 0; i < img_size; ++i) {
              const unsigned int value = img[i];
              if (value < histo_size && histo[value] < UINT8_MAX) {
                ++histo[value];
              }
            }
          }
        } else {
#pragma omp for schedule(static)
          for (size_t i = 0; i < img_size; ++i) {
            const unsigned int value = img[i];
            if (value < histo_size) {
              ++local_hist[value];
            }
          }

          /* Merge local histogram into global with saturation */
#pragma omp for schedule(static)
          for (size_t i = 0; i < histo_size; ++i) {
            unsigned int val = local_hist[i];
            if (val != 0) {
              unsigned int sum = histo[i] + val;
              histo[i] = (unsigned char)(sum > UINT8_MAX ? UINT8_MAX : sum);
            }
          }

          free(local_hist);
        }
#else
        (void)tid; /* avoid unused warning if compiled without OpenMP */
#endif
      }
#else
      {
        size_t i;
        for (i = 0; i < img_size; ++i) {
          const unsigned int value = img[i];
          if (value < histo_size && histo[value] < UINT8_MAX) {
            ++histo[value];
          }
        }
      }
#endif
    }
  }

  /* End timing for kernel execution */
  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                       (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

//  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  pb_SwitchToSubTimer(&timers, outputStr, pb_TimerID_IO);

  if (exit_code == 0 && parameters->outFile) {
    dump_histo_img(histo, histo_height, histo_width, parameters->outFile);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  if (img) {
    free(img);
    img = NULL;
  }
  if (histo) {
    free(histo);
    histo = NULL;
  }

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  printf("\n");
  pb_PrintTimerSet(&timers);
  if (parameters) {
    pb_FreeParameters(parameters);
    parameters = NULL;
  }

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

  return exit_code;
}
