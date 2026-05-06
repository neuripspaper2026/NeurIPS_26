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

  int exit_code = 0; /* defer returns to preserve timing */

  if (!parameters) {
    exit_code = -1;
  }

  int numIterations = 0;
  unsigned int img_width = 0, img_height = 0;
  unsigned int histo_width = 0, histo_height = 0;
  unsigned int *img = NULL;
  unsigned char *histo = NULL;
  FILE *f = NULL;

  if (!exit_code) {
    if (!parameters->inpFiles[0]) {
      fputs("Input file expected\n", stderr);
      exit_code = -1;
    }
  }

  if (!exit_code) {
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
  
  char *inputStr = (char *)"Input";
  char *outputStr = (char *)"Output";
  
  pb_AddSubTimer(&timers, inputStr, pb_TimerID_IO);
  pb_AddSubTimer(&timers, outputStr, pb_TimerID_IO);
  
  pb_SwitchToSubTimer(&timers, inputStr, pb_TimerID_IO);

  int result = 0;

  if (!exit_code) {
    f = fopen(parameters->inpFiles[0], "rb");
    if (!f) {
      fputs("Error opening input file\n", stderr);
      exit_code = -1;
    }
  }

  if (!exit_code) {
    result  = (int)fread(&img_width,    sizeof(unsigned int), 1, f);
    result += (int)fread(&img_height,   sizeof(unsigned int), 1, f);
    result += (int)fread(&histo_width,  sizeof(unsigned int), 1, f);
    result += (int)fread(&histo_height, sizeof(unsigned int), 1, f);

    if (result != 4) {
      fputs("Error reading input and output dimensions from file\n", stderr);
      exit_code = -1;
    }
  }

  if (!exit_code) {
    size_t img_elems = (size_t)img_width * (size_t)img_height;
    size_t histo_elems = (size_t)histo_width * (size_t)histo_height;

    img = (unsigned int*)malloc(img_elems * sizeof(unsigned int));
    histo = (unsigned char*)calloc(histo_elems, sizeof(unsigned char));

    if (!img || !histo) {
      fputs("Error allocating memory\n", stderr);
      exit_code = -1;
    }

    pb_SwitchToSubTimer(&timers, "Input", pb_TimerID_IO);

    if (!exit_code) {
      result = (int)fread(img, sizeof(unsigned int), img_elems, f);

      if ((size_t)result != img_elems) {
        fputs("Error reading input array from file\n", stderr);
        exit_code = -1;
      }
    }

    if (f) {
      fclose(f);
      f = NULL;
    }
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  int iter;
  /* Start timing for kernel execution */
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  if (!exit_code) {
    const size_t img_elems = (size_t)img_width * (size_t)img_height;
    const size_t histo_elems = (size_t)histo_width * (size_t)histo_height;

    /* Parallelized outer iteration loop with privatized histograms */
#ifdef _OPENMP
    #pragma omp parallel
    {
      unsigned int tid = 0;
      int nthreads = 1;
      #ifdef _OPENMP
      tid = (unsigned int)omp_get_thread_num();
      nthreads = omp_get_num_threads();
      #endif

      size_t chunk = img_elems / (size_t)nthreads;
      size_t start = (size_t)tid * chunk;
      size_t end = (tid == (unsigned int)(nthreads - 1)) ? img_elems : start + chunk;

      size_t private_size = histo_elems * sizeof(unsigned int);
      unsigned int *private_histo = (unsigned int *)calloc(histo_elems, sizeof(unsigned int));

      if (!private_histo) {
        /* If allocation fails, fall back to serial within this thread */
        if (tid == 0) {
          /* single-threaded fallback of original loop */
          for (iter = 0; iter < numIterations; iter++) {
            memset(histo, 0, histo_elems * sizeof(unsigned char));
            size_t i;
            for (i = 0; i < img_elems; ++i) {
              const unsigned int value = img[i];
              if (histo[value] < UINT8_MAX) {
                ++histo[value];
              }
            }
          }
        }
      } else {
        /* First compute per-thread histograms across all iterations */
        for (iter = 0; iter < numIterations; ++iter) {
          size_t i;
          for (i = start; i < end; ++i) {
            const unsigned int value = img[i];
            if (value < histo_elems && private_histo[value] < UINT8_MAX) {
              ++private_histo[value];
            }
          }
        }

        /* Merge private histograms into the global one (single merge section) */
        #pragma omp barrier
        #pragma omp for
        for (size_t h = 0; h < histo_elems; ++h) {
          unsigned int total = 0;
          int t;
          for (t = 0; t < nthreads; ++t) {
            /* reinterpret thread-private histograms laid out consecutively */
            unsigned int *base = NULL;
            /* recompute base pointer for each thread */
            #ifdef _OPENMP
            base = (unsigned int *)((char *)private_histo + (size_t)t * private_size);
            #else
            base = private_histo;
            #endif
            total += base[h];
            if (total >= UINT8_MAX) {
              total = UINT8_MAX;
              break;
            }
          }
          histo[h] = (unsigned char)total;
        }

        free(private_histo);
      }
    }
#else
    /* Serial version as fallback when OpenMP is not available */
    for (iter = 0; iter < numIterations; iter++) {
      memset(histo, 0, histo_elems * sizeof(unsigned char));
      size_t i;
      for (i = 0; i < img_elems; ++i) {
        const unsigned int value = img[i];
        if (histo[value] < UINT8_MAX) {
          ++histo[value];
        }
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

  if (!exit_code && parameters->outFile) {
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
