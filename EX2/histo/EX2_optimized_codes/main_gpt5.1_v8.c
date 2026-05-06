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
  int status = 0;
  if (!parameters)
    status = -1;

  if (status == 0 && !parameters->inpFiles[0]){
    fputs("Input file expected\n", stderr);
    status = -1;
  }

  int numIterations = 0;
  if (status == 0) {
    if (argc >= 2){
      numIterations = atoi(argv[1]);
    } else {
      fputs("Expected at least one command line argument\n", stderr);
      status = -1;
    }
  }

  pb_InitializeTimerSet(&timers);
  
  char *inputStr = (char *)"Input";
  char *outputStr = (char *)"Output";
  
  pb_AddSubTimer(&timers, inputStr, pb_TimerID_IO);
  pb_AddSubTimer(&timers, outputStr, pb_TimerID_IO);
  
  pb_SwitchToSubTimer(&timers, inputStr, pb_TimerID_IO);  

  unsigned int img_width = 0, img_height = 0;
  unsigned int histo_width = 0, histo_height = 0;

  FILE* f = NULL;
  unsigned int* img = NULL;
  unsigned char* histo = NULL;

  if (status == 0) {
    f = fopen(parameters->inpFiles[0],"rb");
    if (!f) {
      fputs("Error opening input file\n", stderr);
      status = -1;
    }
  }

  if (status == 0) {
    int result = 0;

    result += (int)fread(&img_width,    sizeof(unsigned int), 1, f);
    result += (int)fread(&img_height,   sizeof(unsigned int), 1, f);
    result += (int)fread(&histo_width,  sizeof(unsigned int), 1, f);
    result += (int)fread(&histo_height, sizeof(unsigned int), 1, f);

    if (result != 4){
      fputs("Error reading input and output dimensions from file\n", stderr);
      status = -1;
    }
  }

  if (status == 0) {

    size_t img_elems = (size_t)img_width * (size_t)img_height;
    size_t histo_elems = (size_t)histo_width * (size_t)histo_height;

    img = (unsigned int*) malloc (img_elems*sizeof(unsigned int));
    histo = (unsigned char*) calloc (histo_elems, sizeof(unsigned char));

    if (!img || !histo) {
      fputs("Error allocating memory\n", stderr);
      status = -1;
    }
  }

  if (status == 0) {
    pb_SwitchToSubTimer(&timers, "Input", pb_TimerID_IO);

    size_t img_elems = (size_t)img_width * (size_t)img_height;
    size_t result = fread(img, sizeof(unsigned int), img_elems, f);

    fclose(f);
    f = NULL;

    if (result != img_elems){
      fputs("Error reading input array from file\n", stderr);
      status = -1;
    }
  } else {
    if (f) {
      fclose(f);
      f = NULL;
    }
  }

  double kernel_time = 0.0;

  if (status == 0) {
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    /* Start timing for kernel execution */
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    size_t img_size = (size_t)img_width * (size_t)img_height;
    size_t histo_size = (size_t)histo_width * (size_t)histo_height;

    int iter;
    for (iter = 0; iter < numIterations; iter++){
      /* Parallel memset equivalent: zero histogram */
      memset(histo, 0, histo_size * sizeof(unsigned char));

      /* Parallel histogram using per-thread private histograms */
#ifdef _OPENMP
      int max_value = (int)histo_size;
      /* Allocate private histograms for each thread */
      int nthreads = 1;
#pragma omp parallel
      {
#ifdef _OPENMP
#pragma omp single
        {
          nthreads = omp_get_num_threads();
        }
#endif
      }

      unsigned char *local_hist = (unsigned char*)calloc((size_t)nthreads * histo_size,
                                                         sizeof(unsigned char));
      if (!local_hist) {
        /* Fallback to serial if allocation fails */
        size_t i;
        for (i = 0; i < img_size; ++i) {
          unsigned int value = img[i];
          if (value < (unsigned int)max_value && histo[value] < UINT8_MAX) {
            ++histo[value];
          }
        }
      } else {
#pragma omp parallel
        {
#ifdef _OPENMP
          int tid = omp_get_thread_num();
#else
          int tid = 0;
#endif
          unsigned char *hist_local = local_hist + (size_t)tid * histo_size;

#pragma omp for schedule(static)
          for (size_t i = 0; i < img_size; ++i) {
            unsigned int value = img[i];
            if (value < (unsigned int)max_value) {
              unsigned char *h = &hist_local[value];
              if (*h < UINT8_MAX) {
                ++(*h);
              }
            }
          }
        }

        /* Reduce private histograms into global histogram */
#pragma omp parallel for schedule(static)
        for (size_t v = 0; v < histo_size; ++v) {
          unsigned int sum = 0;
          unsigned char cap = UINT8_MAX;
          for (int t = 0; t < nthreads; ++t) {
            sum += local_hist[(size_t)t * histo_size + v];
            if (sum >= cap) {
              sum = cap;
              break;
            }
          }
          histo[v] = (unsigned char)sum;
        }

        free(local_hist);
      }
#else
      /* Serial fallback */
      size_t i;
      for (i = 0; i < img_size; ++i) {
        const unsigned int value = img[i];
        if (value < histo_size && histo[value] < UINT8_MAX) {
          ++histo[value];
        }
      }
#endif
    }

    /* End timing for kernel execution */
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
  }

  if (status == 0) {
    pb_SwitchToSubTimer(&timers, outputStr, pb_TimerID_IO);

    if (parameters->outFile) {
      dump_histo_img(histo, histo_height, histo_width, parameters->outFile);
    }

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  }

  free(img);
  free(histo);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  printf("\n");
  pb_PrintTimerSet(&timers);
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

  return status;
}
