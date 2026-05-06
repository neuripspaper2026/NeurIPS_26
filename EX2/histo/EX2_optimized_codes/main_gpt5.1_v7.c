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
  if (!parameters)
    exit_code = -1;

  int have_input = 1;
  if (exit_code == 0 && !parameters->inpFiles[0]) {
    fputs("Input file expected\n", stderr);
    exit_code = -1;
    have_input = 0;
  }

  int numIterations = 0;
  if (exit_code == 0) {
    if (argc >= 2){
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

  unsigned int img_width = 0, img_height = 0;
  unsigned int histo_width = 0, histo_height = 0;
  unsigned int* img = NULL;
  unsigned char* histo = NULL;

  if (exit_code == 0 && have_input) {
    FILE* f = fopen(parameters->inpFiles[0],"rb");
    if (!f) {
      fputs("Error opening input file\n", stderr);
      exit_code = -1;
    } else {
      size_t result = 0;

      result += fread(&img_width,    sizeof(unsigned int), 1, f);
      result += fread(&img_height,   sizeof(unsigned int), 1, f);
      result += fread(&histo_width,  sizeof(unsigned int), 1, f);
      result += fread(&histo_height, sizeof(unsigned int), 1, f);

      if (result != 4){
        fputs("Error reading input and output dimensions from file\n", stderr);
        exit_code = -1;
      } else {
        /* Validate dimensions to avoid overflow and invalid memory access */
        if (histo_width == 0 || histo_height == 0 ||
            img_width == 0 || img_height == 0) {
          fputs("Invalid image or histogram dimensions\n", stderr);
          exit_code = -1;
        } else {
          /* Check that histogram is large enough for all possible pixel values */
          unsigned long long histo_size = (unsigned long long)histo_width *
                                          (unsigned long long)histo_height;
          if (histo_size <= UINT8_MAX) {
            fputs("Histogram size must be greater than 255\n", stderr);
            exit_code = -1;
          } else {
            unsigned long long img_size = (unsigned long long)img_width *
                                          (unsigned long long)img_height;
            if (img_size == 0) {
              fputs("Invalid image size\n", stderr);
              exit_code = -1;
            } else {
              if (img_size > (SIZE_MAX / sizeof(unsigned int)) ||
                  histo_size > (SIZE_MAX / sizeof(unsigned char))) {
                fputs("Requested memory size is too large\n", stderr);
                exit_code = -1;
              } else {
                img = (unsigned int*) malloc ((size_t)img_size * sizeof(unsigned int));
                histo = (unsigned char*) calloc ((size_t)histo_size, sizeof(unsigned char));
                if (!img || !histo) {
                  fputs("Memory allocation failure\n", stderr);
                  exit_code = -1;
                }
              }
            }
          }
        }
      }

      pb_SwitchToSubTimer(&timers, "Input", pb_TimerID_IO);

      if (exit_code == 0) {
        size_t img_elems = (size_t)img_width * (size_t)img_height;
        result = fread(img, sizeof(unsigned int), img_elems, f);

        if (result != img_elems){
          fputs("Error reading input array from file\n", stderr);
          exit_code = -1;
        }
      }

      fclose(f);
    }
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  int iter;
  /* Start timing for kernel execution */
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

#ifdef _OPENMP
  if (exit_code == 0) {
#pragma omp parallel
  {
    int num_threads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    size_t img_size = (size_t)img_width * (size_t)img_height;
    size_t histo_size = (size_t)histo_width * (size_t)histo_height;

    /* Allocate per-thread histogram once per thread */
    unsigned char *local_histo = (unsigned char*) calloc(histo_size, sizeof(unsigned char));

    for (iter = 0; iter < numIterations; iter++){
#pragma omp for
      for (size_t i = 0; i < img_size; ++i) {
        unsigned int value = img[i];
        if (value < histo_size) {
          unsigned char *bin = &local_histo[value];
          if (*bin < UINT8_MAX) {
            ++(*bin);
          }
        }
      }

#pragma omp barrier

#pragma omp for
      for (size_t i = 0; i < histo_size; ++i) {
        unsigned int sum = 0;
        for (int t = 0; t < num_threads; ++t) {
          /* Each thread adds only its own contribution; threads are partitioned by i */
          /* local_histo is per-thread, so only current thread's histogram is used here */
        }
      }

#pragma omp single
      memset(histo, 0, histo_size * sizeof(unsigned char));

#pragma omp barrier

#pragma omp for
      for (size_t i = 0; i < histo_size; ++i) {
        unsigned int sum = 0;
        /* Accumulate from each thread's local histogram using reduction-style loop */
        /* To avoid additional storage for all thread histograms, we fold iterations:
           each outer iteration uses current local_histo and atomically adds to global. */
        unsigned char val = local_histo[i];
        if (val) {
#pragma omp atomic update
          histo[i] += val;
        }
      }
    }

    free(local_histo);
  }
  }
#else
  if (exit_code == 0) {
    size_t img_size = (size_t)img_width * (size_t)img_height;
    size_t histo_size = (size_t)histo_width * (size_t)histo_height;
    for (iter = 0; iter < numIterations; iter++){
      memset(histo,0,histo_size*sizeof(unsigned char));
      for (size_t i = 0; i < img_size; ++i) {
        const unsigned int value = img[i];
        if (value < histo_size) {
          unsigned char *bin = &histo[value];
          if (*bin < UINT8_MAX) {
            ++(*bin);
          }
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

  if (exit_code == 0 && parameters->outFile) {
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
