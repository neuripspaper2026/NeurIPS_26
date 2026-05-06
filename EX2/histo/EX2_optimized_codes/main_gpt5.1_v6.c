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

  int exit_status = 0; /* track exit status to avoid early returns */

  printf("Base implementation of histogramming.\n");
  printf("Maintained by Nady Obeid <obeid1@ece.uiuc.edu>\n");

  parameters = pb_ReadParameters(&argc, argv);
  if (!parameters)
    exit_status = -1;

  unsigned int *img = NULL;
  unsigned char *histo = NULL;
  unsigned int img_width = 0, img_height = 0;
  unsigned int histo_width = 0, histo_height = 0;
  int numIterations = 0;

  FILE* f = NULL;

  if (exit_status == 0) {
    if(!parameters->inpFiles[0]){
      fputs("Input file expected\n", stderr);
      exit_status = -1;
    }
  }

  if (exit_status == 0) {
    if (argc >= 2){
      numIterations = atoi(argv[1]);
    } else {
      fputs("Expected at least one command line argument\n", stderr);
      exit_status = -1;
    }
  }

  pb_InitializeTimerSet(&timers);
  
  char *inputStr = (char *)"Input";
  char *outputStr = (char *)"Output";
  
  pb_AddSubTimer(&timers, inputStr, pb_TimerID_IO);
  pb_AddSubTimer(&timers, outputStr, pb_TimerID_IO);
  
  pb_SwitchToSubTimer(&timers, inputStr, pb_TimerID_IO);  

  int result = 0;

  if (exit_status == 0) {
    f = fopen(parameters->inpFiles[0],"rb");
    if (!f) {
      fputs("Error opening input file\n", stderr);
      exit_status = -1;
    }
  }

  if (exit_status == 0) {
    result += (int)fread(&img_width,    sizeof(unsigned int), 1, f);
    result += (int)fread(&img_height,   sizeof(unsigned int), 1, f);
    result += (int)fread(&histo_width,  sizeof(unsigned int), 1, f);
    result += (int)fread(&histo_height, sizeof(unsigned int), 1, f);

    if (result != 4){
      fputs("Error reading input and output dimensions from file\n", stderr);
      exit_status = -1;
    }
  }

  if (exit_status == 0) {
    size_t img_elems = (size_t)img_width * (size_t)img_height;
    size_t histo_elems = (size_t)histo_width * (size_t)histo_height;

    img = (unsigned int*) malloc (img_elems * sizeof(unsigned int));
    histo = (unsigned char*) calloc (histo_elems, sizeof(unsigned char));

    if (!img || !histo) {
      fputs("Error allocating memory\n", stderr);
      exit_status = -1;
    }
  }

  pb_SwitchToSubTimer(&timers, "Input", pb_TimerID_IO);

  if (exit_status == 0) {
    size_t img_elems = (size_t)img_width * (size_t)img_height;
    result = (int)fread(img, sizeof(unsigned int), img_elems, f);

    fclose(f);
    f = NULL;

    if ((size_t)result != img_elems){
      fputs("Error reading input array from file\n", stderr);
      exit_status = -1;
    }
  } else {
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

  if (exit_status == 0) {
    const size_t img_size = (size_t)img_width * (size_t)img_height;
    const size_t histo_size = (size_t)histo_width * (size_t)histo_height;

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    if (num_threads < 1) num_threads = 1;
    unsigned char **local_histos = (unsigned char **)malloc((size_t)num_threads * sizeof(unsigned char *));
    if (!local_histos) {
      /* Fallback to serial if allocation fails */
      num_threads = 1;
    } else {
      size_t t;
      for (t = 0; t < (size_t)num_threads; ++t) {
        local_histos[t] = (unsigned char *)calloc(histo_size, sizeof(unsigned char));
        if (!local_histos[t]) {
          size_t k;
          for (k = 0; k < t; ++k) {
            free(local_histos[k]);
          }
          free(local_histos);
          local_histos = NULL;
          num_threads = 1;
          break;
        }
      }
    }

    if (num_threads > 1 && local_histos != NULL) {
      for (iter = 0; iter < numIterations; iter++){
        memset(histo, 0, histo_size * sizeof(unsigned char));
        int i;
#pragma omp parallel
        {
          int tid = omp_get_thread_num();
          unsigned char *lh = local_histos[tid];
#pragma omp for
          for (i = 0; i < (int)img_size; ++i) {
            const unsigned int value = img[i];
            if (value < histo_size) {
              if (lh[value] < UINT8_MAX) {
                ++lh[value];
              }
            }
          }
        }
        size_t h;
        for (h = 0; h < histo_size; ++h) {
          unsigned int sum = 0;
          int t;
          for (t = 0; t < num_threads; ++t) {
            sum += local_histos[t][h];
          }
          histo[h] = (unsigned char)(sum > UINT8_MAX ? UINT8_MAX : sum);
        }
      }
      int t;
      for (t = 0; t < num_threads; ++t) {
        free(local_histos[t]);
      }
      free(local_histos);
    } else
#endif
    {
      for (iter = 0; iter < numIterations; iter++){
        memset(histo, 0, histo_size * sizeof(unsigned char));
        size_t i;
        for (i = 0; i < img_size; ++i) {
          const unsigned int value = img[i];
          if (value < histo_size && histo[value] < UINT8_MAX) {
            ++histo[value];
          }
        }
      }
    }
  }

  /* End timing for kernel execution */
  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                       (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

//  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  pb_SwitchToSubTimer(&timers, outputStr, pb_TimerID_IO);

  if (exit_status == 0 && parameters->outFile) {
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

  return exit_status;
}
