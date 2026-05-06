
/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "file.h"
#include "common.h"
#include "kernels.h"


static int read_data(float *A0, int nx,int ny,int nz,FILE *fp) 
{	
	int s=0;
        int i, j, k;
	for(i=0;i<nz;i++)
	{
		for(j=0;j<ny;j++)
		{
			for(k=0;k<nx;k++)
			{
                                fread(A0+s,sizeof(float),1,fp);
				s++;
			}
		}
	}
	return 0;
}

int main(int argc, char** argv) {
	struct pb_TimerSet timers;
	struct pb_Parameters *parameters;
	struct timespec main_start, main_end;
	clock_gettime(CLOCK_MONOTONIC, &main_start);
	

	
	printf("CPU-based 7 points stencil codes****\n");
	printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and I-Jui Sung<sung10@illinois.edu>\n");
	printf("This version maintained by Chris Rodrigues  ***********\n");
	parameters = pb_ReadParameters(&argc, argv);

	pb_InitializeTimerSet(&timers);
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	//declaration
	int nx,ny,nz;
	int size;
    int iteration;
	float c0=1.0f/6.0f;
	float c1=1.0f/6.0f/6.0f;

	if (argc<5) 
    {
      printf("Usage: probe nx ny nz tx ty t\n"
	     "nx: the grid size x\n"
	     "ny: the grid size y\n"
	     "nz: the grid size z\n"
		  "t: the iteration time\n");
      return -1;
    }

	nx = atoi(argv[1]);
	if (nx<1)
		return -1;
	ny = atoi(argv[2]);
	if (ny<1)
		return -1;
	nz = atoi(argv[3]);
	if (nz<1)
		return -1;
	iteration = atoi(argv[4]);
	if(iteration<1)
		return -1;

	
	//host data
	float *h_A0;
	float *h_Anext;

	size=nx*ny*nz;
	
	h_A0=(float*)malloc(sizeof(float)*size);
	h_Anext=(float*)malloc(sizeof(float)*size);
  FILE *fp = fopen(parameters->inpFiles[0], "rb");
	read_data(h_A0, nx,ny,nz,fp);
  fclose(fp);
  memcpy (h_Anext,h_A0 ,sizeof(float)*size);

	struct timespec kernel_start, kernel_end;
	clock_gettime(CLOCK_MONOTONIC, &kernel_start);
	int t;
	for(t=0;t<iteration;t++)
	{
		cpu_stencil(c0,c1, h_A0, h_Anext, nx, ny,  nz);
    float *temp=h_A0;
    h_A0 = h_Anext;
    h_Anext = temp;

	}
	clock_gettime(CLOCK_MONOTONIC, &kernel_end);
	double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
	                     (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

  float *temp=h_A0;
  h_A0 = h_Anext;
  h_Anext = temp;

 
	if (parameters->outFile) {
		 pb_SwitchToTimer(&timers, pb_TimerID_IO);
		outputData(parameters->outFile,h_Anext,nx,ny,nz);
		
	}
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
		
	free (h_A0);
	free (h_Anext);
	pb_SwitchToTimer(&timers, pb_TimerID_NONE);

	pb_PrintTimerSet(&timers);
	pb_FreeParameters(parameters);

	clock_gettime(CLOCK_MONOTONIC, &main_end);
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
