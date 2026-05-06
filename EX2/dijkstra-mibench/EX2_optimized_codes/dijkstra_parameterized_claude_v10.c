#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define MAX_NODES                          10000
#define NONE                               9999

struct _NODE
{
  int iDist;
  int iPrev;
};
typedef struct _NODE NODE;

struct _QITEM
{
  int iNode;
  int iDist;
  int iPrev;
  struct _QITEM *qNext;
};
typedef struct _QITEM QITEM;

QITEM *qHead = NULL;

// Dynamic adjacency matrix
int **AdjMatrix = NULL;
int NUM_NODES = 0;

int g_qCount = 0;
NODE *rgnNodes = NULL;
int ch;
int iPrev, iNode;
int i, iCost, iDist;

static double dijkstra_kernel_time_acc = 0.0;

void reset_dijkstra_kernel_time(void) { dijkstra_kernel_time_acc = 0.0; }
double get_dijkstra_kernel_time(void) { return dijkstra_kernel_time_acc; }


void print_path (NODE *rgnNodes, int chNode)
{
  if (rgnNodes[chNode].iPrev != NONE)
    {
      print_path(rgnNodes, rgnNodes[chNode].iPrev);
    }
  printf (" %d", chNode);
  fflush(stdout);
}


void enqueue (int iNode, int iDist, int iPrev)
{
  QITEM *qNew = (QITEM *) malloc(sizeof(QITEM));
  QITEM *qLast = qHead;
  
  if (!qNew) 
    {
      fprintf(stderr, "Out of memory.\n");
      exit(1);
    }
  qNew->iNode = iNode;
  qNew->iDist = iDist;
  qNew->iPrev = iPrev;
  qNew->qNext = NULL;
  
  if (!qLast) 
    {
      qHead = qNew;
    }
  else
    {
      while (qLast->qNext) qLast = qLast->qNext;
      qLast->qNext = qNew;
    }
  g_qCount++;
}


void dequeue (int *piNode, int *piDist, int *piPrev)
{
  QITEM *qKill = qHead;
  
  if (qHead)
    {
      *piNode = qHead->iNode;
      *piDist = qHead->iDist;
      *piPrev = qHead->iPrev;
      qHead = qHead->qNext;
      free(qKill);
      g_qCount--;
    }
}


int qcount (void)
{
  return(g_qCount);
}

int dijkstra(int chStart, int chEnd) 
{
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);
  
  int local_ch, local_i, local_iNode, local_iDist, local_iPrev, local_iCost;
  
  for (local_ch = 0; local_ch < NUM_NODES; local_ch++)
    {
      rgnNodes[local_ch].iDist = NONE;
      rgnNodes[local_ch].iPrev = NONE;
    }

  if (chStart == chEnd) 
    {
      printf("Shortest path is 0 in cost. Just stay where you are.\n");
    }
  else
    {
      rgnNodes[chStart].iDist = 0;
      rgnNodes[chStart].iPrev = NONE;
      
      enqueue (chStart, 0, NONE);
      
     while (qcount() > 0)
	{
	  dequeue (&local_iNode, &local_iDist, &local_iPrev);
	  for (local_i = 0; local_i < NUM_NODES; local_i++)
	    {
	      if ((local_iCost = AdjMatrix[local_iNode][local_i]) != NONE)
		{
		  if ((NONE == rgnNodes[local_i].iDist) || 
		      (rgnNodes[local_i].iDist > (local_iCost + local_iDist)))
		    {
		      rgnNodes[local_i].iDist = local_iDist + local_iCost;
		      rgnNodes[local_i].iPrev = local_iNode;
		      enqueue (local_i, local_iDist + local_iCost, local_iNode);
		    }
		}
	    }
	}
      
      printf("Shortest path is %d in cost. ", rgnNodes[chEnd].iDist);
      printf("Path is: ");
      print_path(rgnNodes, chEnd);
      printf("\n");
    }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  dijkstra_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                              (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
  return 0;
}

// Count number of integers in file to determine matrix size
int detect_matrix_size(const char *filename) {
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open file %s\n", filename);
    return -1;
  }
  
  int count = 0;
  int val;
  while (fscanf(fp, "%d", &val) == 1) {
    count++;
  }
  fclose(fp);
  
  // Matrix size is sqrt(count)
  int n = 0;
  while (n * n < count) n++;
  if (n * n != count) {
    fprintf(stderr, "Error: File does not contain a square matrix (%d values)\n", count);
    return -1;
  }
  
  return n;
}

int main(int argc, char *argv[]) {
  int i,j,k;
  FILE *fp = NULL;
  int ret = 0;
  int adj_allocated = 0;
  int nodes_allocated = 0;
  struct timespec main_start, main_end;
  FILE *timing_file = stderr;
  const char *timing_path = getenv("TIMING_LOG_FILE");

  clock_gettime(CLOCK_MONOTONIC, &main_start);
  if (timing_path && timing_path[0] != '\0') {
    FILE *tmp = fopen(timing_path, "w");
    if (tmp) {
      timing_file = tmp;
    }
  }
  reset_dijkstra_kernel_time();
  
  if (argc<2) {
    fprintf(stderr, "Usage: dijkstra <filename>\n");
    fprintf(stderr, "Supports dynamic matrix sizes up to %d×%d.\n", MAX_NODES, MAX_NODES);
    ret = 1;
    goto timing_cleanup;
  }

  // Detect matrix size
  NUM_NODES = detect_matrix_size(argv[1]);
  if (NUM_NODES < 0) {
    ret = 1;
    goto timing_cleanup;
  }
  if (NUM_NODES > MAX_NODES) {
    fprintf(stderr, "Error: Matrix size %d exceeds maximum %d\n", NUM_NODES, MAX_NODES);
    ret = 1;
    goto timing_cleanup;
  }
  
  printf("Detected %d×%d adjacency matrix\n", NUM_NODES, NUM_NODES);

  // Allocate dynamic memory
  AdjMatrix = (int **)malloc(NUM_NODES * sizeof(int *));
  if (!AdjMatrix) {
    fprintf(stderr, "Error: Failed to allocate memory for adjacency matrix\n");
    ret = 1;
    goto timing_cleanup;
  }
  adj_allocated = 1;
  for (i = 0; i < NUM_NODES; i++) {
    AdjMatrix[i] = (int *)malloc(NUM_NODES * sizeof(int));
    if (!AdjMatrix[i]) {
      fprintf(stderr, "Error: Failed to allocate memory for adjacency matrix row %d\n", i);
      ret = 1;
      goto timing_cleanup;
    }
  }
  
  rgnNodes = (NODE *)malloc(NUM_NODES * sizeof(NODE));
  if (!rgnNodes) {
    fprintf(stderr, "Error: Failed to allocate memory for nodes\n");
    ret = 1;
    goto timing_cleanup;
  }
  nodes_allocated = 1;

  // Open and read the adjacency matrix file
  fp = fopen(argv[1],"r");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open file %s\n", argv[1]);
    ret = 1;
    goto timing_cleanup;
  }

  for (i=0; i<NUM_NODES; i++) {
    for (j=0; j<NUM_NODES; j++) {
      if (fscanf(fp,"%d",&k) != 1) {
        fprintf(stderr, "Error: Failed to read matrix element [%d][%d]\n", i, j);
        fclose(fp);
        fp = NULL;
        ret = 1;
        goto timing_cleanup;
      }
      AdjMatrix[i][j] = k;
    }
  }
  fclose(fp);
  fp = NULL;

  // Find shortest paths between nodes
  for (i=0,j=NUM_NODES/2; i<NUM_NODES && i<100; i++,j++) {
    j=j%NUM_NODES;
    dijkstra(i,j);
  }
  
  // Free memory
  ret = 0;

timing_cleanup:
  if (fp) {
    fclose(fp);
  }
  if (adj_allocated && AdjMatrix) {
    for (i = 0; i < NUM_NODES; i++) {
      if (AdjMatrix[i]) {
        free(AdjMatrix[i]);
      }
    }
    free(AdjMatrix);
    AdjMatrix = NULL;
  }
  if (nodes_allocated && rgnNodes) {
    free(rgnNodes);
    rgnNodes = NULL;
  }
  clock_gettime(CLOCK_MONOTONIC, &main_end);
  double kernel_time = get_dijkstra_kernel_time();
  double main_time = (main_end.tv_sec - main_start.tv_sec) +
                     (main_end.tv_nsec - main_start.tv_nsec) / 1e9;
  fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
  fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);
  fflush(timing_file);
  if (timing_file != stderr) {
    fclose(timing_file);
  }
  return ret;
}

