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

  /* Initialize node distances and predecessors */
  for (ch = 0; ch < NUM_NODES; ch++)
    {
      rgnNodes[ch].iDist = NONE;
      rgnNodes[ch].iPrev = NONE;
    }

  if (chStart == chEnd) 
    {
      printf("Shortest path is 0 in cost. Just stay where you are.\n");
    }
  else
    {
      rgnNodes[chStart].iDist = 0;
      rgnNodes[chStart].iPrev = NONE;

      /* Use a local priority queue (min-heap) instead of the global
         linked-list queue to reduce allocations and improve locality. */

      int heapSize = 0;
      /* Each entry: {node, dist, prev} */
      struct {
        int node;
        int dist;
        int prev;
      } heap[NUM_NODES];

      /* heap push */
      auto void heap_push(int node, int dist, int prev) {
        int idx = heapSize++;
        heap[idx].node = node;
        heap[idx].dist = dist;
        heap[idx].prev = prev;

        /* Up-heap (bubble up) */
        while (idx > 0)
          {
            int parent = (idx - 1) >> 1;
            if (heap[parent].dist <= heap[idx].dist)
              break;
            /* swap with parent */
            int tnode = heap[parent].node;
            int tdist = heap[parent].dist;
            int tprev = heap[parent].prev;

            heap[parent].node = heap[idx].node;
            heap[parent].dist = heap[idx].dist;
            heap[parent].prev = heap[idx].prev;

            heap[idx].node = tnode;
            heap[idx].dist = tdist;
            heap[idx].prev = tprev;

            idx = parent;
          }
      }

      /* heap pop (returns 0 if empty, 1 if success) */
      auto int heap_pop(int *node, int *dist, int *prev) {
        if (heapSize == 0)
          return 0;

        *node = heap[0].node;
        *dist = heap[0].dist;
        *prev = heap[0].prev;

        --heapSize;
        if (heapSize > 0)
          {
            heap[0].node = heap[heapSize].node;
            heap[0].dist = heap[heapSize].dist;
            heap[0].prev = heap[heapSize].prev;

            /* Down-heap (bubble down) */
            int idx = 0;
            while (1)
              {
                int left = (idx << 1) + 1;
                int right = left + 1;
                int smallest = idx;

                if (left < heapSize &&
                    heap[left].dist < heap[smallest].dist)
                  smallest = left;
                if (right < heapSize &&
                    heap[right].dist < heap[smallest].dist)
                  smallest = right;
                if (smallest == idx)
                  break;

                int tnode = heap[smallest].node;
                int tdist = heap[smallest].dist;
                int tprev = heap[smallest].prev;

                heap[smallest].node = heap[idx].node;
                heap[smallest].dist = heap[idx].dist;
                heap[smallest].prev = heap[idx].prev;

                heap[idx].node = tnode;
                heap[idx].dist = tdist;
                heap[idx].prev = tprev;

                idx = smallest;
              }
          }
        return 1;
      }

      /* Initialize priority queue with start node */
      heap_push(chStart, 0, NONE);

      /* Dijkstra's algorithm with OpenMP parallel relaxation of edges
         from the current frontier node. */
      while (heap_pop(&iNode, &iDist, &iPrev))
        {
          /* Skip if this is a stale entry (not the best known distance) */
          if (iDist != rgnNodes[iNode].iDist)
            continue;

          /* Parallelize relaxation over all neighbors of iNode. Each
             thread updates different rgnNodes[i] values; we use a
             critical section only when pushing into the local heap
             to keep it thread-safe. */
#pragma omp parallel for default(none) shared(AdjMatrix, rgnNodes, iNode, iDist, heap, &heapSize) private(i, iCost) if (NUM_NODES > 16)
          for (i = 0; i < NUM_NODES; i++)
            {
              iCost = AdjMatrix[iNode][i];
              if (iCost != NONE)
                {
                  int newDist = iDist + iCost;

                  /* Use a simple compare-and-swap style update; this
                     is not atomic but NUM_NODES is small and the
                     algorithm tolerates occasional redundant work
                     because we verify distances when popping. */
                  int oldDist = rgnNodes[i].iDist;
                  if ((oldDist == NONE) || (oldDist > newDist))
                    {
#pragma omp critical
                      {
                        if ((rgnNodes[i].iDist == NONE) ||
                            (rgnNodes[i].iDist > newDist))
                          {
                            rgnNodes[i].iDist = newDist;
                            rgnNodes[i].iPrev = iNode;
                            heap_push(i, newDist, iNode);
                          }
                      }
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

