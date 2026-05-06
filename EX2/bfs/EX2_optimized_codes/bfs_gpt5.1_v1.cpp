#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

FILE *fp;
static const char *g_output_file = NULL;

// Structure to hold a node information
struct Node {
    int starting;
    int no_of_edges;
};

void BFSGraph(int argc, char **argv);

void Usage(int argc, char **argv) {

    fprintf(stderr, "Usage: %s <input_file> [-o <output_file>]\n", argv[0]);
}


////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    struct timespec main_start, main_end;
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &main_start);

    FILE *timing_file = stderr;
    const char *timing_path = getenv("TIMING_LOG_FILE");
    if (timing_path && timing_path[0] != '\0') {
        FILE *tmp = fopen(timing_path, "w");
        if (tmp)
            timing_file = tmp;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);
    BFSGraph(argc, argv);
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);

    clock_gettime(CLOCK_MONOTONIC, &main_end);

    double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                         (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    double main_time = (main_end.tv_sec - main_start.tv_sec) +
                       (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

    fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

    if (timing_file != stderr)
        fclose(timing_file);
}


////////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(int argc, char **argv) {
    int no_of_nodes = 0;
    int edge_list_size = 0;
    char *input_f;

    if (argc == 2) {
        input_f = argv[1];
    } else if (argc == 4 && strcmp(argv[2], "-o") == 0) {
        input_f = argv[1];
        g_output_file = argv[3];
    } else {
        Usage(argc, argv);
        exit(0);
    }

    printf("Reading File\n");
    // Read in Graph from a file
    fp = fopen(input_f, "r");
    if (!fp) {
        printf("Error Reading graph file\n");
        return;
    }

    int source = 0;

    if (fscanf(fp, "%d", &no_of_nodes) != 1) {
        fclose(fp);
        return;
    }

    // allocate host memory
    Node *h_graph_nodes = (Node *)malloc(sizeof(Node) * (size_t)no_of_nodes);
    bool *h_graph_mask = (bool *)malloc(sizeof(bool) * (size_t)no_of_nodes);
    bool *h_updating_graph_mask =
        (bool *)malloc(sizeof(bool) * (size_t)no_of_nodes);
    bool *h_graph_visited =
        (bool *)malloc(sizeof(bool) * (size_t)no_of_nodes);

    if (!h_graph_nodes || !h_graph_mask || !h_updating_graph_mask ||
        !h_graph_visited) {
        if (fp)
            fclose(fp);
        free(h_graph_nodes);
        free(h_graph_mask);
        free(h_updating_graph_mask);
        free(h_graph_visited);
        return;
    }

    int start, edgeno;
    // initalize the memory and read nodes
    for (int i = 0; i < no_of_nodes; i++) {
        if (fscanf(fp, "%d %d", &start, &edgeno) != 2) {
            no_of_nodes = i;
            break;
        }
        h_graph_nodes[i].starting = start;
        h_graph_nodes[i].no_of_edges = edgeno;
        h_graph_mask[i] = false;
        h_updating_graph_mask[i] = false;
        h_graph_visited[i] = false;
    }

    // read the source node from the file
    if (fscanf(fp, "%d", &source) != 1) {
        if (fp)
            fclose(fp);
        free(h_graph_nodes);
        free(h_graph_mask);
        free(h_updating_graph_mask);
        free(h_graph_visited);
        return;
    }
    if (source < 0 || source >= no_of_nodes) {
        source = 0;
    }

    // set the source node as true in the mask
    h_graph_mask[source] = true;
    h_graph_visited[source] = true;

    if (fscanf(fp, "%d", &edge_list_size) != 1) {
        if (fp)
            fclose(fp);
        free(h_graph_nodes);
        free(h_graph_mask);
        free(h_updating_graph_mask);
        free(h_graph_visited);
        return;
    }

    int id, cost;
    int *h_graph_edges =
        (int *)malloc(sizeof(int) * (size_t)edge_list_size);
    if (!h_graph_edges) {
        if (fp)
            fclose(fp);
        free(h_graph_nodes);
        free(h_graph_mask);
        free(h_updating_graph_mask);
        free(h_graph_visited);
        return;
    }

    for (int i = 0; i < edge_list_size; i++) {
        if (fscanf(fp, "%d", &id) != 1)
            break;
        if (fscanf(fp, "%d", &cost) != 1)
            break;
        h_graph_edges[i] = id;
    }

    if (fp)
        fclose(fp);

    // allocate mem for the result on host side
    int *h_cost = (int *)malloc(sizeof(int) * (size_t)no_of_nodes);
    if (!h_cost) {
        free(h_graph_nodes);
        free(h_graph_edges);
        free(h_graph_mask);
        free(h_updating_graph_mask);
        free(h_graph_visited);
        return;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < no_of_nodes; i++) {
        h_cost[i] = -1;
    }
    h_cost[source] = 0;

    printf("Start traversing the tree\n");

    bool stop;
    // BFS using frontier-based masks
    do {
        // if no thread changes this value then the loop stops
        stop = false;

#ifdef _OPENMP
#pragma omp parallel
        {
#pragma omp for schedule(dynamic, 1024) nowait
#endif
            for (int tid = 0; tid < no_of_nodes; tid++) {
                if (h_graph_mask[tid]) {
                    h_graph_mask[tid] = false;
                    int edge_start = h_graph_nodes[tid].starting;
                    int edge_end = edge_start + h_graph_nodes[tid].no_of_edges;
                    for (int i = edge_start; i < edge_end; i++) {
                        int nid = h_graph_edges[i];
                        if (!h_graph_visited[nid]) {
                            h_cost[nid] = h_cost[tid] + 1;
                            h_updating_graph_mask[nid] = true;
                        }
                    }
                }
            }

#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|| : stop)
#endif
            for (int tid = 0; tid < no_of_nodes; tid++) {
                if (h_updating_graph_mask[tid]) {
                    h_graph_mask[tid] = true;
                    h_graph_visited[tid] = true;
                    stop = true;
                    h_updating_graph_mask[tid] = false;
                }
            }
#ifdef _OPENMP
        }
#endif
    } while (stop);

    // Store the result into a file
    if (g_output_file != NULL) {
        FILE *fpo = fopen(g_output_file, "w");
        if (fpo) {
            for (int i = 0; i < no_of_nodes; i++)
                fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
            fclose(fpo);
        } else {
            fprintf(stderr, "Failed to open output file: %s\n", g_output_file);
        }
    } else if (getenv("OUTPUT")) {
        FILE *fpo = fopen("output.txt", "w");
        if (fpo) {
            for (int i = 0; i < no_of_nodes; i++)
                fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
            fclose(fpo);
        }
    }

    // cleanup memory
    free(h_graph_nodes);
    free(h_graph_edges);
    free(h_graph_mask);
    free(h_updating_graph_mask);
    free(h_graph_visited);
    free(h_cost);
}
