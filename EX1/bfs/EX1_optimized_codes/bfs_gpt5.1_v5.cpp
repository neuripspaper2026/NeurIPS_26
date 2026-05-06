#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

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

    double kernel_time = (double)(kernel_end.tv_sec - kernel_start.tv_sec) +
                         (double)(kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    double main_time = (double)(main_end.tv_sec - main_start.tv_sec) +
                       (double)(main_end.tv_nsec - main_start.tv_nsec) / 1e9;

    fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

    if (timing_file != stderr)
        fclose(timing_file);

    return 0;
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

    if (fscanf(fp, "%d", &no_of_nodes) != 1 || no_of_nodes <= 0) {
        fclose(fp);
        return;
    }

    // allocate host memory as a single block to improve locality
    Node *h_graph_nodes = (Node *)malloc(sizeof(Node) * (size_t)no_of_nodes);
    int *h_graph_edges = NULL;
    int *h_cost = (int *)malloc(sizeof(int) * (size_t)no_of_nodes);
    if (!h_graph_nodes || !h_cost) {
        if (h_graph_nodes) free(h_graph_nodes);
        if (h_cost) free(h_cost);
        fclose(fp);
        return;
    }

    bool *h_graph_mask = (bool *)malloc(sizeof(bool) * (size_t)no_of_nodes);
    bool *h_updating_graph_mask = (bool *)malloc(sizeof(bool) * (size_t)no_of_nodes);
    bool *h_graph_visited = (bool *)malloc(sizeof(bool) * (size_t)no_of_nodes);

    if (!h_graph_mask || !h_updating_graph_mask || !h_graph_visited) {
        free(h_graph_nodes);
        free(h_cost);
        if (h_graph_mask) free(h_graph_mask);
        if (h_updating_graph_mask) free(h_updating_graph_mask);
        if (h_graph_visited) free(h_graph_visited);
        fclose(fp);
        return;
    }

    int start, edgeno;
    // initialize the memory
    for (int i = 0; i < no_of_nodes; i++) {
        if (fscanf(fp, "%d %d", &start, &edgeno) != 2) {
            free(h_graph_nodes);
            free(h_graph_mask);
            free(h_updating_graph_mask);
            free(h_graph_visited);
            free(h_cost);
            fclose(fp);
            return;
        }
        h_graph_nodes[i].starting = start;
        h_graph_nodes[i].no_of_edges = edgeno;
        h_graph_mask[i] = false;
        h_updating_graph_mask[i] = false;
        h_graph_visited[i] = false;
    }

    // read the source node from the file
    if (fscanf(fp, "%d", &source) != 1) {
        free(h_graph_nodes);
        free(h_graph_mask);
        free(h_updating_graph_mask);
        free(h_graph_visited);
        free(h_cost);
        fclose(fp);
        return;
    }

    if (source < 0 || source >= no_of_nodes) {
        free(h_graph_nodes);
        free(h_graph_mask);
        free(h_updating_graph_mask);
        free(h_graph_visited);
        free(h_cost);
        fclose(fp);
        return;
    }

    // set the source node as true in the mask
    h_graph_mask[source] = true;
    h_graph_visited[source] = true;

    if (fscanf(fp, "%d", &edge_list_size) != 1 || edge_list_size < 0) {
        free(h_graph_nodes);
        free(h_graph_mask);
        free(h_updating_graph_mask);
        free(h_graph_visited);
        free(h_cost);
        fclose(fp);
        return;
    }

    int id, cost;
    h_graph_edges = (int *)malloc(sizeof(int) * (size_t)edge_list_size);
    if (!h_graph_edges) {
        free(h_graph_nodes);
        free(h_graph_mask);
        free(h_updating_graph_mask);
        free(h_graph_visited);
        free(h_cost);
        fclose(fp);
        return;
    }

    for (int i = 0; i < edge_list_size; i++) {
        if (fscanf(fp, "%d", &id) != 1) {
            free(h_graph_nodes);
            free(h_graph_edges);
            free(h_graph_mask);
            free(h_updating_graph_mask);
            free(h_graph_visited);
            free(h_cost);
            fclose(fp);
            return;
        }
        if (fscanf(fp, "%d", &cost) != 1) {
            free(h_graph_nodes);
            free(h_graph_edges);
            free(h_graph_mask);
            free(h_updating_graph_mask);
            free(h_graph_visited);
            free(h_cost);
            fclose(fp);
            return;
        }
        h_graph_edges[i] = id;
    }

    fclose(fp);

    // allocate mem for the result on host side
    for (int i = 0; i < no_of_nodes; i++)
        h_cost[i] = -1;
    h_cost[source] = 0;

    printf("Start traversing the tree\n");

    // optimized BFS using an explicit queue instead of iterative masks
    int *queue = (int *)malloc(sizeof(int) * (size_t)no_of_nodes);
    if (!queue) {
        free(h_graph_nodes);
        free(h_graph_edges);
        free(h_graph_mask);
        free(h_updating_graph_mask);
        free(h_graph_visited);
        free(h_cost);
        return;
    }

    int head = 0;
    int tail = 0;
    queue[tail++] = source;

    while (head < tail) {
        int tid = queue[head++];
        int base = h_graph_nodes[tid].starting;
        int end = base + h_graph_nodes[tid].no_of_edges;
        int next_cost = h_cost[tid] + 1;

        for (int i = base; i < end; i++) {
            int nid = h_graph_edges[i];
            if (!h_graph_visited[nid]) {
                h_graph_visited[nid] = true;
                h_cost[nid] = next_cost;
                queue[tail++] = nid;
            }
        }
    }

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
    free(queue);
    free(h_graph_nodes);
    free(h_graph_edges);
    free(h_graph_mask);
    free(h_updating_graph_mask);
    free(h_graph_visited);
    free(h_cost);
}
