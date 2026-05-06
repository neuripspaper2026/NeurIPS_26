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

    // allocate host memory (single malloc to improve locality)
    size_t nodes_sz = (size_t)no_of_nodes * sizeof(struct Node);
    size_t mask_sz = (size_t)no_of_nodes * sizeof(bool);
    size_t cost_sz = (size_t)no_of_nodes * sizeof(int);
    unsigned char *block = (unsigned char *)malloc(nodes_sz + 3 * mask_sz + cost_sz);
    if (!block) {
        fclose(fp);
        return;
    }

    struct Node *h_graph_nodes = (struct Node *)block;
    bool *h_graph_mask = (bool *)(block + nodes_sz);
    bool *h_updating_graph_mask = (bool *)(block + nodes_sz + mask_sz);
    bool *h_graph_visited = (bool *)(block + nodes_sz + 2 * mask_sz);
    int *h_cost = (int *)(block + nodes_sz + 3 * mask_sz);

    int start, edgeno;
    // initialize the memory
    for (int i = 0; i < no_of_nodes; i++) {
        if (fscanf(fp, "%d %d", &start, &edgeno) != 2) {
            fclose(fp);
            free(block);
            return;
        }
        h_graph_nodes[i].starting = start;
        h_graph_nodes[i].no_of_edges = edgeno;
        h_graph_mask[i] = false;
        h_updating_graph_mask[i] = false;
        h_graph_visited[i] = false;
        h_cost[i] = -1;
    }

    // read the source node from the file
    if (fscanf(fp, "%d", &source) != 1 || source < 0 || source >= no_of_nodes) {
        fclose(fp);
        free(block);
        return;
    }

    // set the source node as true in the mask
    h_graph_mask[source] = true;
    h_graph_visited[source] = true;
    h_cost[source] = 0;

    if (fscanf(fp, "%d", &edge_list_size) != 1 || edge_list_size < 0) {
        fclose(fp);
        free(block);
        return;
    }

    int id, cost;
    int *h_graph_edges = (int *)malloc((size_t)edge_list_size * sizeof(int));
    if (!h_graph_edges) {
        fclose(fp);
        free(block);
        return;
    }

    for (int i = 0; i < edge_list_size; i++) {
        if (fscanf(fp, "%d", &id) != 1) {
            fclose(fp);
            free(h_graph_edges);
            free(block);
            return;
        }
        if (fscanf(fp, "%d", &cost) != 1) {
            fclose(fp);
            free(h_graph_edges);
            free(block);
            return;
        }
        h_graph_edges[i] = id;
    }

    fclose(fp);

    printf("Start traversing the tree\n");

    // BFS using frontier queue (serial optimization)
    int *queue = (int *)malloc((size_t)no_of_nodes * sizeof(int));
    if (!queue) {
        free(h_graph_edges);
        free(block);
        return;
    }
    int head = 0, tail = 0;
    queue[tail++] = source;

    while (head < tail) {
        int tid = queue[head++];
        const int edge_start = h_graph_nodes[tid].starting;
        const int edge_end = edge_start + h_graph_nodes[tid].no_of_edges;
        const int base_cost = h_cost[tid] + 1;

        for (int i = edge_start; i < edge_end; i++) {
            int nid = h_graph_edges[i];
            if (!h_graph_visited[nid]) {
                h_graph_visited[nid] = true;
                h_cost[nid] = base_cost;
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
    free(h_graph_edges);
    free(block);
}
