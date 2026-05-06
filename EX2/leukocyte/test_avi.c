#include <stdio.h>
#include <stdlib.h>
#include "../../common_rodinia/avi/avilib.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <avi_file>\n", argv[0]);
        return 1;
    }
    
    printf("Opening AVI file: %s\n", argv[1]);
    avi_t *avi = AVI_open_input_file(argv[1], 1);
    
    if (avi == NULL) {
        AVI_print_error("Error opening AVI file");
        return 1;
    }
    
    printf("AVI file opened successfully!\n");
    printf("  Width: %d\n", AVI_video_width(avi));
    printf("  Height: %d\n", AVI_video_height(avi));
    printf("  Frames: %ld\n", AVI_video_frames(avi));
    printf("  FPS: %.2f\n", AVI_frame_rate(avi));
    
    // Try to read first frame
    int width = AVI_video_width(avi);
    int height = AVI_video_height(avi);
    unsigned char *buffer = malloc(width * height);
    
    AVI_set_video_position(avi, 0);
    int dummy;
    if (AVI_read_frame(avi, (char *)buffer, &dummy) == -1) {
        AVI_print_error("Error reading frame");
        free(buffer);
        AVI_close(avi);
        return 1;
    }
    
    printf("Successfully read first frame!\n");
    
    free(buffer);
    AVI_close(avi);
    return 0;
}

