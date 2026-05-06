#include <math.h>
 #include <string.h>
 #include <stdio.h>
 #include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
 
 #define EPSILON 1e-3
 struct uchar4 { unsigned char x, y, z, w; };
 extern "C" void LoadBMPFile(uchar4 **dst, unsigned int *width,
                             unsigned int *height, const char *name);
 extern "C" void updateGaussianGold(float delta, int radius);
 extern "C" void bilateralFilterGold(unsigned int *pSrc,
                                     unsigned int *pDest,
                                     float e_d,
                                     int w, int h, int r);
 //variables
 float gaussian[50];
 
 struct float4
 {
     float x;
     float y;
     float z;
     float w;
 
     float4() {};
     float4(float value)
     {
         x = y = z = w = value;
     }
 };
 
 void updateGaussianGold(float delta, int radius)
 {
     for (int i = 0; i < 2 * radius + 1; i++)
     {
         int x = i - radius;
         gaussian[i] = exp(-(x * x) /
                           (2 * delta * delta));
     }
 }
 
 float heuclideanLen(float4 a, float4 b, float d)
 {
     float mod = (b.x - a.x) * (b.x - a.x) +
                 (b.y - a.y) * (b.y - a.y) +
                 (b.z - a.z) * (b.z - a.z) +
                 (b.w - a.w) * (b.w - a.w);
 
     return exp(-mod / (2 * d * d));
 }
 
 unsigned int hrgbaFloatToInt(float4 rgba)
 {
     unsigned int w = (((unsigned int)(fabs(rgba.w) * 255.0f)) & 0xff) << 24;
     unsigned int z = (((unsigned int)(fabs(rgba.z) * 255.0f)) & 0xff) << 16;
     unsigned int y = (((unsigned int)(fabs(rgba.y) * 255.0f)) & 0xff) << 8;
     unsigned int x = ((unsigned int)(fabs(rgba.x) * 255.0f)) & 0xff;
 
     return (w | z | y | x);
 }
 
 float4 hrgbaIntToFloat(unsigned int c)
 {
     float4 rgba;
     rgba.x = (c & 0xff) * 0.003921568627f;       //  /255.0f;
     rgba.y = ((c>>8) & 0xff) * 0.003921568627f;  //  /255.0f;
     rgba.z = ((c>>16) & 0xff) * 0.003921568627f; //  /255.0f;
     rgba.w = ((c>>24) & 0xff) * 0.003921568627f; //  /255.0f;
     return rgba;
 }
 
 float4 mul(float a, float4 b)
 {
     float4 ans;
     ans.x = a * b.x;
     ans.y = a * b.y;
     ans.z = a * b.z;
     ans.w = a * b.w;
 
     return ans;
 }
 
 float4 add4(float4 a, float4 b)
 {
     float4 ans;
     ans.x = a.x + b.x;
     ans.y = a.y + b.y;
     ans.z = a.z + b.z;
     ans.w = a.w + b.w;
 
     return ans;
 }
 
void bilateralFilterGold(unsigned int *pSrc,
                        unsigned int *pDest,
                        float e_d,
                        int w, int h, int r)
{
    float4 *hImage = new float4[w * h];
    
    // Convert input to float4 format with parallel processing
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; y++)
    {
        int row_offset = y * w;
        for (int x = 0; x < w; x++)
        {
            hImage[row_offset + x] = hrgbaIntToFloat(pSrc[row_offset + x]);
        }
    }
    
    // Precompute Gaussian products for domain distance
    float gaussian_products[(2*r+1)*(2*r+1)];
    for (int i = 0; i <= 2*r; i++)
    {
        for (int j = 0; j <= 2*r; j++)
        {
            gaussian_products[i*(2*r+1) + j] = gaussian[i] * gaussian[j];
        }
    }
    
    // Main bilateral filter loop with parallel processing
    #pragma omp parallel for schedule(dynamic, 4)
    for (int y = 0; y < h; y++)
    {
        int row_offset = y * w;
        
        for (int x = 0; x < w; x++)
        {
            float4 t(0.0f);
            float sum = 0.0f;
            float4 center_pixel = hImage[row_offset + x];
            
            // Clamp neighbor Y range once per pixel
            int minY = (y - r < 0) ? 0 : y - r;
            int maxY = (y + r >= h) ? h - 1 : y + r;
            
            for (int i = minY; i <= maxY; i++)
            {
                int neighbor_row = i * w;
                int gauss_i_idx = (i - y + r) * (2*r+1);
                
                // Clamp neighbor X range once per pixel
                int minX = (x - r < 0) ? 0 : x - r;
                int maxX = (x + r >= w) ? w - 1 : x + r;
                
                for (int j = minX; j <= maxX; j++)
                {
                    float domainDist = gaussian_products[gauss_i_idx + (j - x + r)];
                    
                    float4 neighbor_pixel = hImage[neighbor_row + j];
                    float colorDist = heuclideanLen(neighbor_pixel, center_pixel, e_d);
                    float factor = domainDist * colorDist;
                    
                    sum += factor;
                    t = add4(t, mul(factor, neighbor_pixel));
                }
            }
            
            pDest[row_offset + x] = hrgbaFloatToInt(mul(1.0f / sum, t));
        }
    }
    
    delete[] hImage;
}
 
 int main(int argc, char **argv)
 {
     if (argc != 6)
     {
         fprintf(stderr, "Usage: %s <image> <e_d> <g_d> <radius> <output>\n", argv[0]);
         return 1;
     }
 
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
 
     const char *input = argv[1];
     float e_d = atof(argv[2]);
     float g_d = atof(argv[3]);
     int radius = atoi(argv[4]);
     const char *output = argv[5];
 
     uchar4 *img;
     unsigned int w, h;
     LoadBMPFile(&img, &w, &h, input);
 
     unsigned int *src = (unsigned int *)img;
     unsigned int *dest = (unsigned int *)malloc(sizeof(unsigned int) * w * h);
 
     updateGaussianGold(g_d, radius);
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);
     bilateralFilterGold(src, dest, e_d, w, h, radius);
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
 
     FILE *f = fopen(output, "wb");
     if (f)
     {
         fwrite(dest, sizeof(unsigned int), w * h, f);
         fclose(f);
     }
 
     free(img);
     free(dest);

    clock_gettime(CLOCK_MONOTONIC, &main_end);
    double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                         (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    double main_time = (main_end.tv_sec - main_start.tv_sec) +
                       (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

    fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

    if (timing_file != stderr)
        fclose(timing_file);

     return 0;
 }