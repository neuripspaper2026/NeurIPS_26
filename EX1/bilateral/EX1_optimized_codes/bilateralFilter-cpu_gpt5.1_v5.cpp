 #include <math.h>
 #include <string.h>
 #include <stdio.h>
 #include <stdlib.h>
#include <time.h>
 
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
    const int wh = w * h;
    float4 *hImage = new float4[wh];

    for (int i = 0; i < wh; ++i)
    {
        hImage[i] = hrgbaIntToFloat(pSrc[i]);
    }

    for (int y = 0; y < h; ++y)
    {
        const int yOffset = y * w;

        for (int x = 0; x < w; ++x)
        {
            const int centerIdx = yOffset + x;
            const float4 centerPix = hImage[centerIdx];

            float4 t(0.0f);
            float sum = 0.0f;

            for (int i = -r; i <= r; ++i)
            {
                int neighborY = y + i;

                if (neighborY < 0)
                {
                    neighborY = 0;
                }
                else if (neighborY >= h)
                {
                    neighborY = h - 1;
                }

                const int nyOffset = neighborY * w;
                const float gy = gaussian[r + i];

                for (int j = -r; j <= r; ++j)
                {
                    const float domainDist = gy * gaussian[r + j];

                    int neighborX = x + j;

                    if (neighborX < 0)
                    {
                        neighborX = 0;
                    }
                    else if (neighborX >= w)
                    {
                        neighborX = w - 1;
                    }

                    const int nIdx = nyOffset + neighborX;
                    const float4 neighborPix = hImage[nIdx];

                    const float colorDist = heuclideanLen(neighborPix, centerPix, e_d);
                    const float factor = domainDist * colorDist;

                    sum += factor;
                    t = add4(t, mul(factor, neighborPix));
                }
            }

            const float invSum = 1.0f / sum;
            pDest[centerIdx] = hrgbaFloatToInt(mul(invSum, t));
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