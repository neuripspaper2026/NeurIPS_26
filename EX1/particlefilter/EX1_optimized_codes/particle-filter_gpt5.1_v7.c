#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <limits.h>
#include <string.h>
#include <time.h>
#define PI 3.1415926535897932
/* optional output file for final results */
static const char *g_output_file = NULL;
static double g_kernel_time = 0.0;
/**
@var M value for Linear Congruential Generator (LCG); use GCC's value
*/
long M = INT_MAX;
/**
@var A value for LCG
*/
int A = 1103515245;
/**
@var C value for LCG
*/
int C = 12345;
/*****************************
*GET_TIME
*returns a long int representing the time
*****************************/
long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}
// Returns the number of seconds elapsed between the two specified times
float elapsed_time(long long start_time, long long end_time) {
    return (float)(end_time - start_time) / (1000 * 1000);
}
/**
* Takes in a double and returns an integer that approximates to that double
* @return if the mantissa < .5 => return value < input value; else return value
* > input value
*/
double roundDouble(double value) {
    int newValue = (int)(value);
    if (value - newValue < .5)
        return newValue;
    else
        return newValue++;
}
/**
* Set values of the 3D array to a newValue if that value is equal to the
* testValue
* @param testValue The value to be replaced
* @param newValue The value to replace testValue with
* @param array3D The image vector
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
*/
void setIf(int testValue, int newValue, int *array3D, int *dimX, int *dimY,
           int *dimZ) {
    int x, y, z;
    for (x = 0; x < *dimX; x++) {
        for (y = 0; y < *dimY; y++) {
            for (z = 0; z < *dimZ; z++) {
                if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue)
                    array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
            }
        }
    }
}
/**
* Generates a uniformly distributed random number using the provided seed and
* GCC's settings for the Linear Congruential Generator (LCG)
* @see http://en.wikipedia.org/wiki/Linear_congruential_generator
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a uniformly distributed number [0, 1)
*/
double randu(int *seed, int index) {
    int num = A * seed[index] + C;
    seed[index] = num % M;
    return fabs(seed[index] / ((double)M));
}
/**
* Generates a normally distributed random number using the Box-Muller
* transformation
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a double representing random number generated using the Box-Muller
* algorithm
* @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value
* for normal random distribution
*/
double randn(int *seed, int index) {
    /*Box-Muller algorithm*/
    double u = randu(seed, index);
    double v = randu(seed, index);
    double cosine = cos(2 * PI * v);
    double rt = -2 * log(u);
    return sqrt(rt) * cosine;
}
/**
* Sets values of 3D matrix using randomly generated numbers from a normal
* distribution
* @param array3D The video to be modified
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
* @param seed The seed array
*/
void addNoise(int *array3D, int *dimX, int *dimY, int *dimZ, int *seed) {
    int x, y, z;
    for (x = 0; x < *dimX; x++) {
        for (y = 0; y < *dimY; y++) {
            for (z = 0; z < *dimZ; z++) {
                array3D[x * *dimY * *dimZ + y * *dimZ + z] =
                    array3D[x * *dimY * *dimZ + y * *dimZ + z] +
                    (int)(5 * randn(seed, 0));
            }
        }
    }
}
/**
* Fills a radius x radius matrix representing the disk
* @param disk The pointer to the disk to be made
* @param radius  The radius of the disk to be made
*/
void strelDisk(int *disk, int radius) {
    int diameter = radius * 2 - 1;
    int x, y;
    for (x = 0; x < diameter; x++) {
        for (y = 0; y < diameter; y++) {
            double distance = sqrt(pow((double)(x - radius + 1), 2) +
                                   pow((double)(y - radius + 1), 2));
            if (distance < radius)
                disk[x * diameter + y] = 1;
            else
                disk[x * diameter + y] = 0;
        }
    }
}
/**
* Dilates the provided video
* @param matrix The video to be dilated
* @param posX The x location of the pixel to be dilated
* @param posY The y location of the pixel to be dilated
* @param poxZ The z location of the pixel to be dilated
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
* @param error The error radius
*/
void dilate_matrix(int *matrix, int posX, int posY, int posZ, int dimX,
                   int dimY, int dimZ, int error) {
    int startX = posX - error;
    while (startX < 0)
        startX++;
    int startY = posY - error;
    while (startY < 0)
        startY++;
    int endX = posX + error;
    while (endX > dimX)
        endX--;
    int endY = posY + error;
    while (endY > dimY)
        endY--;
    int x, y;
    for (x = startX; x < endX; x++) {
        for (y = startY; y < endY; y++) {
            double distance =
                sqrt(pow((double)(x - posX), 2) + pow((double)(y - posY), 2));
            if (distance < error)
                matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
        }
    }
}

/**
* Dilates the target matrix using the radius as a guide
* @param matrix The reference matrix
* @param dimX The x dimension of the video
* @param dimY The y dimension of the video
* @param dimZ The z dimension of the video
* @param error The error radius to be dilated
* @param newMatrix The target matrix
*/
void imdilate_disk(int *matrix, int dimX, int dimY, int dimZ, int error,
                   int *newMatrix) {
    int x, y, z;
    for (z = 0; z < dimZ; z++) {
        for (x = 0; x < dimX; x++) {
            for (y = 0; y < dimY; y++) {
                if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
                    dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
                }
            }
        }
    }
}
/**
* Fills a 2D array describing the offsets of the disk object
* @param se The disk object
* @param numOnes The number of ones in the disk
* @param neighbors The array that will contain the offsets
* @param radius The radius used for dilation
*/
void getneighbors(int *se, int numOnes, double *neighbors, int radius) {
    int x, y;
    int neighY = 0;
    int center = radius - 1;
    int diameter = radius * 2 - 1;
    for (x = 0; x < diameter; x++) {
        for (y = 0; y < diameter; y++) {
            if (se[x * diameter + y]) {
                neighbors[neighY * 2] = (int)(y - center);
                neighbors[neighY * 2 + 1] = (int)(x - center);
                neighY++;
            }
        }
    }
}
/**
* The synthetic video sequence we will work with here is composed of a
* single moving object, circular in shape (fixed radius)
* The motion here is a linear motion
* the foreground intensity and the backgrounf intensity is known
* the image is corrupted with zero mean Gaussian noise
* @param I The video itself
* @param IszX The x dimension of the video
* @param IszY The y dimension of the video
* @param Nfr The number of frames of the video
* @param seed The seed array used for number generation
*/
void videoSequence(int *I, int IszX, int IszY, int Nfr, int *seed) {
    int k;
    int max_size = IszX * IszY * Nfr;
    /*get object centers*/
    int x0 = (int)roundDouble(IszY / 2.0);
    int y0 = (int)roundDouble(IszX / 2.0);
    I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

    /*move point*/
    int xk, yk, pos;
    for (k = 1; k < Nfr; k++) {
        xk = abs(x0 + (k - 1));
        yk = abs(y0 - 2 * (k - 1));
        pos = yk * IszY * Nfr + xk * Nfr + k;
        if (pos >= max_size)
            pos = 0;
        I[pos] = 1;
    }

    /*dilate matrix*/
    int *newMatrix = (int *)malloc(sizeof(int) * IszX * IszY * Nfr);
    imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
    int x, y;
    for (x = 0; x < IszX; x++) {
        for (y = 0; y < IszY; y++) {
            for (k = 0; k < Nfr; k++) {
                I[x * IszY * Nfr + y * Nfr + k] =
                    newMatrix[x * IszY * Nfr + y * Nfr + k];
            }
        }
    }
    free(newMatrix);

    /*define background, add noise*/
    setIf(0, 100, I, &IszX, &IszY, &Nfr);
    setIf(1, 228, I, &IszX, &IszY, &Nfr);
    /*add noise*/
    addNoise(I, &IszX, &IszY, &Nfr, seed);
}
/**
* Determines the likelihood sum based on the formula: SUM( (IK[IND] - 100)^2 -
* (IK[IND] - 228)^2)/ 100
* @param I The 3D matrix
* @param ind The current ind array
* @param numOnes The length of ind array
* @return A double representing the sum
*/
double calcLikelihoodSum(int *I, int *ind, int numOnes) {
    double likelihoodSum = 0.0;
    int y;
    for (y = 0; y < numOnes; y++)
        likelihoodSum +=
            (pow((I[ind[y]] - 100), 2) - pow((I[ind[y]] - 228), 2)) / 50.0;
    return likelihoodSum;
}
/**
* Finds the first element in the CDF that is greater than or equal to the
* provided value and returns that index
* @note This function uses sequential search
* @param CDF The CDF
* @param lengthCDF The length of CDF
* @param value The value to be found
* @return The index of value in the CDF; if value is never found, returns the
* last index
*/
int findIndex(double *CDF, int lengthCDF, double value) {
    int index = -1;
    int x;
    for (x = 0; x < lengthCDF; x++) {
        if (CDF[x] >= value) {
            index = x;
            break;
        }
    }
    if (index == -1) {
        return lengthCDF - 1;
    }
    return index;
}
/**
* Finds the first element in the CDF that is greater than or equal to the
* provided value and returns that index
* @note This function uses binary search before switching to sequential search
* @param CDF The CDF
* @param beginIndex The index to start searching from
* @param endIndex The index to stop searching
* @param value The value to find
* @return The index of value in the CDF; if value is never found, returns the
* last index
* @warning Use at your own risk; not fully tested
*/
int findIndexBin(double *CDF, int beginIndex, int endIndex, double value) {
    if (endIndex < beginIndex)
        return -1;
    int middleIndex = beginIndex + ((endIndex - beginIndex) / 2);
    /*check the value*/
    if (CDF[middleIndex] >= value) {
        /*check that it's good*/
        if (middleIndex == 0)
            return middleIndex;
        else if (CDF[middleIndex - 1] < value)
            return middleIndex;
        else if (CDF[middleIndex - 1] == value) {
            while (middleIndex > 0 && CDF[middleIndex - 1] == value)
                middleIndex--;
            return middleIndex;
        }
    }
    if (CDF[middleIndex] > value)
        return findIndexBin(CDF, beginIndex, middleIndex + 1, value);
    return findIndexBin(CDF, middleIndex - 1, endIndex, value);
}
/**
 * The implementation of the particle filter for many frames
* @see http://openmp.org/wp/
* @note This function is designed to work with a video of several frames. In
* addition, it references a provided MATLAB function which takes the video, the
* objxy matrix and the x and y arrays as arguments and returns the likelihoods
* @param I The video to be run
* @param IszX The x dimension of the video
* @param IszY The y dimension of the video
* @param Nfr The number of frames
* @param seed The seed array used for random number generation
* @param Nparticles The number of particles to be used
*/
void particleFilter(int *I, int IszX, int IszY, int Nfr, int *seed,
                    int Nparticles) {

    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    const int max_size = IszX * IszY * Nfr;
    long long start = get_time();

    const double halfIszY = IszY / 2.0;
    const double halfIszX = IszX / 2.0;
    const double centerY = roundDouble(halfIszY);
    const double centerX = roundDouble(halfIszX);

    double xe = centerY;
    double ye = centerX;

    const int radius = 5;
    const int diameter = radius * 2 - 1;
    const int diskSize = diameter * diameter;
    int *disk = (int *)malloc((size_t)diskSize * sizeof(int));
    strelDisk(disk, radius);

    int countOnes = 0;
    for (int idx = 0; idx < diskSize; ++idx) {
        if (disk[idx] == 1)
            ++countOnes;
    }

    double *objxy = (double *)malloc((size_t)countOnes * 2 * sizeof(double));
    getneighbors(disk, countOnes, objxy, radius);

    long long get_neighbors = get_time();
    printf("TIME TO GET NEIGHBORS TOOK: %f\n",
           elapsed_time(start, get_neighbors));

    double invNparticles = 1.0 / (double)Nparticles;
    double *weights = (double *)malloc((size_t)Nparticles * sizeof(double));
    for (int x = 0; x < Nparticles; x++) {
        weights[x] = invNparticles;
    }

    long long get_weights = get_time();
    printf("TIME TO GET WEIGHTSTOOK: %f\n",
           elapsed_time(get_neighbors, get_weights));

    double *likelihood = (double *)malloc((size_t)Nparticles * sizeof(double));
    double *arrayX = (double *)malloc((size_t)Nparticles * sizeof(double));
    double *arrayY = (double *)malloc((size_t)Nparticles * sizeof(double));
    double *xj = (double *)malloc((size_t)Nparticles * sizeof(double));
    double *yj = (double *)malloc((size_t)Nparticles * sizeof(double));
    double *CDF = (double *)malloc((size_t)Nparticles * sizeof(double));
    double *u = (double *)malloc((size_t)Nparticles * sizeof(double));
    int *ind = (int *)malloc((size_t)countOnes * (size_t)Nparticles * sizeof(int));

    for (int x = 0; x < Nparticles; x++) {
        arrayX[x] = xe;
        arrayY[x] = ye;
    }

    printf("TIME TO SET ARRAYS TOOK: %f\n",
           elapsed_time(get_weights, get_time()));

    const double invCountOnes = 1.0 / (double)countOnes;
    const double inv50 = 1.0 / 50.0;

    for (int k = 1; k < Nfr; k++) {
        long long set_arrays = get_time();

        for (int x = 0; x < Nparticles; x++) {
            arrayX[x] += 1.0 + 5.0 * randn(seed, x);
            arrayY[x] += -2.0 + 2.0 * randn(seed, x);
        }

        long long error = get_time();
        printf("TIME TO SET ERROR TOOK: %f\n", elapsed_time(set_arrays, error));

        for (int x = 0; x < Nparticles; x++) {
            const int baseIndex = x * countOnes;
            const int roundX = (int)roundDouble(arrayX[x]);
            const int roundY = (int)roundDouble(arrayY[x]);
            const int kOffset = k;
            const int lineStride = IszY * Nfr;
            const int frameStride = Nfr;

            for (int y = 0; y < countOnes; y++) {
                const int objIndex = y * 2;
                const int indX = roundX + (int)objxy[objIndex + 1];
                const int indY = roundY + (int)objxy[objIndex];
                long long idx = (long long)indX * lineStride +
                                (long long)indY * frameStride +
                                (long long)kOffset;
                if (idx < 0 || idx >= max_size)
                    idx = 0;
                ind[baseIndex + y] = (int)idx;
            }

            double sumLike = 0.0;
            const int *indPtr = &ind[baseIndex];
            for (int y = 0; y < countOnes; y++) {
                const int imgVal = I[indPtr[y]];
                const double diff100 = (double)imgVal - 100.0;
                const double diff228 = (double)imgVal - 228.0;
                sumLike += (diff100 * diff100 - diff228 * diff228) * inv50;
            }
            likelihood[x] = sumLike * invCountOnes;
        }

        long long likelihood_time = get_time();
        printf("TIME TO GET LIKELIHOODS TOOK: %f\n",
               elapsed_time(error, likelihood_time));

        double sumWeights = 0.0;
        for (int x = 0; x < Nparticles; x++) {
            const double w = weights[x] * exp(likelihood[x]);
            weights[x] = w;
            sumWeights += w;
        }

        long long exponential = get_time();
        printf("TIME TO GET EXP TOOK: %f\n",
               elapsed_time(likelihood_time, exponential));

        long long sum_time = get_time();
        printf("TIME TO SUM WEIGHTS TOOK: %f\n",
               elapsed_time(exponential, sum_time));

        const double invSumWeights = 1.0 / sumWeights;
        for (int x = 0; x < Nparticles; x++) {
            weights[x] *= invSumWeights;
        }

        long long normalize = get_time();
        printf("TIME TO NORMALIZE WEIGHTS TOOK: %f\n",
               elapsed_time(sum_time, normalize));

        xe = 0.0;
        ye = 0.0;

        for (int x = 0; x < Nparticles; x++) {
            const double w = weights[x];
            xe += arrayX[x] * w;
            ye += arrayY[x] * w;
        }

        long long move_time = get_time();
        printf("TIME TO MOVE OBJECT TOOK: %f\n",
               elapsed_time(normalize, move_time));
        printf("XE: %lf\n", xe);
        printf("YE: %lf\n", ye);

        const double dx = xe - centerY;
        const double dy = ye - centerX;
        double distance =
            sqrt(dx * dx + dy * dy);
        printf("%lf\n", distance);

        CDF[0] = weights[0];
        for (int x = 1; x < Nparticles; x++) {
            CDF[x] = weights[x] + CDF[x - 1];
        }

        long long cum_sum = get_time();
        printf("TIME TO CALC CUM SUM TOOK: %f\n",
               elapsed_time(move_time, cum_sum));

        double u1 = invNparticles * randu(seed, 0);
        for (int x = 0; x < Nparticles; x++) {
            u[x] = u1 + (double)x * invNparticles;
        }

        long long u_time = get_time();
        printf("TIME TO CALC U TOOK: %f\n", elapsed_time(cum_sum, u_time));

        for (int j = 0; j < Nparticles; j++) {
            int i = findIndex(CDF, Nparticles, u[j]);
            if (i == -1)
                i = Nparticles - 1;
            xj[j] = arrayX[i];
            yj[j] = arrayY[i];
        }

        long long xyj_time = get_time();
        printf("TIME TO CALC NEW ARRAY X AND Y TOOK: %f\n",
               elapsed_time(u_time, xyj_time));

        for (int x = 0; x < Nparticles; x++) {
            arrayX[x] = xj[x];
            arrayY[x] = yj[x];
            weights[x] = invNparticles;
        }

        long long reset = get_time();
        printf("TIME TO RESET WEIGHTS TOOK: %f\n",
               elapsed_time(xyj_time, reset));
    }

    xe = 0.0;
    ye = 0.0;

    for (int x = 0; x < Nparticles; x++) {
        const double w = weights[x];
        xe += arrayX[x] * w;
        ye += arrayY[x] * w;
    }

    const double dx = xe - centerY;
    const double dy = ye - centerX;
    double distance = sqrt(dx * dx + dy * dy);

    if (g_output_file != NULL) {
        FILE *fpo = fopen(g_output_file, "w");
        if (fpo) {
            fprintf(fpo, "XE: %lf\n", xe);
            fprintf(fpo, "YE: %lf\n", ye);
            fprintf(fpo, "distance: %lf\n", distance);
            fclose(fpo);
        } else {
            fprintf(stderr, "Failed to open output file: %s\n", g_output_file);
        }
    } else if (getenv("OUTPUT")) {
        FILE *fpo = fopen("output.txt", "w");
        if (fpo) {
            fprintf(fpo, "XE: %lf\n", xe);
            fprintf(fpo, "YE: %lf\n", ye);
            fprintf(fpo, "distance: %lf\n", distance);
            fclose(fpo);
        }
    }

    free(disk);
    free(objxy);
    free(weights);
    free(likelihood);
    free(xj);
    free(yj);
    free(arrayX);
    free(arrayY);
    free(CDF);
    free(u);
    free(ind);

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    g_kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
int main(int argc, char *argv[]) {
    struct timespec main_start, main_end;
    clock_gettime(CLOCK_MONOTONIC, &main_start);

    char *usage = "openmp.out -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles> [-o <output_file>]";
    // check number of arguments (allow optional -o <file>)
    if (argc != 9 && argc != 11) {
        printf("%s\n", usage);
        return 0;
    }
    // check args deliminators
    if (strcmp(argv[1], "-x") || strcmp(argv[3], "-y") ||
        strcmp(argv[5], "-z") || strcmp(argv[7], "-np")) {
        printf("%s\n", usage);
        return 0;
    }
    // parse optional -o <file>
    if (argc >= 11) {
        for (int i = 9; i + 1 < argc; ++i) {
            if (strcmp(argv[i], "-o") == 0) {
                g_output_file = argv[i + 1];
                break;
            }
        }
    }

    int IszX, IszY, Nfr, Nparticles;

    // converting a string to a integer
    if (sscanf(argv[2], "%d", &IszX) == EOF) {
        printf("ERROR: dimX input is incorrect");
        return 0;
    }

    if (IszX <= 0) {
        printf("dimX must be > 0\n");
        return 0;
    }

    // converting a string to a integer
    if (sscanf(argv[4], "%d", &IszY) == EOF) {
        printf("ERROR: dimY input is incorrect");
        return 0;
    }

    if (IszY <= 0) {
        printf("dimY must be > 0\n");
        return 0;
    }

    // converting a string to a integer
    if (sscanf(argv[6], "%d", &Nfr) == EOF) {
        printf("ERROR: Number of frames input is incorrect");
        return 0;
    }

    if (Nfr <= 0) {
        printf("number of frames must be > 0\n");
        return 0;
    }

    // converting a string to a integer
    if (sscanf(argv[8], "%d", &Nparticles) == EOF) {
        printf("ERROR: Number of particles input is incorrect");
        return 0;
    }

    if (Nparticles <= 0) {
        printf("Number of particles must be > 0\n");
        return 0;
    }
    // establish seed
    int *seed = (int *)malloc(sizeof(int) * Nparticles);
    int i;
    for (i = 0; i < Nparticles; i++)
        seed[i] = i;
    // malloc matrix
    int *I = (int *)malloc(sizeof(int) * IszX * IszY * Nfr);
    long long start = get_time();
    // call video sequence
    videoSequence(I, IszX, IszY, Nfr, seed);
    long long endVideoSequence = get_time();
    printf("VIDEO SEQUENCE TOOK %f\n", elapsed_time(start, endVideoSequence));
    // call particle filter
    particleFilter(I, IszX, IszY, Nfr, seed, Nparticles);
    long long endParticleFilter = get_time();
    printf("PARTICLE FILTER TOOK %f\n",
           elapsed_time(endVideoSequence, endParticleFilter));
    printf("ENTIRE PROGRAM TOOK %f\n", elapsed_time(start, endParticleFilter));

    free(seed);
    free(I);

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

    fprintf(timing_file, "KERNEL_TIME: %.9f\n", g_kernel_time);
    fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

    if (timing_file != stderr)
        fclose(timing_file);

    return 0;
}