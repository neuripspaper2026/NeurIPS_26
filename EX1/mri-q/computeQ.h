// computeQ.h
#pragma once

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

void ComputePhiMagCPU(int numK, float* phiR, float* phiI, float* phiMag);

void ComputeQCPU(int numK, int numX,
                 struct kValues *kVals,
                 float* x, float* y, float* z,
                 float *Qr, float *Qi);

void createDataStructsCPU(int numK, int numX, float** phiMag,
                          float** Qr, float** Qi);
