#include "../dwt_cuda/common.h"
#include "../dwt_cuda/transform_buffer.h"
#include "../dwt_cuda/io.h"

namespace dwt_cuda {

template <int WIN_SIZE_X, int WIN_SIZE_Y> class FDWT97 {
  private:
    typedef TransformBuffer<float, WIN_SIZE_X, WIN_SIZE_Y + 7, 4> FDWT97Buffer;
    FDWT97Buffer buffer;
    enum { STRIDE = FDWT97Buffer::VERTICAL_STRIDE };

    template <bool CHECKED> struct FDWT97ColumnLoadingInfo {
        VerticalDWTPixelLoader<float, CHECKED> loader;
        int offset;
    };

    __device__ __forceinline__ void horizontalFDWT97(const int lines,
                                                     const int firstLine) {
        __syncthreads();
        buffer.forEachHorizontalOdd(firstLine, lines,
                                    AddScaledSum(f97Predict1));
        __syncthreads();
        buffer.forEachHorizontalEven(firstLine, lines,
                                     AddScaledSum(f97Update1));
        __syncthreads();
        buffer.forEachHorizontalOdd(firstLine, lines,
                                    AddScaledSum(f97Predict2));
        __syncthreads();
        buffer.forEachHorizontalEven(firstLine, lines,
                                     AddScaledSum(f97Update2));
        __syncthreads();
        buffer.scaleHorizontal(scale97Div, scale97Mul, firstLine, lines);
        __syncthreads();
    }

    template <bool CHECKED>
    __device__ __forceinline__ void
    initColumn(FDWT97ColumnLoadingInfo<CHECKED> &column,
               const int columnIndex, const float *const input, const int sizeX,
               const int sizeY, const int firstY) {
        column.offset = buffer.getColumnOffset(columnIndex);
        const int firstX = blockIdx.x * WIN_SIZE_X + columnIndex;

        if (blockIdx.y == 0) {
            column.loader.init(sizeX, sizeY, firstX, firstY);

            float v0 = column.loader.loadFrom(input);
            buffer[column.offset + 4 * STRIDE] = v0;

            float v1 = column.loader.loadFrom(input);
            buffer[column.offset + 3 * STRIDE] = v1;
            buffer[column.offset + 5 * STRIDE] = v1;

            float v2 = column.loader.loadFrom(input);
            buffer[column.offset + 2 * STRIDE] = v2;
            buffer[column.offset + 6 * STRIDE] = v2;

            float v3 = column.loader.loadFrom(input);
            buffer[column.offset + 1 * STRIDE] = v3;

            float v4 = column.loader.loadFrom(input);
            buffer[column.offset + 0 * STRIDE] = v4;

            column.loader.init(sizeX, sizeY, firstX, firstY + 3);
        } else {
            column.loader.init(sizeX, sizeY, firstX, firstY - 4);

#pragma unroll
            for (int i = 0; i < 7; i++) {
                buffer[column.offset + i * STRIDE] =
                    column.loader.loadFrom(input);
            }
        }
    }

    template <bool CHECKED>
    __device__ __forceinline__ void
    loadWindowIntoColumn(const float *const input,
                         FDWT97ColumnLoadingInfo<CHECKED> &column) {
#pragma unroll
        for (int i = 7; i < (7 + WIN_SIZE_Y); i++) {
            buffer[column.offset + i * STRIDE] = column.loader.loadFrom(input);
        }
    }

    template <bool CHECK_LOADS, bool CHECK_WRITES>
    __device__ void transform(const float *const in, float *const out,
                              const int sizeX, const int sizeY,
                              const int winSteps) {
        FDWT97ColumnLoadingInfo<CHECK_LOADS> loadedColumn;
        FDWT97ColumnLoadingInfo<CHECK_LOADS> boundaryColumn;

        const int firstY = blockIdx.y * WIN_SIZE_Y * winSteps;
        initColumn(loadedColumn, threadIdx.x, in, sizeX, sizeY, firstY);

        boundaryColumn.offset = 0;
        boundaryColumn.loader.clear();

        if (threadIdx.x < 7) {
            const int colId =
                threadIdx.x + ((threadIdx.x < 3) ? WIN_SIZE_X : -7);
            initColumn(boundaryColumn, colId, in, sizeX, sizeY, firstY);
        }

        horizontalFDWT97(7, 0);

        const int outColumnIndex = parityIdx<WIN_SIZE_X>();
        const int firstX = blockIdx.x * WIN_SIZE_X + outColumnIndex;
        VerticalDWTBandWriter<float, CHECK_WRITES> writer;
        writer.init(sizeX, sizeY, firstX, firstY);

        const int outColumnOffset = buffer.getColumnOffset(outColumnIndex);

#pragma unroll 1
        for (int w = 0; w < winSteps; w++) {
            loadWindowIntoColumn(in, loadedColumn);

            if (threadIdx.x < 7) {
                loadWindowIntoColumn(in, boundaryColumn);
            }

            horizontalFDWT97(WIN_SIZE_Y, 7);

            float last7Lines[7];
#pragma unroll
            for (int i = 0; i < 7; i++) {
                last7Lines[i] =
                    buffer[outColumnOffset + (WIN_SIZE_Y + i) * STRIDE];
            }

            buffer.forEachVerticalOdd(outColumnOffset,
                                      AddScaledSum(f97Predict1));
            buffer.forEachVerticalEven(outColumnOffset,
                                       AddScaledSum(f97Update1));
            buffer.forEachVerticalOdd(outColumnOffset,
                                      AddScaledSum(f97Predict2));
            buffer.forEachVerticalEven(outColumnOffset,
                                       AddScaledSum(f97Update2));

#pragma unroll
            for (int i = 4; i < (4 + WIN_SIZE_Y); i += 2) {
                const int index = outColumnOffset + i * STRIDE;
                const float low = buffer[index] * scale97Div;
                const float high = buffer[index + STRIDE] * scale97Mul;
                writer.writeLowInto(out, low);
                writer.writeHighInto(out, high);
            }

#pragma unroll
            for (int i = 0; i < 7; i++) {
                buffer[outColumnOffset + i * STRIDE] = last7Lines[i];
            }

            __syncthreads();
        }
    }

  public:
    __device__ static void run(const float *const input, float *const output,
                               const int sx, const int sy, const int steps) {
        __shared__ FDWT97<WIN_SIZE_X, WIN_SIZE_Y> fdwt97;

        const int maxX = (blockIdx.x + 1) * WIN_SIZE_X + 3;
        const int maxY = (blockIdx.y + 1) * WIN_SIZE_Y * steps + 3;
        const bool atRightBoudary = maxX >= sx;
        const bool atBottomBoudary = maxY >= sy;

        if (atBottomBoudary) {
            fdwt97.transform<true, true>(input, output, sx, sy, steps);
        } else if (atRightBoudary) {
            fdwt97.transform<false, true>(input, output, sx, sy, steps);
        } else {
            fdwt97.transform<false, false>(input, output, sx, sy, steps);
        }
    }

};

template <int WIN_SX, int WIN_SY>
__launch_bounds__(WIN_SX, CTMIN(SHM_SIZE / sizeof(FDWT97<WIN_SX, WIN_SY>), 8))
__global__ void fdwt97Kernel(const float *const input, float *const output,
                             const int sx, const int sy, const int steps) {
    FDWT97<WIN_SX, WIN_SY>::run(input, output, sx, sy, steps);
}

template <int WIN_SX, int WIN_SY>
void launchFDWT97Kernel(float *in, float *out, int sx, int sy) {
    const int steps = divRndUp(sy, 15 * WIN_SY);
    dim3 gSize(divRndUp(sx, WIN_SX), divRndUp(sy, WIN_SY * steps));

    PERF_BEGIN
    fdwt97Kernel<WIN_SX, WIN_SY><<<gSize, WIN_SX>>>(in, out, sx, sy, steps);
    PERF_END("        FDWT97", sx, sy)
    CudaDWTTester::checkLastKernelCall("FDWT 9/7 kernel");
}

void fdwt97(float *in, float *out, int sizeX, int sizeY, int levels) {
    if (sizeX >= 960) {
        launchFDWT97Kernel<192, 8>(in, out, sizeX, sizeY);
    } else if (sizeX >= 480) {
        launchFDWT97Kernel<128, 6>(in, out, sizeX, sizeY);
    } else {
        launchFDWT97Kernel<64, 6>(in, out, sizeX, sizeY);
    }

    if (levels > 1) {
        const int llSizeX = divRndUp(sizeX, 2);
        const int llSizeY = divRndUp(sizeY, 2);
        memCopy(in, out, llSizeX, llSizeY);
        fdwt97(in, out, llSizeX, llSizeY, levels - 1);
    }
}

} // end of namespace dwt_cuda
