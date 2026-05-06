<<<CODE>>>
#include "../dwt_cuda/common.h"
#include "../dwt_cuda/transform_buffer.h"
#include "../dwt_cuda/io.h"


namespace dwt_cuda {


/// Wraps a buffer and methods for computing 9/7 FDWT with sliding window
/// of specified size. Template arguments specify this size.
/// @tparam WIN_SIZE_X  width of sliding window
/// @tparam WIN_SIZE_Y  height of sliding window
template <int WIN_SIZE_X, int WIN_SIZE_Y> class FDWT97 {
  private:
    /// Type of shared memory buffer used for 9/7 DWT.
    typedef TransformBuffer<float, WIN_SIZE_X, WIN_SIZE_Y + 7, 4> FDWT97Buffer;

    /// Actual shared buffer used for forward 9/7 DWT.
    FDWT97Buffer buffer;

    /// Difference of indices of two vertically neighboring items in buffer.
    enum { STRIDE = FDWT97Buffer::VERTICAL_STRIDE };


    /// One thread's info about loading input image
    /// @tparam CHECKED  true if loader should check for image boundaries
    template <bool CHECKED> struct FDWT97ColumnLoadingInfo {
        /// Loader of pixels from some input image.
        VerticalDWTPixelLoader<float, CHECKED> loader;

        /// Offset of column loaded by loader. (Offset in shared buffer.)
        int offset;
    };


    /// Horizontal 9/7 FDWT on specified lines of transform buffer.
    /// @param lines      number of lines to be transformed
    /// @param firstLine  index of the first line to be transformed
    __device__ void horizontalFDWT97(const int lines, const int firstLine) {
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


    /// Initializes one column of shared transform buffer with 7 input pixels.
    /// Those 7 pixels will not be transformed. Also initializes given loader.
    /// @tparam CHECKED     true if loader should check for image boundaries
    /// @param column       (uninitialized) object for loading input pixels
    /// @param columnIndex  index (not offset!) of the column to be loaded
    ///                     (relative to threadblock's first column)
    /// @param input        pointer to input image in GPU memory
    /// @param sizeX        width of the input image
    /// @param sizeY        height of the input image
    /// @param firstY       index of first row to be loaded from image
    template <bool CHECKED>
    __device__ void initColumn(FDWT97ColumnLoadingInfo<CHECKED> &column,
                               const int columnIndex, const float *const input,
                               const int sizeX, const int sizeY,
                               const int firstY) {
        // get offset of the column with index 'columnIndex'
        column.offset = buffer.getColumnOffset(columnIndex);

        // x-coordinate of the first pixel to be loaded by given loader
        const int firstX = blockIdx.x * WIN_SIZE_X + columnIndex;

        if (blockIdx.y == 0) {
            // topmost block - apply mirroring rules when loading first 7 rows
            column.loader.init(sizeX, sizeY, firstX, firstY);

            // load pixels in mirrored way
            float p0 = column.loader.loadFrom(input);
            float p1 = column.loader.loadFrom(input);
            float p2 = column.loader.loadFrom(input);
            float p3 = column.loader.loadFrom(input);
            float p4 = column.loader.loadFrom(input);
            
            buffer[column.offset + 4 * STRIDE] = p0;
            buffer[column.offset + 3 * STRIDE] = p1;
            buffer[column.offset + 5 * STRIDE] = p1;
            buffer[column.offset + 2 * STRIDE] = p2;
            buffer[column.offset + 6 * STRIDE] = p2;
            buffer[column.offset + 1 * STRIDE] = p3;
            buffer[column.offset + 0 * STRIDE] = p4;

            // reinitialize loader to start with pixel #3 again
            column.loader.init(sizeX, sizeY, firstX, firstY + 3);
        } else {
            // non-topmost row - regular loading:
            column.loader.init(sizeX, sizeY, firstX, firstY - 4);

            // load 7 rows into the transform buffer with loop unrolling
            #pragma unroll
            for (int i = 0; i < 7; i++) {
                buffer[column.offset + i * STRIDE] =
                    column.loader.loadFrom(input);
            }
        }
        // Now, the next pixel, which will be loaded by loader, is pixel #3.
    }


    /// Loads another WIN_SIZE_Y pixels into given column using given loader.
    /// @tparam CHECKED  true if loader should check for image boundaries
    /// @param input     input image to load from
    /// @param column    loader and offset of loaded column in shared buffer
    template <bool CHECKED>
    inline __device__ void
    loadWindowIntoColumn(const float *const input,
                         FDWT97ColumnLoadingInfo<CHECKED> &column) {
        #pragma unroll
        for (int i = 7; i < (7 + WIN_SIZE_Y); i++) {
            buffer[column.offset + i * STRIDE] = column.loader.loadFrom(input);
        }
    }


    /// Main GPU 9/7 FDWT entry point.
    /// @tparam CHECK_LOADS   true if boundaries should be checked when loading
    /// @tparam CHECK_WRITES  true if boundaries should be checked when writing
    /// @param in        input image
    /// @param out       output buffer
    /// @param sizeX     width of the input image
    /// @param sizeY     height of the input image
    /// @param winSteps  number of steps of sliding window
    template <bool CHECK_LOADS, bool CHECK_WRITES>
    __device__ void transform(const float *const in, float *const out,
                              const int sizeX, const int sizeY,
                              const int winSteps) {
        // info about columns loaded by this thread: one main column and
        // possibly
        // one boundary column. (Only some threads load some boundary column.)
        FDWT97ColumnLoadingInfo<CHECK_LOADS> loadedColumn;
        FDWT97ColumnLoadingInfo<CHECK_LOADS> boundaryColumn;

        // Initialize first 7 lines of transform buffer.
        const int firstY = blockIdx.y * WIN_SIZE_Y * winSteps;
        initColumn(loadedColumn, threadIdx.x, in, sizeX, sizeY, firstY);

        // Some threads initialize boundary columns.
        boundaryColumn.offset = 0;
        boundaryColumn.loader.clear();
        if (threadIdx.x < 7) {
            // each thread among first 7 ones gets index of one of boundary
            // columns
            const int colId =
                threadIdx.x + ((threadIdx.x < 3) ? WIN_SIZE_X : -7);

            // Thread initializes offset of the boundary column (in shared
            // buffer),
            // first 7 pixels of the column and a loader for this column.
            initColumn(boundaryColumn, colId, in, sizeX, sizeY, firstY);
        }

        // horizontally transform first 7 rows in all columns
        horizontalFDWT97(7, 0);

        // Index of column handled by this thread. (First half of threads handle
        // even columns and others handle odd columns.)
        const int outColumnIndex = parityIdx<WIN_SIZE_X>();

        // writer of output linear bands - initialize it
        const int firstX = blockIdx.x * WIN_SIZE_X + outColumnIndex;
        VerticalDWTBandWriter<float, CHECK_WRITES> writer;
        writer.init(sizeX, sizeY, firstX, firstY);

        // transform buffer offset of column transformed and saved by this
        // thread
        const int outColumnOffset = buffer.getColumnOffset(outColumnIndex);

        // (Each iteration of this loop assumes that first 7 rows of transform
        // buffer are already loaded with horizontally transformed
        // coefficients.)
        #pragma unroll 2
        for (int w = 0; w < winSteps; w++) {
            // Load another WIN_SIZE_Y lines of thread's column into the buffer.
            loadWindowIntoColumn(in, loadedColumn);

            // some threads also load boundary columns
            if (threadIdx.x < 7) {
                loadWindowIntoColumn(in, boundaryColumn);
            }

            // horizontally transform all newly loaded lines
            horizontalFDWT97(WIN_SIZE_Y, 7);

            // Using 7 registers, remember current values of last 7 rows of
            // transform buffer. These rows are transformed horizontally only
            // and will be used in next iteration.
            float last7Lines[7];
            #pragma unroll
            for (int i = 0; i < 7; i++) {
                last7Lines[i] =
                    buffer[outColumnOffset + (WIN_SIZE_Y + i) * STRIDE];
            }

            // vertically transform all central columns (do not scale yet)
            buffer.forEachVerticalOdd(outColumnOffset,
                                      AddScaledSum(f97Predict1));
            buffer.forEachVerticalEven(outColumnOffset,
                                       AddScaledSum(f97Update1));
            buffer.forEachVerticalOdd(outColumnOffset,
                                      AddScaledSum(f97Predict2));
            buffer.forEachVerticalEven(outColumnOffset,
                                       AddScaledSum(f97Update2));

            // Save all results of current window. Results are in transform
            // buffer
            // at rows from #4 to #(4 + WIN_SIZE_Y). Other rows are invalid now.
            // (They only served as a boundary for vertical FDWT.)
            #pragma unroll
            for (int i = 4; i < (4 + WIN_SIZE_Y); i += 2) {
                const int index = outColumnOffset + i * STRIDE;
                // Write low coefficients from column into low band ...
                writer.writeLowInto(out, buffer[index] * scale97Div);
                // ... and high coeficients into the high band.
                writer.writeHighInto(out, buffer[index + STRIDE] * scale97Mul);
            }

            // Use last 7 remembered lines as first 7 lines for next iteration.
            // As expected, these lines are already horizontally transformed.
            #pragma unroll
            for (int i = 0; i < 7; i++) {
                buffer[outColumnOffset + i * STRIDE] = last7Lines[i];
            }

            // Wait for all writing threads before proceeding to loading new
            // pixels in next iteration. (Not to overwrite those which
            // are not written yet.)
            __syncthreads();
        }
    }


  public:
    /// Runs one of specialized variants of 9/7 FDWT according to distance of
    /// processed pixels to image boudnary. Some variants do not check for
    /// boudnary and thus are slightly faster.
    /// @param in     input image
    /// @param out    output buffer
    /// @param sx     width of the input image
    /// @param sy     height of the input image
    /// @param steps  number of steps of sliding window
    __device__ static void run(const float *const input, float *const output,
                               const int sx, const int sy, const int steps) {
        // object with transform buffer in shared memory
        __shared__ FDWT97<WIN_SIZE_X, WIN_SIZE_Y> fdwt97;

        // Compute limits of this threadblock's block of pixels and use them to
        // determine, whether this threadblock will have to deal with boundary.
        // (3 in next expressions is for radius of impulse response of 9/7
        // FDWT.)
        const int maxX = (blockIdx.x + 1) * WIN_SIZE_X + 3;
        const int maxY = (blockIdx.y + 1) * WIN_SIZE_Y * steps + 3;
        const bool atRightBoudary = maxX >= sx;
        const bool atBottomBoudary = maxY >= sy;

        // Select specialized version of code according to distance of this
        // threadblock's pixels from image boundary.
        if (atBottomBoudary) {
            // near bottom boundary => check both writing and reading
            fdwt97.transform<true, true>(input, output, sx, sy, steps);
        } else if (atRightBoudary) {
            // near right boundary only => check writing only
            fdwt97.transform<false, true>(input, output, sx, sy, steps);
        } else {
            // no nearby boundary => check nothing
            fdwt97.transform<false, false>(input, output, sx, sy, steps);
        }
    }

}; // end of class FDWT97


/// Main GPU 9/7 FDWT entry point.
/// @param input   input image
/// @parma output  output buffer
/// @param sx      width of the input image
/// @param sy      height of the input image
/// @param steps   number of steps of sliding window
template <int WIN_SX, int WIN_SY>
__launch_bounds__(WIN_SX, CTMIN(SHM_SIZE / sizeof(FDWT97<WIN_SX, WIN_SY>),
                                8)) __global__
    void fdwt97Kernel(const float *const __restrict__ input, float *const __restrict__ output,
                      const int sx, const int sy, const int steps) {
    // Excuse me, dear reader of this code - this call have to be here. If you
    // try to simply put contents of following method right here, CUDA compiler
    // (version 3.2) will spit tons of nonsense messy errors ...
    // Hope they will not break it even more in future releases.
    FDWT97<WIN_SX, WIN_SY>::run(input, output, sx, sy, steps);
}


/// Only computes optimal number of sliding window steps,
/// number of threadblocks and then lanches the 9/7 FDWT kernel.
/// @tparam WIN_SX  width of sliding window
/// @tparam WIN_SY  height of sliding window
/// @param in       input image
/// @param out      output buffer
/// @param sx       width of the input image
/// @param sy       height of the input image
template <int WIN_SX, int WIN_SY>
void launchFDWT97Kernel(float *in, float *out, int sx, int sy) {
    // compute optimal number of steps of each sliding window
    const int steps = divRndUp(sy, 15 * WIN_SY);

    // prepare grid size
    dim3 gSize(divRndUp(sx, WIN_SX), divRndUp(sy, WIN_SY * steps));

    // run kernel, possibly measure time and finally check the call
    PERF_BEGIN
    fdwt97Kernel<WIN_SX, WIN_SY><<<gSize, WIN_SX>>>(in, out, sx, sy, steps);
    PERF_END("        FDWT97", sx, sy)
    CudaDWTTester::checkLastKernelCall("FDWT
