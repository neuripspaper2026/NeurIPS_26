#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "sad.h"

static void sad4_one_macroblock(unsigned short *blk_sad,
                                unsigned short *frame,
                                unsigned short *ref,
                                int frame_y,
                                int frame_x,
                                int mb_width,
                                int mb_height);

void sad4_cpu(unsigned short *blk_sad,
              unsigned short *frame,
              unsigned short *ref,
              int mb_width,
              int mb_height)
{
  int mb_x, mb_y;
  unsigned int frame_yoff;

  /* Go to the starting offset in blk_sad */
  blk_sad += SAD_TYPE_7_IX(mb_width * mb_height);

  /* For each block */
#ifdef _OPENMP
#pragma omp parallel for private(mb_x, frame_yoff) schedule(static)
#endif
  for (mb_y = 0; mb_y < mb_height; mb_y++)
    {
      frame_yoff = (unsigned int)mb_y * 256u * (unsigned int)mb_width;
      for (mb_x = 0; mb_x < mb_width; mb_x++)
        {
          sad4_one_macroblock
            (blk_sad + (mb_y * mb_width + mb_x) * (SAD_TYPE_7_CT * MAX_POS_PADDED),
             frame + frame_yoff + mb_x * 16,
             ref,
             mb_y * 16,
             mb_x * 16,
             mb_width,
             mb_height);
        }
    }
}

void
sad4_one_macroblock(unsigned short *macroblock_sad,
                    unsigned short *frame,
                    unsigned short *ref,
                    int frame_y,
                    int frame_x,
                    int mb_width,
                    int mb_height)
{
  int pos_x, pos_y;
  const int width  = mb_width * 16;
  const int height = mb_height * 16;
  int pos;                  /* search position */

  /* Each search position */
  pos = 0;
  for (pos_y = -SEARCH_RANGE; pos_y <= SEARCH_RANGE; pos_y++) {
    const int base_ref_y = frame_y + pos_y;
    for (pos_x = -SEARCH_RANGE; pos_x <= SEARCH_RANGE; pos_x++, pos++) {
      int blky, blkx;

      /* Each 4x4 block in the macroblock */
      for (blky = 0; blky < 4; blky++) {
        const int blk_y_offset = blky * 4;
        for (blkx = 0; blkx < 4; blkx++) {
          const int blk_x_offset = blkx * 4;
          int y, x;
          unsigned short sad = 0;

          /* Each pixel */
          for (y = 0; y < 4; y++) {
            const int frame_row_index = (blk_y_offset + y) * width;
            int ref_y = base_ref_y + blk_y_offset + y;
            /* Clip ref_y to image boundaries */
            if (ref_y < 0) ref_y = 0;
            else if (ref_y >= height) ref_y = height - 1;
            const int ref_row_index = ref_y * width;

            for (x = 0; x < 4; x++) {
              int ref_x = frame_x + pos_x + blk_x_offset + x;
              unsigned int a, b;

              /* Clip ref_x to image boundary */
              if (ref_x < 0) ref_x = 0;
              else if (ref_x >= width) ref_x = width - 1;

              b = ref[ref_row_index + ref_x];
              a = frame[frame_row_index + blk_x_offset + x];

              sad += (unsigned short)abs((int)a - (int)b);
            }
          }

          /* Save the SAD */
          macroblock_sad[MAX_POS_PADDED*(4*blky+blkx) + pos] = sad;
        }
      }
    }
  }
}

void larger_sads(unsigned short *sads, int mbs)
{
  int macroblock;
  int block_x, block_y;
  unsigned short *x, *y;    /* inputs to vector addition */
  unsigned short *z;        /* output of vector addition */
  int count;

#ifdef _OPENMP
#pragma omp parallel for private(block_x, block_y, x, y, z, count) schedule(static)
#endif
  for (macroblock = 0; macroblock < mbs; macroblock++)
    {
      /* Block type 6 */
      for (block_y = 0; block_y < 2; block_y++)
        for (block_x = 0; block_x < 4; block_x++)
          {
            x = sads + SAD_TYPE_7_IX(mbs) +
              macroblock * SAD_TYPE_7_CT * MAX_POS_PADDED +
              (8 * block_y + block_x) * MAX_POS_PADDED;
            y = x + 4 * MAX_POS_PADDED;
            z = sads + SAD_TYPE_6_IX(mbs) +
              macroblock * SAD_TYPE_6_CT * MAX_POS_PADDED +
              (4 * block_y + block_x) * MAX_POS_PADDED;

#pragma clang loop vectorize(enable)
#pragma GCC ivdep
            for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));

          }

      /* Block type 5 */
      for (block_y = 0; block_y < 4; block_y++)
        for (block_x = 0; block_x < 2; block_x++)
          {
            x = sads + SAD_TYPE_7_IX(mbs) +
              macroblock * SAD_TYPE_7_CT * MAX_POS_PADDED +
              (4 * block_y + 2 * block_x) * MAX_POS_PADDED;
            y = x + MAX_POS_PADDED;
            z = sads + SAD_TYPE_5_IX(mbs) +
              macroblock * SAD_TYPE_6_CT * MAX_POS_PADDED +
              (2 * block_y + block_x) * MAX_POS_PADDED;

#pragma clang loop vectorize(enable)
#pragma GCC ivdep
            for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));

          }

      /* Block type 4 */
      for (block_y = 0; block_y < 2; block_y++)
        for (block_x = 0; block_x < 2; block_x++)
          {
            x = sads + SAD_TYPE_5_IX(mbs) +
              macroblock * SAD_TYPE_5_CT * MAX_POS_PADDED +
              (4 * block_y + block_x) * MAX_POS_PADDED;
            y = x + 2 * MAX_POS_PADDED;
            z = sads + SAD_TYPE_4_IX(mbs) +
              macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED +
              (2 * block_y + block_x) * MAX_POS_PADDED;

#pragma clang loop vectorize(enable)
#pragma GCC ivdep
            for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));

          }

      /* Block type 3 */
      x = sads + SAD_TYPE_4_IX(mbs) +
        macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + SAD_TYPE_3_IX(mbs) +
        macroblock * SAD_TYPE_3_CT * MAX_POS_PADDED;

#pragma clang loop vectorize(enable)
#pragma GCC ivdep
      for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));

      x = sads + SAD_TYPE_4_IX(mbs) +
        macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED +
        MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + SAD_TYPE_3_IX(mbs) +
        macroblock * SAD_TYPE_3_CT * MAX_POS_PADDED +
        MAX_POS_PADDED;

#pragma clang loop vectorize(enable)
#pragma GCC ivdep
      for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));

      /* Block type 2 */
      x = sads + SAD_TYPE_4_IX(mbs) +
        macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_2_IX(mbs) +
        macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;

#pragma clang loop vectorize(enable)
#pragma GCC ivdep
      for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));

      x = sads + SAD_TYPE_4_IX(mbs) +
        macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED +
        2 * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_2_IX(mbs) +
        macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED +
        MAX_POS_PADDED;

#pragma clang loop vectorize(enable)
#pragma GCC ivdep
      for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));

      /* Block type 1 */
      x = sads + SAD_TYPE_2_IX(mbs) +
        macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_1_IX(mbs) +
        macroblock * SAD_TYPE_1_CT * MAX_POS_PADDED;

#pragma clang loop vectorize(enable)
#pragma GCC ivdep
      for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
    }
}


