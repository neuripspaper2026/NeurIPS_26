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
  for (mb_y = 0; mb_y < mb_height; mb_y++) {
    frame_yoff = (unsigned int)mb_y * 256u * (unsigned int)mb_width;

    for (mb_x = 0; mb_x < mb_width; mb_x++) {
      sad4_one_macroblock(
          blk_sad + (mb_y * mb_width + mb_x) * (SAD_TYPE_7_CT * MAX_POS_PADDED),
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
  int pos;                   /* search position */

  const int bx_base = frame_x - SEARCH_RANGE;
  const int by_base = frame_y - SEARCH_RANGE;
  const int pos_x_min = -SEARCH_RANGE;
  const int pos_x_max = SEARCH_RANGE;
  const int pos_y_min = -SEARCH_RANGE;
  const int pos_y_max = SEARCH_RANGE;

  /* Each search position */
  pos = 0;
  for (pos_y = pos_y_min; pos_y <= pos_y_max; pos_y++) {
    const int base_y4 = by_base + pos_y * 4;
    for (pos_x = pos_x_min; pos_x <= pos_x_max; pos_x++, pos++) {

      int blky, blkx;

      /* Each 4x4 block in the macroblock */
      for (blky = 0; blky < 4; blky++) {
        const int blk_y  = blky * 4;
        const int fy_row = blk_y * width;
        const int ref_y0 = base_y4 + blky * 4;

        for (blkx = 0; blkx < 4; blkx++) {
          const int blk_x = blkx * 4;

          /* Manually unrolled 4x4 SAD */
          unsigned int sad0 = 0, sad1 = 0, sad2 = 0, sad3 = 0;

          int ry = ref_y0;
          if (ry < 0) ry = 0;
          else if (ry >= height) ry = height - 1;

          {
            const int fy   = fy_row + blk_x;
            const int ryw  = ry * width;
            const int rx0b = bx_base + pos_x * 4 + blk_x;

            int rx0 = rx0b + 0;
            int rx1 = rx0b + 1;
            int rx2 = rx0b + 2;
            int rx3 = rx0b + 3;

            if (rx0 < 0) rx0 = 0;
            else if (rx0 >= width) rx0 = width - 1;
            if (rx1 < 0) rx1 = 0;
            else if (rx1 >= width) rx1 = width - 1;
            if (rx2 < 0) rx2 = 0;
            else if (rx2 >= width) rx2 = width - 1;
            if (rx3 < 0) rx3 = 0;
            else if (rx3 >= width) rx3 = width - 1;

            const unsigned int a0 = frame[fy + 0];
            const unsigned int a1 = frame[fy + 1];
            const unsigned int a2 = frame[fy + 2];
            const unsigned int a3 = frame[fy + 3];

            const unsigned int b0 = ref[ryw + rx0];
            const unsigned int b1 = ref[ryw + rx1];
            const unsigned int b2 = ref[ryw + rx2];
            const unsigned int b3 = ref[ryw + rx3];

            sad0 += (a0 > b0) ? (a0 - b0) : (b0 - a0);
            sad0 += (a1 > b1) ? (a1 - b1) : (b1 - a1);
            sad0 += (a2 > b2) ? (a2 - b2) : (b2 - a2);
            sad0 += (a3 > b3) ? (a3 - b3) : (b3 - a3);
          }

          ry = ref_y0 + 1;
          if (ry < 0) ry = 0;
          else if (ry >= height) ry = height - 1;

          {
            const int fy   = fy_row + width + blk_x;
            const int ryw  = ry * width;
            const int rx0b = bx_base + pos_x * 4 + blk_x;

            int rx0 = rx0b + 0;
            int rx1 = rx0b + 1;
            int rx2 = rx0b + 2;
            int rx3 = rx0b + 3;

            if (rx0 < 0) rx0 = 0;
            else if (rx0 >= width) rx0 = width - 1;
            if (rx1 < 0) rx1 = 0;
            else if (rx1 >= width) rx1 = width - 1;
            if (rx2 < 0) rx2 = 0;
            else if (rx2 >= width) rx2 = width - 1;
            if (rx3 < 0) rx3 = 0;
            else if (rx3 >= width) rx3 = width - 1;

            const unsigned int a0 = frame[fy + 0];
            const unsigned int a1 = frame[fy + 1];
            const unsigned int a2 = frame[fy + 2];
            const unsigned int a3 = frame[fy + 3];

            const unsigned int b0 = ref[ryw + rx0];
            const unsigned int b1 = ref[ryw + rx1];
            const unsigned int b2 = ref[ryw + rx2];
            const unsigned int b3 = ref[ryw + rx3];

            sad1 += (a0 > b0) ? (a0 - b0) : (b0 - a0);
            sad1 += (a1 > b1) ? (a1 - b1) : (b1 - a1);
            sad1 += (a2 > b2) ? (a2 - b2) : (b2 - a2);
            sad1 += (a3 > b3) ? (a3 - b3) : (b3 - a3);
          }

          ry = ref_y0 + 2;
          if (ry < 0) ry = 0;
          else if (ry >= height) ry = height - 1;

          {
            const int fy   = fy_row + (width << 1) + blk_x;
            const int ryw  = ry * width;
            const int rx0b = bx_base + pos_x * 4 + blk_x;

            int rx0 = rx0b + 0;
            int rx1 = rx0b + 1;
            int rx2 = rx0b + 2;
            int rx3 = rx0b + 3;

            if (rx0 < 0) rx0 = 0;
            else if (rx0 >= width) rx0 = width - 1;
            if (rx1 < 0) rx1 = 0;
            else if (rx1 >= width) rx1 = width - 1;
            if (rx2 < 0) rx2 = 0;
            else if (rx2 >= width) rx2 = width - 1;
            if (rx3 < 0) rx3 = 0;
            else if (rx3 >= width) rx3 = width - 1;

            const unsigned int a0 = frame[fy + 0];
            const unsigned int a1 = frame[fy + 1];
            const unsigned int a2 = frame[fy + 2];
            const unsigned int a3 = frame[fy + 3];

            const unsigned int b0 = ref[ryw + rx0];
            const unsigned int b1 = ref[ryw + rx1];
            const unsigned int b2 = ref[ryw + rx2];
            const unsigned int b3 = ref[ryw + rx3];

            sad2 += (a0 > b0) ? (a0 - b0) : (b0 - a0);
            sad2 += (a1 > b1) ? (a1 - b1) : (b1 - a1);
            sad2 += (a2 > b2) ? (a2 - b2) : (b2 - a2);
            sad2 += (a3 > b3) ? (a3 - b3) : (b3 - a3);
          }

          ry = ref_y0 + 3;
          if (ry < 0) ry = 0;
          else if (ry >= height) ry = height - 1;

          {
            const int fy   = fy_row + width * 3 + blk_x;
            const int ryw  = ry * width;
            const int rx0b = bx_base + pos_x * 4 + blk_x;

            int rx0 = rx0b + 0;
            int rx1 = rx0b + 1;
            int rx2 = rx0b + 2;
            int rx3 = rx0b + 3;

            if (rx0 < 0) rx0 = 0;
            else if (rx0 >= width) rx0 = width - 1;
            if (rx1 < 0) rx1 = 0;
            else if (rx1 >= width) rx1 = width - 1;
            if (rx2 < 0) rx2 = 0;
            else if (rx2 >= width) rx2 = width - 1;
            if (rx3 < 0) rx3 = 0;
            else if (rx3 >= width) rx3 = width - 1;

            const unsigned int a0 = frame[fy + 0];
            const unsigned int a1 = frame[fy + 1];
            const unsigned int a2 = frame[fy + 2];
            const unsigned int a3 = frame[fy + 3];

            const unsigned int b0 = ref[ryw + rx0];
            const unsigned int b1 = ref[ryw + rx1];
            const unsigned int b2 = ref[ryw + rx2];
            const unsigned int b3 = ref[ryw + rx3];

            sad3 += (a0 > b0) ? (a0 - b0) : (b0 - a0);
            sad3 += (a1 > b1) ? (a1 - b1) : (b1 - a1);
            sad3 += (a2 > b2) ? (a2 - b2) : (b2 - a2);
            sad3 += (a3 > b3) ? (a3 - b3) : (b3 - a3);
          }

          const unsigned short sad =
              (unsigned short)(sad0 + sad1 + sad2 + sad3);

          /* Save the SAD */
          macroblock_sad[MAX_POS_PADDED * (4 * blky + blkx) + pos] = sad;
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
  for (macroblock = 0; macroblock < mbs; macroblock++) {
    /* Block type 6 */
    for (block_y = 0; block_y < 2; block_y++) {
      for (block_x = 0; block_x < 4; block_x++) {
        x = sads + SAD_TYPE_7_IX(mbs) +
            macroblock * SAD_TYPE_7_CT * MAX_POS_PADDED +
            (8 * block_y + block_x) * MAX_POS_PADDED;
        y = x + 4 * MAX_POS_PADDED;
        z = sads + SAD_TYPE_6_IX(mbs) +
            macroblock * SAD_TYPE_6_CT * MAX_POS_PADDED +
            (4 * block_y + block_x) * MAX_POS_PADDED;

        for (count = 0; count < MAX_POS; count++) {
          *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
        }
      }
    }

    /* Block type 5 */
    for (block_y = 0; block_y < 4; block_y++) {
      for (block_x = 0; block_x < 2; block_x++) {
        x = sads + SAD_TYPE_7_IX(mbs) +
            macroblock * SAD_TYPE_7_CT * MAX_POS_PADDED +
            (4 * block_y + 2 * block_x) * MAX_POS_PADDED;
        y = x + MAX_POS_PADDED;
        z = sads + SAD_TYPE_5_IX(mbs) +
            macroblock * SAD_TYPE_6_CT * MAX_POS_PADDED +
            (2 * block_y + block_x) * MAX_POS_PADDED;

        for (count = 0; count < MAX_POS; count++) {
          *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
        }
      }
    }

    /* Block type 4 */
    for (block_y = 0; block_y < 2; block_y++) {
      for (block_x = 0; block_x < 2; block_x++) {
        x = sads + SAD_TYPE_5_IX(mbs) +
            macroblock * SAD_TYPE_5_CT * MAX_POS_PADDED +
            (4 * block_y + block_x) * MAX_POS_PADDED;
        y = x + 2 * MAX_POS_PADDED;
        z = sads + SAD_TYPE_4_IX(mbs) +
            macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED +
            (2 * block_y + block_x) * MAX_POS_PADDED;

        for (count = 0; count < MAX_POS; count++) {
          *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
        }
      }
    }

    /* Block type 3 */
    x = sads + SAD_TYPE_4_IX(mbs) +
        macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
    y = x + 2 * MAX_POS_PADDED;
    z = sads + SAD_TYPE_3_IX(mbs) +
        macroblock * SAD_TYPE_3_CT * MAX_POS_PADDED;

    for (count = 0; count < MAX_POS; count++) {
      *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
    }

    x = sads + SAD_TYPE_4_IX(mbs) +
        macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED +
        MAX_POS_PADDED;
    y = x + 2 * MAX_POS_PADDED;
    z = sads + SAD_TYPE_3_IX(mbs) +
        macroblock * SAD_TYPE_3_CT * MAX_POS_PADDED +
        MAX_POS_PADDED;

    for (count = 0; count < MAX_POS; count++) {
      *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
    }

    /* Block type 2 */
    x = sads + SAD_TYPE_4_IX(mbs) +
        macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
    y = x + MAX_POS_PADDED;
    z = sads + SAD_TYPE_2_IX(mbs) +
        macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;

    for (count = 0; count < MAX_POS; count++) {
      *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
    }

    x = sads + SAD_TYPE_4_IX(mbs) +
        macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED +
        2 * MAX_POS_PADDED;
    y = x + MAX_POS_PADDED;
    z = sads + SAD_TYPE_2_IX(mbs) +
        macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED +
        MAX_POS_PADDED;

    for (count = 0; count < MAX_POS; count++) {
      *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
    }

    /* Block type 1 */
    x = sads + SAD_TYPE_2_IX(mbs) +
        macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;
    y = x + MAX_POS_PADDED;
    z = sads + SAD_TYPE_1_IX(mbs) +
        macroblock * SAD_TYPE_1_CT * MAX_POS_PADDED;

    for (count = 0; count < MAX_POS; count++) {
      *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
    }
  }
}


