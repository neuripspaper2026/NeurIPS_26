#include <stdlib.h>
#include "sad.h"

static inline unsigned short uabs_diff(unsigned int a, unsigned int b) {
  return (unsigned short)(a > b ? a - b : b - a);
}

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
  const int mb_count = mb_width * mb_height;
  const int mb_sad_stride = SAD_TYPE_7_CT * MAX_POS_PADDED;

  /* Go to the starting offset in blk_sad */
  blk_sad += SAD_TYPE_7_IX(mb_count);

  /* For each block */
  for (mb_y = 0, frame_yoff = 0;
       mb_y < mb_height;
       mb_y++, frame_yoff += 256 * mb_width) {
    unsigned short *blk_row = blk_sad + mb_y * mb_width * mb_sad_stride;
    unsigned short *frame_row = frame + frame_yoff;
    const int frame_y_base = mb_y * 16;
    for (mb_x = 0; mb_x < mb_width; mb_x++) {
      sad4_one_macroblock(
        blk_row + mb_x * mb_sad_stride,
        frame_row + mb_x * 16,
        ref,
        frame_y_base,
        mb_x * 16,
        mb_width,
        mb_height
      );
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
  int pos = 0;                  /* search position */

  /* Precompute clipped reference range per direction to minimize branches */
  const int min_ref_x = 0;
  const int max_ref_x = width  - 1;
  const int min_ref_y = 0;
  const int max_ref_y = height - 1;

  /* precompute frame row pointers for the 16x16 macroblock (in 4x4 layout) */
  unsigned short *frame_rows[16];
  {
    int r;
    for (r = 0; r < 16; ++r) {
      frame_rows[r] = frame + r * width;
    }
  }

  /* Each search position */
  for (pos_y = -SEARCH_RANGE; pos_y <= SEARCH_RANGE; pos_y++) {
    const int base_ref_y = frame_y + pos_y;
    for (pos_x = -SEARCH_RANGE; pos_x <= SEARCH_RANGE; pos_x++, pos++) {
      const int base_ref_x = frame_x + pos_x;
      int blky, blkx;

      /* Each 4x4 block in the macroblock */
      for (blky = 0; blky < 4; blky++) {
        const int blk_y_off = blky * 4;
        for (blkx = 0; blkx < 4; blkx++) {
          const int blk_x_off = blkx * 4;
          int y, x;
          unsigned int sad = 0;

          /* Each pixel */
          for (y = 0; y < 4; y++) {
            const int ref_y_unclipped = base_ref_y + blk_y_off + y;
            int ref_y = ref_y_unclipped;
            if (ref_y < min_ref_y) ref_y = min_ref_y;
            else if (ref_y > max_ref_y) ref_y = max_ref_y;

            unsigned short *frame_row = frame_rows[blk_y_off + y];
            unsigned short *ref_row = ref + ref_y * width;

            for (x = 0; x < 4; x++) {
              const int ref_x_unclipped = base_ref_x + blk_x_off + x;
              int ref_x = ref_x_unclipped;
              if (ref_x < min_ref_x) ref_x = min_ref_x;
              else if (ref_x > max_ref_x) ref_x = max_ref_x;

              const unsigned int b = ref_row[ref_x];
              const unsigned int a = frame_row[blk_x_off + x];

              sad += uabs_diff(a, b);
            }
          }

          /* Save the SAD */
          macroblock_sad[MAX_POS_PADDED * (4 * blky + blkx) + pos] =
            (unsigned short)sad;
        }
      }
    }
  }
}

void larger_sads(unsigned short *sads, int mbs)
{
  int macroblock;
  int block_x, block_y;
  unsigned short *x, *y;        /* inputs to vector addition */
  unsigned short *z;            /* output of vector addition */
  int count;

  const int sad7_ix = SAD_TYPE_7_IX(mbs);
  const int sad6_ix = SAD_TYPE_6_IX(mbs);
  const int sad5_ix = SAD_TYPE_5_IX(mbs);
  const int sad4_ix = SAD_TYPE_4_IX(mbs);
  const int sad3_ix = SAD_TYPE_3_IX(mbs);
  const int sad2_ix = SAD_TYPE_2_IX(mbs);
  const int sad1_ix = SAD_TYPE_1_IX(mbs);

  const int sad7_stride = SAD_TYPE_7_CT * MAX_POS_PADDED;
  const int sad6_stride = SAD_TYPE_6_CT * MAX_POS_PADDED;
  const int sad5_stride = SAD_TYPE_5_CT * MAX_POS_PADDED;
  const int sad4_stride = SAD_TYPE_4_CT * MAX_POS_PADDED;
  const int sad3_stride = SAD_TYPE_3_CT * MAX_POS_PADDED;
  const int sad2_stride = SAD_TYPE_2_CT * MAX_POS_PADDED;
  const int sad1_stride = SAD_TYPE_1_CT * MAX_POS_PADDED;

  for (macroblock = 0; macroblock < mbs; macroblock++) {
    unsigned short *sad7_base = sads + sad7_ix + macroblock * sad7_stride;
    unsigned short *sad6_base = sads + sad6_ix + macroblock * sad6_stride;
    unsigned short *sad5_base = sads + sad5_ix + macroblock * sad5_stride;
    unsigned short *sad4_base = sads + sad4_ix + macroblock * sad4_stride;
    unsigned short *sad3_base = sads + sad3_ix + macroblock * sad3_stride;
    unsigned short *sad2_base = sads + sad2_ix + macroblock * sad2_stride;
    unsigned short *sad1_base = sads + sad1_ix + macroblock * sad1_stride;

    /* Block type 6 */
    for (block_y = 0; block_y < 2; block_y++) {
      const int by8 = 8 * block_y;
      const int by4 = 4 * block_y;
      for (block_x = 0; block_x < 4; block_x++) {
        x = sad7_base + (by8 + block_x) * MAX_POS_PADDED;
        y = x + 4 * MAX_POS_PADDED;
        z = sad6_base + (by4 + block_x) * MAX_POS_PADDED;

        for (count = 0; count < MAX_POS; count++) {
          *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
        }
      }
    }

    /* Block type 5 */
    for (block_y = 0; block_y < 4; block_y++) {
      const int by4 = 4 * block_y;
      const int by2 = 2 * block_y;
      for (block_x = 0; block_x < 2; block_x++) {
        x = sad7_base + (by4 + 2 * block_x) * MAX_POS_PADDED;
        y = x + MAX_POS_PADDED;
        z = sad5_base + (by2 + block_x) * MAX_POS_PADDED;

        for (count = 0; count < MAX_POS; count++) {
          *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
        }
      }
    }

    /* Block type 4 */
    for (block_y = 0; block_y < 2; block_y++) {
      const int by4 = 4 * block_y;
      const int by2 = 2 * block_y;
      for (block_x = 0; block_x < 2; block_x++) {
        x = sad5_base + (by4 + block_x) * MAX_POS_PADDED;
        y = x + 2 * MAX_POS_PADDED;
        z = sad4_base + (by2 + block_x) * MAX_POS_PADDED;

        for (count = 0; count < MAX_POS; count++) {
          *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
        }
      }
    }

    /* Block type 3 */
    x = sad4_base;
    y = x + 2 * MAX_POS_PADDED;
    z = sad3_base;

    for (count = 0; count < MAX_POS; count++) {
      *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
    }

    x = sad4_base + MAX_POS_PADDED;
    y = x + 2 * MAX_POS_PADDED;
    z = sad3_base + MAX_POS_PADDED;

    for (count = 0; count < MAX_POS; count++) {
      *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
    }

    /* Block type 2 */
    x = sad4_base;
    y = x + MAX_POS_PADDED;
    z = sad2_base;

    for (count = 0; count < MAX_POS; count++) {
      *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
    }

    x = sad4_base + 2 * MAX_POS_PADDED;
    y = x + MAX_POS_PADDED;
    z = sad2_base + MAX_POS_PADDED;

    for (count = 0; count < MAX_POS; count++) {
      *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
    }

    /* Block type 1 */
    x = sad2_base;
    y = x + MAX_POS_PADDED;
    z = sad1_base;

    for (count = 0; count < MAX_POS; count++) {
      *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
    }
  }
}


