#include <stdlib.h>
#include "sad.h"

static void sad4_one_macroblock(unsigned short *blk_sad,
				const unsigned short *frame,
				const unsigned short *ref,
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
  const int mb_size = mb_width * mb_height;
  const int mb_stride = SAD_TYPE_7_CT * MAX_POS_PADDED;
  const int frame_row_stride = 256 * mb_width;
  int mb_x, mb_y;
  unsigned int frame_yoff = 0;

  /* Go to the starting offset in blk_sad */
  blk_sad += SAD_TYPE_7_IX(mb_size);

  /* For each block */
  for (mb_y = 0; mb_y < mb_height; mb_y++, frame_yoff += frame_row_stride) {
    const int mb_y_off = mb_y * mb_width;
    const int frame_y_pix = mb_y * 16;
    for (mb_x = 0; mb_x < mb_width; mb_x++) {
      sad4_one_macroblock(
        blk_sad + (mb_y_off + mb_x) * mb_stride,
        frame + frame_yoff + mb_x * 16,
        ref,
        frame_y_pix,
        mb_x * 16,
        mb_width,
        mb_height
      );
    }
  }
}

static void
sad4_one_macroblock(unsigned short *macroblock_sad,
		    const unsigned short *frame,
		    const unsigned short *ref,
		    int frame_y,
		    int frame_x,
		    int mb_width,
		    int mb_height)
{
  const int width  = mb_width * 16;
  const int height = mb_height * 16;
  const int frame_block_row_stride = width;       /* distance between rows in frame */
  const int ref_stride = width;                   /* distance between rows in ref   */
  const int blocks_per_mb = 16;                   /* 4x4 blocks in a macroblock */
  const int block_side = 4;
  int pos = 0;
  int pos_y, pos_x;

  /* Precompute base frame offsets for the 16 blocks (4x4) within macroblock */
  unsigned int frame_block_base[blocks_per_mb];
  {
    int blky, blkx, idx = 0;
    for (blky = 0; blky < 4; blky++) {
      const unsigned int row_off = (unsigned int)(blky * block_side) * (unsigned int)frame_block_row_stride;
      for (blkx = 0; blkx < 4; blkx++, idx++) {
        frame_block_base[idx] = row_off + (unsigned int)(blkx * block_side);
      }
    }
  }

  /* Each search position */
  for (pos_y = -SEARCH_RANGE; pos_y <= SEARCH_RANGE; pos_y++) {
    const int base_ref_y = frame_y + pos_y;
    for (pos_x = -SEARCH_RANGE; pos_x <= SEARCH_RANGE; pos_x++, pos++) {
      const int base_ref_x = frame_x + pos_x;
      int blky, blkx, blk_idx = 0;

      /* Each 4x4 block in the macroblock */
      for (blky = 0; blky < 4; blky++) {
        const int blk_y_off = blky * block_side;
        for (blkx = 0; blkx < 4; blkx++, blk_idx++) {
          const int blk_x_off = blkx * block_side;
          unsigned short sad = 0;
          int y, x;

          /* Each pixel */
          for (y = 0; y < block_side; y++) {
            const unsigned int frame_row_off = frame_block_base[blk_idx] + (unsigned int)y * (unsigned int)frame_block_row_stride;

            /* compute unclipped reference y once per row */
            int ref_y = base_ref_y + blk_y_off + y;
            if (ref_y < 0) {
              ref_y = 0;
            } else if (ref_y >= height) {
              ref_y = height - 1;
            }
            const unsigned int ref_row_off = (unsigned int)ref_y * (unsigned int)ref_stride;

            for (x = 0; x < block_side; x++) {
              /* Get reference pixel coordinate, clipped to image boundary */
              int ref_x = base_ref_x + blk_x_off + x;
              if (ref_x < 0) {
                ref_x = 0;
              } else if (ref_x >= width) {
                ref_x = width - 1;
              }

              const unsigned int ref_idx = ref_row_off + (unsigned int)ref_x;
              const unsigned int frame_idx = frame_row_off + (unsigned int)x;

              const unsigned int b = ref[ref_idx];
              const unsigned int a = frame[frame_idx];

              sad = (unsigned short)(sad + (unsigned short)abs((int)a - (int)b));
            }
          }

          /* Save the SAD */
          macroblock_sad[MAX_POS_PADDED * blk_idx + pos] = sad;
        }
      }
    }
  }
}

void larger_sads(unsigned short *sads, int mbs)
{
  int macroblock;

  for (macroblock = 0; macroblock < mbs; macroblock++) {
    int block_x, block_y;
    unsigned short *x, *y, *z;
    int count;

    /* Block type 6 */
    {
      const int base7 = SAD_TYPE_7_IX(mbs) +
                        macroblock * SAD_TYPE_7_CT * MAX_POS_PADDED;
      const int base6 = SAD_TYPE_6_IX(mbs) +
                        macroblock * SAD_TYPE_6_CT * MAX_POS_PADDED;

      for (block_y = 0; block_y < 2; block_y++) {
        const int row_off7 = base7 + block_y * 8 * MAX_POS_PADDED;
        const int row_off6 = base6 + block_y * 4 * MAX_POS_PADDED;
        for (block_x = 0; block_x < 4; block_x++) {
          x = sads + row_off7 + block_x * MAX_POS_PADDED;
          y = x + 4 * MAX_POS_PADDED;
          z = sads + row_off6 + block_x * MAX_POS_PADDED;

          for (count = 0; count < MAX_POS; count++) {
            *z++ = (unsigned short)((unsigned int)*x++ + (unsigned int)*y++);
          }
        }
      }
    }

    /* Block type 5 */
    {
      const int base7 = SAD_TYPE_7_IX(mbs) +
                        macroblock * SAD_TYPE_7_CT * MAX_POS_PADDED;
      const int base5 = SAD_TYPE_5_IX(mbs) +
                        macroblock * SAD_TYPE_6_CT * MAX_POS_PADDED;

      for (block_y = 0; block_y < 4; block_y++) {
        const int row_off7 = base7 + block_y * 4 * MAX_POS_PADDED;
        const int row_off5 = base5 + block_y * 2 * MAX_POS_PADDED;
        for (block_x = 0; block_x < 2; block_x++) {
          x = sads + row_off7 + (2 * block_x) * MAX_POS_PADDED;
          y = x + MAX_POS_PADDED;
          z = sads + row_off5 + block_x * MAX_POS_PADDED;

          for (count = 0; count < MAX_POS; count++) {
            *z++ = (unsigned short)((unsigned int)*x++ + (unsigned int)*y++);
          }
        }
      }
    }

    /* Block type 4 */
    {
      const int base5 = SAD_TYPE_5_IX(mbs) +
                        macroblock * SAD_TYPE_5_CT * MAX_POS_PADDED;
      const int base4 = SAD_TYPE_4_IX(mbs) +
                        macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;

      for (block_y = 0; block_y < 2; block_y++) {
        const int row_off5 = base5 + block_y * 4 * MAX_POS_PADDED;
        const int row_off4 = base4 + block_y * 2 * MAX_POS_PADDED;
        for (block_x = 0; block_x < 2; block_x++) {
          x = sads + row_off5 + block_x * MAX_POS_PADDED;
          y = x + 2 * MAX_POS_PADDED;
          z = sads + row_off4 + block_x * MAX_POS_PADDED;

          for (count = 0; count < MAX_POS; count++) {
            *z++ = (unsigned short)((unsigned int)*x++ + (unsigned int)*y++);
          }
        }
      }
    }

    /* Block type 3 */
    {
      const int base4 = SAD_TYPE_4_IX(mbs) +
                        macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
      const int base3 = SAD_TYPE_3_IX(mbs) +
                        macroblock * SAD_TYPE_3_CT * MAX_POS_PADDED;

      x = sads + base4;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + base3;

      for (count = 0; count < MAX_POS; count++) {
        *z++ = (unsigned short)((unsigned int)*x++ + (unsigned int)*y++);
      }

      x = sads + base4 + MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + base3 + MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) {
        *z++ = (unsigned short)((unsigned int)*x++ + (unsigned int)*y++);
      }
    }

    /* Block type 2 */
    {
      const int base4 = SAD_TYPE_4_IX(mbs) +
                        macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
      const int base2 = SAD_TYPE_2_IX(mbs) +
                        macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;

      x = sads + base4;
      y = x + MAX_POS_PADDED;
      z = sads + base2;

      for (count = 0; count < MAX_POS; count++) {
        *z++ = (unsigned short)((unsigned int)*x++ + (unsigned int)*y++);
      }

      x = sads + base4 + 2 * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + base2 + MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) {
        *z++ = (unsigned short)((unsigned int)*x++ + (unsigned int)*y++);
      }
    }

    /* Block type 1 */
    {
      const int base2 = SAD_TYPE_2_IX(mbs) +
                        macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;
      const int base1 = SAD_TYPE_1_IX(mbs) +
                        macroblock * SAD_TYPE_1_CT * MAX_POS_PADDED;

      x = sads + base2;
      y = x + MAX_POS_PADDED;
      z = sads + base1;

      for (count = 0; count < MAX_POS; count++) {
        *z++ = (unsigned short)((unsigned int)*x++ + (unsigned int)*y++);
      }
    }
  }
}


