#include <stdlib.h>
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
  for (mb_y = 0, frame_yoff = 0;
       mb_y < mb_height;
       mb_y++, frame_yoff += 256 * mb_width) {
    const int mb_y_off = mb_y * mb_width;
    const int frame_y_base = mb_y * 16;
    for (mb_x = 0; mb_x < mb_width; mb_x++) {
      sad4_one_macroblock(
        blk_sad + (mb_y_off + mb_x) * (SAD_TYPE_7_CT * MAX_POS_PADDED),
        frame + frame_yoff + (mb_x * 16),
        ref,
        frame_y_base,
        mb_x * 16,
        mb_width,
        mb_height
      );
    }
  }
}

static void
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
  int pos = 0;			/* search position */

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
	  unsigned short sad = 0;

	  /* Each pixel */
	  int y;
	  for (y = 0; y < 4; y++) {
            const int fy = blk_y_off + y;
            const unsigned short *frame_row = frame + fy * width;
            int ref_y = base_ref_y + blk_y_off + y;
            if (ref_y < 0) ref_y = 0;
            else if (ref_y >= height) ref_y = height - 1;
            const int ref_row_off = ref_y * width;

	    int x;
	    for (x = 0; x < 4; x++) {
	      int ref_x = base_ref_x + blk_x_off + x;
	      unsigned int a, b;

	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      b = ref[ref_row_off + ref_x];
	      a = frame_row[blk_x_off + x];

	      sad += (unsigned short)(a > b ? (a - b) : (b - a));
	    }
	  }

	  /* Save the SAD */
	  macroblock_sad[(MAX_POS_PADDED * (4 * blky + blkx)) + pos] = sad;
	}
      }
    }
  }
}

void larger_sads(unsigned short *sads, int mbs)
{
  int macroblock;
  int block_x, block_y;
  unsigned short *x, *y;	/* inputs to vector addition */
  unsigned short *z;		/* output of vector addition */
  int count;

  const int sad7_ix = SAD_TYPE_7_IX(mbs);
  const int sad6_ix = SAD_TYPE_6_IX(mbs);
  const int sad5_ix = SAD_TYPE_5_IX(mbs);
  const int sad4_ix = SAD_TYPE_4_IX(mbs);
  const int sad3_ix = SAD_TYPE_3_IX(mbs);
  const int sad2_ix = SAD_TYPE_2_IX(mbs);
  const int sad1_ix = SAD_TYPE_1_IX(mbs);

  const int sad7_ct = SAD_TYPE_7_CT;
  const int sad6_ct = SAD_TYPE_6_CT;
  const int sad5_ct = SAD_TYPE_5_CT;
  const int sad4_ct = SAD_TYPE_4_CT;
  const int sad3_ct = SAD_TYPE_3_CT;
  const int sad2_ct = SAD_TYPE_2_CT;
  const int sad1_ct = SAD_TYPE_1_CT;

  for (macroblock = 0; macroblock < mbs; macroblock++) {
    const int mb7_off = sad7_ix + macroblock * sad7_ct * MAX_POS_PADDED;
    const int mb6_off = sad6_ix + macroblock * sad6_ct * MAX_POS_PADDED;
    const int mb5_off = sad5_ix + macroblock * sad5_ct * MAX_POS_PADDED;
    const int mb4_off = sad4_ix + macroblock * sad4_ct * MAX_POS_PADDED;
    const int mb3_off = sad3_ix + macroblock * sad3_ct * MAX_POS_PADDED;
    const int mb2_off = sad2_ix + macroblock * sad2_ct * MAX_POS_PADDED;
    const int mb1_off = sad1_ix + macroblock * sad1_ct * MAX_POS_PADDED;

      /* Block type 6 */
      for (block_y = 0; block_y < 2; block_y++) {
	for (block_x = 0; block_x < 4; block_x++) {
	  x = sads + mb7_off +
	    (8 * block_y + block_x) * MAX_POS_PADDED;
	  y = x + 4 * MAX_POS_PADDED;
	  z = sads + mb6_off +
	    (4 * block_y + block_x) * MAX_POS_PADDED;

	  for (count = 0; count < MAX_POS; count++)
            z[count] = (unsigned short)(x[count] + y[count]);
	}
      }

      /* Block type 5 */
      for (block_y = 0; block_y < 4; block_y++) {
	for (block_x = 0; block_x < 2; block_x++) {
	  x = sads + mb7_off +
	    (4 * block_y + 2 * block_x) * MAX_POS_PADDED;
	  y = x + MAX_POS_PADDED;
	  z = sads + mb5_off +
	    (2 * block_y + block_x) * MAX_POS_PADDED;

	  for (count = 0; count < MAX_POS; count++)
            z[count] = (unsigned short)(x[count] + y[count]);
	}
      }

      /* Block type 4 */
      for (block_y = 0; block_y < 2; block_y++) {
	for (block_x = 0; block_x < 2; block_x++) {
	  x = sads + mb5_off +
	    (4 * block_y + block_x) * MAX_POS_PADDED;
	  y = x + 2 * MAX_POS_PADDED;
	  z = sads + mb4_off +
	    (2 * block_y + block_x) * MAX_POS_PADDED;

	  for (count = 0; count < MAX_POS; count++)
            z[count] = (unsigned short)(x[count] + y[count]);
	}
      }

      /* Block type 3 */
      x = sads + mb4_off;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + mb3_off;

      for (count = 0; count < MAX_POS; count++)
        z[count] = (unsigned short)(x[count] + y[count]);

      x = sads + mb4_off +
	MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + mb3_off +
	MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++)
        z[count] = (unsigned short)(x[count] + y[count]);

      /* Block type 2 */
      x = sads + mb4_off;
      y = x + MAX_POS_PADDED;
      z = sads + mb2_off;

      for (count = 0; count < MAX_POS; count++)
        z[count] = (unsigned short)(x[count] + y[count]);

      x = sads + mb4_off +
	2 * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + mb2_off +
	MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++)
        z[count] = (unsigned short)(x[count] + y[count]);

      /* Block type 1 */
      x = sads + mb2_off;
      y = x + MAX_POS_PADDED;
      z = sads + mb1_off;

      for (count = 0; count < MAX_POS; count++)
        z[count] = (unsigned short)(x[count] + y[count]);
    }
}


