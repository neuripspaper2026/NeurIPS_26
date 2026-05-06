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
  int mb_x, mb_y;
  unsigned int frame_yoff;

  /* Go to the starting offset in blk_sad */
  blk_sad += SAD_TYPE_7_IX(mb_width * mb_height);

  /* For each block */
  for (mb_y = 0, frame_yoff = 0;
       mb_y < mb_height;
       mb_y++, frame_yoff += 256 * mb_width)
    {
      unsigned short *frame_row = frame + frame_yoff;
      for (mb_x = 0; mb_x < mb_width; mb_x++)
	{
	  sad4_one_macroblock
	    (blk_sad + (mb_y * mb_width + mb_x) * (SAD_TYPE_7_CT * MAX_POS_PADDED),
	     frame_row + (mb_x << 4),
	     ref,
	     mb_y << 4,
	     mb_x << 4,
	     mb_width,
	     mb_height);
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
  int pos_x, pos_y;
  const int width  = mb_width  << 4;
  const int height = mb_height << 4;
  int pos = 0;			/* search position */

  /* Precompute clamped base offsets for macroblock pixels in reference frame
   * for all 16x16 positions to avoid recomputing per search position. */
  int base_x[16];
  int base_y[16];
  {
    int blky, blkx, y, x, idx = 0;
    for (blky = 0; blky < 4; blky++) {
      const int by = frame_y + (blky << 2);
      for (blkx = 0; blkx < 4; blkx++) {
	const int bx = frame_x + (blkx << 2);
	for (y = 0; y < 4; y++) {
	  int ry = by + y;
	  if (ry < 0) ry = 0;
	  else if (ry >= height) ry = height - 1;
	  base_y[idx] = ry;
	  for (x = 0; x < 4; x++, idx++) {
	    int rx = bx + x;
	    if (rx < 0) rx = 0;
	    else if (rx >= width) rx = width - 1;
	    base_x[idx] = rx;
	  }
	}
      }
    }
  }

  /* Each search position */
  for (pos_y = -SEARCH_RANGE; pos_y <= SEARCH_RANGE; pos_y++) {
    for (pos_x = -SEARCH_RANGE; pos_x <= SEARCH_RANGE; pos_x++, pos++) {
      int blky, blkx;

      /* Each 4x4 block in the macroblock */
      int pix_idx = 0;
      for (blky = 0; blky < 4; blky++) {
	for (blkx = 0; blkx < 4; blkx++) {
	  int y, x;
	  unsigned short sad = 0;

	  /* Each pixel */
	  for (y = 0; y < 4; y++) {
	    const int row_off = (blky * 4 + y) * width;
	    for (x = 0; x < 4; x++, pix_idx++) {
	      const unsigned int a = frame[row_off + (blkx * 4 + x)];

	      int ref_x = base_x[pix_idx] + pos_x;
	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      int ref_y = base_y[pix_idx] + pos_y;
	      if (ref_y < 0) ref_y = 0;
	      else if (ref_y >= height) ref_y = height - 1;

	      const unsigned int b = ref[ref_y * width + ref_x];

	      sad += (unsigned short) (a > b ? (a - b) : (b - a));
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

  const int sad7_blk_stride = SAD_TYPE_7_CT * MAX_POS_PADDED;
  const int sad6_blk_stride = SAD_TYPE_6_CT * MAX_POS_PADDED;
  const int sad5_blk_stride = SAD_TYPE_5_CT * MAX_POS_PADDED;
  const int sad4_blk_stride = SAD_TYPE_4_CT * MAX_POS_PADDED;
  const int sad3_blk_stride = SAD_TYPE_3_CT * MAX_POS_PADDED;
  const int sad2_blk_stride = SAD_TYPE_2_CT * MAX_POS_PADDED;
  const int sad1_blk_stride = SAD_TYPE_1_CT * MAX_POS_PADDED;

  for (macroblock = 0; macroblock < mbs; macroblock++)
    {
      const int mb7_off = sad7_ix + macroblock * sad7_blk_stride;
      const int mb6_off = sad6_ix + macroblock * sad6_blk_stride;
      const int mb5_off = sad5_ix + macroblock * sad5_blk_stride;
      const int mb4_off = sad4_ix + macroblock * sad4_blk_stride;
      const int mb3_off = sad3_ix + macroblock * sad3_blk_stride;
      const int mb2_off = sad2_ix + macroblock * sad2_blk_stride;
      const int mb1_off = sad1_ix + macroblock * sad1_blk_stride;

      /* Block type 6 */
      for (block_y = 0; block_y < 2; block_y++)
	for (block_x = 0; block_x < 4; block_x++)
	  {
	    x = sads + mb7_off +
	      (8 * block_y + block_x) * MAX_POS_PADDED;
	    y = x + 4 * MAX_POS_PADDED;
	    z = sads + mb6_off +
	      (4 * block_y + block_x) * MAX_POS_PADDED;

	    for (count = 0; count < MAX_POS; count++)
	      *z++ = (unsigned short)((unsigned int)*x++ + (unsigned int)*y++);
	  }

      /* Block type 5 */
      for (block_y = 0; block_y < 4; block_y++)
	for (block_x = 0; block_x < 2; block_x++)
	  {
	    x = sads + mb7_off +
	      (4 * block_y + 2 * block_x) * MAX_POS_PADDED;
	    y = x + MAX_POS_PADDED;
	    z = sads + mb5_off +
	      (2 * block_y + block_x) * MAX_POS_PADDED;

	    for (count = 0; count < MAX_POS; count++)
	      *z++ = (unsigned short)((unsigned int)*x++ + (unsigned int)*y++);
	  }

      /* Block type 4 */
      for (block_y = 0; block_y < 2; block_y++)
	for (block_x = 0; block_x < 2; block_x++)
	  {
	    x = sads + mb5_off +
	      (4 * block_y + block_x) * MAX_POS_PADDED;
	    y = x + 2 * MAX_POS_PADDED;
	    z = sads + mb4_off +
	      (2 * block_y + block_x) * MAX_POS_PADDED;
	    
	    for (count = 0; count < MAX_POS; count++)
	      *z++ = (unsigned short)((unsigned int)*x++ + (unsigned int)*y++);
	  }
      
      /* Block type 3 */
      x = sads + mb4_off;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + mb3_off;
      
      for (count = 0; count < MAX_POS; count++)
	*z++ = (unsigned short)((unsigned int)*x++ + (unsigned int)*y++);

      x = sads + mb4_off + MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + mb3_off + MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++)
	*z++ = (unsigned short)((unsigned int)*x++ + (unsigned int)*y++);

      /* Block type 2 */
      x = sads + mb4_off;
      y = x + MAX_POS_PADDED;
      z = sads + mb2_off;

      for (count = 0; count < MAX_POS; count++)
	*z++ = (unsigned short)((unsigned int)*x++ + (unsigned int)*y++);

      x = sads + mb4_off + 2 * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + mb2_off + MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++)
	*z++ = (unsigned short)((unsigned int)*x++ + (unsigned int)*y++);

      /* Block type 1 */
      x = sads + mb2_off;
      y = x + MAX_POS_PADDED;
      z = sads + mb1_off;

      for (count = 0; count < MAX_POS; count++)
	*z++ = (unsigned short)((unsigned int)*x++ + (unsigned int)*y++);
    }
}


