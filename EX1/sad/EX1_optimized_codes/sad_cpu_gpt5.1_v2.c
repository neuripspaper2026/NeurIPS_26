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

  const int mb_line_stride = 256 * mb_width;
  const int mb_sad_stride = SAD_TYPE_7_CT * MAX_POS_PADDED;

  /* For each block */
  for (mb_y = 0, frame_yoff = 0;
       mb_y < mb_height;
       mb_y++, frame_yoff += mb_line_stride)
    {
      const int mb_y16 = mb_y * 16;
      const int mb_y_index = mb_y * mb_width;

      for (mb_x = 0; mb_x < mb_width; mb_x++)
	{
	  sad4_one_macroblock
	    (blk_sad + (mb_y_index + mb_x) * mb_sad_stride,
	     frame + frame_yoff + (mb_x << 4),
	     ref,
	     mb_y16,
	     mb_x << 4,
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
  const int width = mb_width * 16;
  const int height = mb_height * 16;
  int pos = 0;			/* search position */

  /* Precompute unclipped base coordinates for each 4x4 block */
  int blk_base_x[4];
  int blk_base_y[4];
  int blky, blkx;
  for (blky = 0; blky < 4; blky++) {
    blk_base_y[blky] = frame_y + blky * 4;
  }
  for (blkx = 0; blkx < 4; blkx++) {
    blk_base_x[blkx] = frame_x + blkx * 4;
  }

  /* Each search position */
  for (pos_y = -SEARCH_RANGE; pos_y <= SEARCH_RANGE; pos_y++) {
    const int off_y = pos_y;
    for (pos_x = -SEARCH_RANGE; pos_x <= SEARCH_RANGE; pos_x++, pos++) {
      const int off_x = pos_x;

      /* Each 4x4 block in the macroblock */
      for (blky = 0; blky < 4; blky++) {
	for (blkx = 0; blkx < 4; blkx++) {
	  int y, x;
	  unsigned short sad = 0;

	  const int base_y = blk_base_y[blky] + off_y;
	  const int base_x = blk_base_x[blkx] + off_x;

	  /* Each pixel */
	  for (y = 0; y < 4; y++) {
	    const int fy = (blky * 4 + y) * width;
	    const int ry_unclipped = base_y + y;
	    int ry = ry_unclipped;
	    if (ry < 0) ry = 0;
	    else if (ry >= height) ry = height - 1;
	    const int ry_width = ry * width;

	    for (x = 0; x < 4; x++) {
	      int rx_unclipped = base_x + x;
	      int rx = rx_unclipped;
	      if (rx < 0) rx = 0;
	      else if (rx >= width) rx = width - 1;

	      const unsigned int b = ref[ry_width + rx];
	      const unsigned int a = frame[fy + (blkx * 4 + x)];

	      sad += (unsigned short)abs((int)a - (int)b);
	    }
	  }

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

  const int sad7_blk_span = SAD_TYPE_7_CT * MAX_POS_PADDED;
  const int sad6_blk_span = SAD_TYPE_6_CT * MAX_POS_PADDED;
  const int sad5_blk_span = SAD_TYPE_5_CT * MAX_POS_PADDED;
  const int sad4_blk_span = SAD_TYPE_4_CT * MAX_POS_PADDED;
  const int sad3_blk_span = SAD_TYPE_3_CT * MAX_POS_PADDED;
  const int sad2_blk_span = SAD_TYPE_2_CT * MAX_POS_PADDED;
  const int sad1_blk_span = SAD_TYPE_1_CT * MAX_POS_PADDED;

  for (macroblock = 0; macroblock < mbs; macroblock++)
    {
      const int mb7_off = sad7_ix + macroblock * sad7_blk_span;
      const int mb6_off = sad6_ix + macroblock * sad6_blk_span;
      const int mb5_off = sad5_ix + macroblock * sad5_blk_span;
      const int mb4_off = sad4_ix + macroblock * sad4_blk_span;
      const int mb3_off = sad3_ix + macroblock * sad3_blk_span;
      const int mb2_off = sad2_ix + macroblock * sad2_blk_span;
      const int mb1_off = sad1_ix + macroblock * sad1_blk_span;

      /* Block type 6 */
      for (block_y = 0; block_y < 2; block_y++)
	for (block_x = 0; block_x < 4; block_x++)
	  {
	    const int idx7 = mb7_off + (8 * block_y + block_x) * MAX_POS_PADDED;
	    x = sads + idx7;
	    y = x + 4 * MAX_POS_PADDED;
	    const int idx6 = mb6_off + (4 * block_y + block_x) * MAX_POS_PADDED;
	    z = sads + idx6;

	    for (count = 0; count < MAX_POS; count++)
	      z[count] = x[count] + y[count];
	  }

      /* Block type 5 */
      for (block_y = 0; block_y < 4; block_y++)
	for (block_x = 0; block_x < 2; block_x++)
	  {
	    const int idx7 = mb7_off + (4 * block_y + 2 * block_x) * MAX_POS_PADDED;
	    x = sads + idx7;
	    y = x + MAX_POS_PADDED;
	    const int idx5 = mb5_off + (2 * block_y + block_x) * MAX_POS_PADDED;
	    z = sads + idx5;

	    for (count = 0; count < MAX_POS; count++)
	      z[count] = x[count] + y[count];
	  }

      /* Block type 4 */
      for (block_y = 0; block_y < 2; block_y++)
	for (block_x = 0; block_x < 2; block_x++)
	  {
	    const int idx5 = mb5_off + (4 * block_y + block_x) * MAX_POS_PADDED;
	    x = sads + idx5;
	    y = x + 2 * MAX_POS_PADDED;
	    const int idx4 = mb4_off + (2 * block_y + block_x) * MAX_POS_PADDED;
	    z = sads + idx4;

	    for (count = 0; count < MAX_POS; count++)
	      z[count] = x[count] + y[count];
	  }

      /* Block type 3 */
      x = sads + mb4_off;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + mb3_off;

      for (count = 0; count < MAX_POS; count++)
	z[count] = x[count] + y[count];

      x = sads + mb4_off + MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + mb3_off + MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++)
	z[count] = x[count] + y[count];

      /* Block type 2 */
      x = sads + mb4_off;
      y = x + MAX_POS_PADDED;
      z = sads + mb2_off;

      for (count = 0; count < MAX_POS; count++)
	z[count] = x[count] + y[count];

      x = sads + mb4_off + 2 * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + mb2_off + MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++)
	z[count] = x[count] + y[count];

      /* Block type 1 */
      x = sads + mb2_off;
      y = x + MAX_POS_PADDED;
      z = sads + mb1_off;

      for (count = 0; count < MAX_POS; count++)
	z[count] = x[count] + y[count];
    }
}


