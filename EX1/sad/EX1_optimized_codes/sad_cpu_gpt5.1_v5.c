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

  const int mb_stride = 256 * mb_width;

  /* For each block */
  for (mb_y = 0, frame_yoff = 0;
       mb_y < mb_height;
       mb_y++, frame_yoff += mb_stride)
    {
      const int mb_y16 = mb_y * 16;
      for (mb_x = 0; mb_x < mb_width; mb_x++)
	{
	  sad4_one_macroblock
	    (blk_sad + (mb_y * mb_width + mb_x) * (SAD_TYPE_7_CT * MAX_POS_PADDED),
	     frame + frame_yoff + (mb_x * 16),
	     ref,
	     mb_y16,
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
  const int width  = mb_width  * 16;
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
	for (blkx = 0; blkx < 4; blkx++) {
	  int y, x;
	  unsigned short sad = 0;
	  const int blk_y_off = blky * 4;
	  const int blk_x_off = blkx * 4;
	  const int ref_blk_y = base_ref_y + blk_y_off;
	  const int ref_blk_x = base_ref_x + blk_x_off;

	  /* Each pixel */
	  for (y = 0; y < 4; y++) {
	    const int frame_row = (blk_y_off + y) * width;
	    const int ref_y_unclipped = ref_blk_y + y;
	    int ref_y_idx = ref_y_unclipped;
	    if (ref_y_idx < 0) ref_y_idx = 0;
	    else if (ref_y_idx >= height) ref_y_idx = height - 1;
	    const int ref_y_row = ref_y_idx * width;

	    for (x = 0; x < 4; x++) {
	      const int frame_idx = frame_row + blk_x_off + x;

	      int ref_x_unclipped = ref_blk_x + x;
	      if (ref_x_unclipped < 0) ref_x_unclipped = 0;
	      else if (ref_x_unclipped >= width) ref_x_unclipped = width - 1;

	      const unsigned int b = ref[ref_y_row + ref_x_unclipped];
	      const unsigned int a = frame[frame_idx];

	      sad += (unsigned short)((a > b) ? (a - b) : (b - a));
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

  const int sad7_blk   = SAD_TYPE_7_CT * MAX_POS_PADDED;
  const int sad6_blk   = SAD_TYPE_6_CT * MAX_POS_PADDED;
  const int sad5_blk   = SAD_TYPE_5_CT * MAX_POS_PADDED;
  const int sad4_blk   = SAD_TYPE_4_CT * MAX_POS_PADDED;
  const int sad3_blk   = SAD_TYPE_3_CT * MAX_POS_PADDED;
  const int sad2_blk   = SAD_TYPE_2_CT * MAX_POS_PADDED;
  const int maxpos_blk = MAX_POS_PADDED;

  for (macroblock = 0; macroblock < mbs; macroblock++)
    {
      const int mb7_off = macroblock * sad7_blk;
      const int mb6_off = macroblock * sad6_blk;
      const int mb5_off = macroblock * sad5_blk;
      const int mb4_off = macroblock * sad4_blk;
      const int mb3_off = macroblock * sad3_blk;
      const int mb2_off = macroblock * sad2_blk;

      /* Block type 6 */
      for (block_y = 0; block_y < 2; block_y++)
	for (block_x = 0; block_x < 4; block_x++)
	  {
	    x = sads + sad7_ix + mb7_off +
	      (8 * block_y + block_x) * maxpos_blk;
	    y = x + 4 * maxpos_blk;
	    z = sads + sad6_ix + mb6_off +
	      (4 * block_y + block_x) * maxpos_blk;

	    for (count = 0; count < MAX_POS; count++) {
	      *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
	    }
	  }

      /* Block type 5 */
      for (block_y = 0; block_y < 4; block_y++)
	for (block_x = 0; block_x < 2; block_x++)
	  {
	    x = sads + sad7_ix + mb7_off +
	      (4 * block_y + 2 * block_x) * maxpos_blk;
	    y = x + maxpos_blk;
	    z = sads + sad5_ix + mb6_off +
	      (2 * block_y + block_x) * maxpos_blk;

	    for (count = 0; count < MAX_POS; count++) {
	      *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
	    }
	  }

      /* Block type 4 */
      for (block_y = 0; block_y < 2; block_y++)
	for (block_x = 0; block_x < 2; block_x++)
	  {
	    x = sads + sad5_ix + mb5_off +
	      (4 * block_y + block_x) * maxpos_blk;
	    y = x + 2 * maxpos_blk;
	    z = sads + sad4_ix + mb4_off +
	      (2 * block_y + block_x) * maxpos_blk;
	    
	    for (count = 0; count < MAX_POS; count++) {
	      *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
	    }
	  }
      
      /* Block type 3 */
      x = sads + sad4_ix + mb4_off;
      y = x + 2 * maxpos_blk;
      z = sads + sad3_ix + mb3_off;
      
      for (count = 0; count < MAX_POS; count++) {
	*z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
      }

      x = sads + sad4_ix + mb4_off + maxpos_blk;
      y = x + 2 * maxpos_blk;
      z = sads + sad3_ix + mb3_off + maxpos_blk;

      for (count = 0; count < MAX_POS; count++) {
	*z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
      }

      /* Block type 2 */
      x = sads + sad4_ix + mb4_off;
      y = x + maxpos_blk;
      z = sads + sad2_ix + mb2_off;

      for (count = 0; count < MAX_POS; count++) {
	*z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
      }

      x = sads + sad4_ix + mb4_off + 2 * maxpos_blk;
      y = x + maxpos_blk;
      z = sads + sad2_ix + mb2_off + maxpos_blk;

      for (count = 0; count < MAX_POS; count++) {
	*z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
      }

      /* Block type 1 */
      x = sads + sad2_ix + mb2_off;
      y = x + maxpos_blk;
      z = sads + sad1_ix + macroblock * SAD_TYPE_1_CT * MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) {
	*z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
      }
    }
}


