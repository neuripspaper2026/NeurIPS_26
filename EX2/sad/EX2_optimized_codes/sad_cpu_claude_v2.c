#include <stdlib.h>
#include <omp.h>
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
  int mb_y, mb_x;
  unsigned int frame_yoff;
  int total_mbs = mb_width * mb_height;

  /* Go to the starting offset in blk_sad */
  blk_sad += SAD_TYPE_7_IX(total_mbs);

  /* Parallelize the outer loop over macroblocks */
  #pragma omp parallel for schedule(dynamic, 4) private(mb_x, frame_yoff)
  for (mb_y = 0; mb_y < mb_height; mb_y++)
    {
      frame_yoff = 256 * mb_width * mb_y;
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
  int width = mb_width * 16;
  int height = mb_height * 16;
  int pos;

  /* Precompute boundary limits */
  int max_width = width - 1;
  int max_height = height - 1;

  /* Each search position */
  pos = 0;
  for (pos_y = -SEARCH_RANGE; pos_y <= SEARCH_RANGE; pos_y++) {
    for (pos_x = -SEARCH_RANGE; pos_x <= SEARCH_RANGE; pos_x++, pos++) {
      int blky, blkx;

      /* Each 4x4 block in the macroblock */
      for (blky = 0; blky < 4; blky++) {
	for (blkx = 0; blkx < 4; blkx++) {
	  int y, x;
	  unsigned short sad = 0;
	  int block_base_x = blkx * 4;
	  int block_base_y = blky * 4;

	  /* Each pixel */
	  for (y = 0; y < 4; y++) {
	    int ref_y_base = frame_y + pos_y + block_base_y + y;
	    int frame_y_offset = (block_base_y + y) * width;
	    
	    /* Clamp ref_y once per row */
	    if (ref_y_base < 0) ref_y_base = 0;
	    if (ref_y_base > max_height) ref_y_base = max_height;
	    int ref_y_offset = ref_y_base * width;

	    for (x = 0; x < 4; x++) {
	      int ref_x = frame_x + pos_x + block_base_x + x;
	      unsigned int a, b;

	      /* Clamp ref_x */
	      if (ref_x < 0) ref_x = 0;
	      if (ref_x > max_width) ref_x = max_width;

	      b = ref[ref_y_offset + ref_x];
	      a = frame[frame_y_offset + block_base_x + x];

	      sad += abs(a - b);
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

  /* Parallelize the outer loop over macroblocks */
  #pragma omp parallel for schedule(static)
  for (macroblock = 0; macroblock < mbs; macroblock++)
    {
      int block_x, block_y;
      unsigned short *x, *y, *z;
      int count;

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

	    for (count = 0; count < MAX_POS; count++) z[count] = x[count] + y[count];
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

	    for (count = 0; count < MAX_POS; count++) z[count] = x[count] + y[count];
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
	    
	    for (count = 0; count < MAX_POS; count++) z[count] = x[count] + y[count];
	  }
      
      /* Block type 3 */
      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + SAD_TYPE_3_IX(mbs) +
	macroblock * SAD_TYPE_3_CT * MAX_POS_PADDED;
      
      for (count = 0; count < MAX_POS; count++) z[count] = x[count] + y[count];

      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED +
	MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + SAD_TYPE_3_IX(mbs) +
	macroblock * SAD_TYPE_3_CT * MAX_POS_PADDED +
	MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) z[count] = x[count] + y[count];

      /* Block type 2 */
      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) z[count] = x[count] + y[count];

      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED +
	2 * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED +
	MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) z[count] = x[count] + y[count];

      /* Block type 1 */
      x = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_1_IX(mbs) +
	macroblock * SAD_TYPE_1_CT * MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) z[count] = x[count] + y[count];
    }
}


