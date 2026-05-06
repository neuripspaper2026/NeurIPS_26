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
  int blky, blkx;
  int y, x;
  
  /* Precompute frame_x and frame_y boundaries */
  int frame_x_min = (frame_x < SEARCH_RANGE) ? (SEARCH_RANGE - frame_x) : 0;
  int frame_x_max = (frame_x + 16 + SEARCH_RANGE > width) ? (width - frame_x - 16) : SEARCH_RANGE;
  int frame_y_min = (frame_y < SEARCH_RANGE) ? (SEARCH_RANGE - frame_y) : 0;
  int frame_y_max = (frame_y + 16 + SEARCH_RANGE > height) ? (height - frame_y - 16) : SEARCH_RANGE;

  /* Each search position */
  pos = 0;
  for (pos_y = -SEARCH_RANGE; pos_y <= SEARCH_RANGE; pos_y++) {
    int need_clamp_y = (pos_y < -frame_y_min || pos_y > frame_y_max);
    
    for (pos_x = -SEARCH_RANGE; pos_x <= SEARCH_RANGE; pos_x++, pos++) {
      int need_clamp_x = (pos_x < -frame_x_min || pos_x > frame_x_max);
      int need_clamp = need_clamp_x || need_clamp_y;

      /* Each 4x4 block in the macroblock */
      for (blky = 0; blky < 4; blky++) {
	int blky_offset = blky * 4;
	for (blkx = 0; blkx < 4; blkx++) {
	  int blkx_offset = blkx * 4;
	  unsigned int sad = 0;

	  if (!need_clamp) {
	    /* Fast path: no boundary clamping needed */
	    int ref_base_y = frame_y + pos_y + blky_offset;
	    int ref_base_x = frame_x + pos_x + blkx_offset;
	    
	    for (y = 0; y < 4; y++) {
	      int ref_row_offset = (ref_base_y + y) * width + ref_base_x;
	      int frame_row_offset = (blky_offset + y) * width + blkx_offset;
	      
	      for (x = 0; x < 4; x++) {
		unsigned int b = ref[ref_row_offset + x];
		unsigned int a = frame[frame_row_offset + x];
		sad += (a > b) ? (a - b) : (b - a);
	      }
	    }
	  } else {
	    /* Slow path: boundary clamping required */
	    for (y = 0; y < 4; y++) {
	      for (x = 0; x < 4; x++) {
		int ref_x = frame_x + pos_x + blkx_offset + x;
		int ref_y = frame_y + pos_y + blky_offset + y;
		
		if (ref_x < 0) ref_x = 0;
		else if (ref_x >= width) ref_x = width - 1;
		
		if (ref_y < 0) ref_y = 0;
		else if (ref_y >= height) ref_y = height - 1;

		unsigned int b = ref[ref_y * width + ref_x];
		unsigned int a = frame[(blky_offset + y) * width + (blkx_offset + x)];
		sad += (a > b) ? (a - b) : (b - a);
	      }
	    }
	  }

	  /* Save the SAD */
	  macroblock_sad[MAX_POS_PADDED*(blky_offset + blkx) + pos] = sad;
	}
      }
    }
  }
}

void larger_sads(unsigned short *sads, int mbs)
{
  int macroblock;
  int block_x, block_y;
  unsigned short *x, *y;
  unsigned short *z;
  int count;

  #pragma omp parallel for schedule(static) private(block_x, block_y, x, y, z, count)
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


