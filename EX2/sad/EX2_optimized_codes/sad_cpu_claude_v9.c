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

  /* Parallelize over macroblocks */
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

  /* Precompute frame_x and frame_y boundaries */
  int frame_x_min = (frame_x < SEARCH_RANGE) ? (SEARCH_RANGE - frame_x) : 0;
  int frame_x_max = (frame_x + 16 + SEARCH_RANGE > width) ? (width - frame_x - 16) : SEARCH_RANGE;
  int frame_y_min = (frame_y < SEARCH_RANGE) ? (SEARCH_RANGE - frame_y) : 0;
  int frame_y_max = (frame_y + 16 + SEARCH_RANGE > height) ? (height - frame_y - 16) : SEARCH_RANGE;

  /* Each search position */
  pos = 0;
  for (pos_y = -SEARCH_RANGE; pos_y <= SEARCH_RANGE; pos_y++) {
    int ref_y_base = frame_y + pos_y;
    int need_y_clamp = (pos_y < frame_y_min || pos_y > frame_y_max);

    for (pos_x = -SEARCH_RANGE; pos_x <= SEARCH_RANGE; pos_x++, pos++) {
      int ref_x_base = frame_x + pos_x;
      int need_x_clamp = (pos_x < frame_x_min || pos_x > frame_x_max);
      int need_clamp = need_x_clamp || need_y_clamp;

      int blky, blkx;

      /* Each 4x4 block in the macroblock */
      for (blky = 0; blky < 4; blky++) {
	int blky4 = blky * 4;
	for (blkx = 0; blkx < 4; blkx++) {
	  int blkx4 = blkx * 4;
	  unsigned int sad = 0;

	  if (!need_clamp) {
	    /* Fast path: no boundary clipping needed */
	    for (int y = 0; y < 4; y++) {
	      int ref_y = ref_y_base + blky4 + y;
	      int ref_row_offset = ref_y * width;
	      int frame_row_offset = (blky4 + y) * width;
	      
	      for (int x = 0; x < 4; x++) {
		int ref_x = ref_x_base + blkx4 + x;
		unsigned int b = ref[ref_row_offset + ref_x];
		unsigned int a = frame[frame_row_offset + blkx4 + x];
		sad += (a > b) ? (a - b) : (b - a);
	      }
	    }
	  } else {
	    /* Slow path: boundary clipping required */
	    for (int y = 0; y < 4; y++) {
	      int ref_y = ref_y_base + blky4 + y;
	      if (ref_y < 0) ref_y = 0;
	      if (ref_y >= height) ref_y = height - 1;
	      int ref_row_offset = ref_y * width;
	      int frame_row_offset = (blky4 + y) * width;

	      for (int x = 0; x < 4; x++) {
		int ref_x = ref_x_base + blkx4 + x;
		if (ref_x < 0) ref_x = 0;
		if (ref_x >= width) ref_x = width - 1;

		unsigned int b = ref[ref_row_offset + ref_x];
		unsigned int a = frame[frame_row_offset + blkx4 + x];
		sad += (a > b) ? (a - b) : (b - a);
	      }
	    }
	  }

	  /* Save the SAD */
	  macroblock_sad[MAX_POS_PADDED*(blky4+blkx) + pos] = (unsigned short)sad;
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

	    for (count = 0; count < MAX_POS; count++) *z++ = *x++ + *y++;
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

	    for (count = 0; count < MAX_POS; count++) *z++ = *x++ + *y++;
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
	    
	    for (count = 0; count < MAX_POS; count++) *z++ = *x++ + *y++;
	  }
      
      /* Block type 3 */
      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + SAD_TYPE_3_IX(mbs) +
	macroblock * SAD_TYPE_3_CT * MAX_POS_PADDED;
      
      for (count = 0; count < MAX_POS; count++) *z++ = *x++ + *y++;

      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED +
	MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + SAD_TYPE_3_IX(mbs) +
	macroblock * SAD_TYPE_3_CT * MAX_POS_PADDED +
	MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) *z++ = *x++ + *y++;

      /* Block type 2 */
      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) *z++ = *x++ + *y++;

      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED +
	2 * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED +
	MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) *z++ = *x++ + *y++;

      /* Block type 1 */
      x = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_1_IX(mbs) +
	macroblock * SAD_TYPE_1_CT * MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) *z++ = *x++ + *y++;
    }
}


