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
  const int mbs = mb_width * mb_height;
  const int macroblock_stride = SAD_TYPE_7_CT * MAX_POS_PADDED;
  blk_sad += SAD_TYPE_7_IX(mbs);

  /* For each block */
  for (mb_y = 0, frame_yoff = 0;
       mb_y < mb_height;
       mb_y++, frame_yoff += 256U * (unsigned int)mb_width)
    {
      unsigned short *row_blk_sad =
        blk_sad + (mb_y * mb_width) * macroblock_stride;
      const int frame_y = mb_y * 16;

      for (mb_x = 0; mb_x < mb_width; mb_x++)
	{
	  const int frame_x = mb_x * 16;
	  sad4_one_macroblock
	    (row_blk_sad + mb_x * macroblock_stride,
	     frame + frame_yoff + (unsigned int)frame_x,
	     ref,
	     frame_y,
	     frame_x,
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
  int pos;			/* search position */

  /* Precompute block-relative frame rows */
  unsigned short const *frame_row[16];
  {
    unsigned short *base = frame;
    int r;
    for (r = 0; r < 16; ++r) {
      frame_row[r] = base + r * width;
    }
  }

  /* Each search position */
  pos = 0;
  for (pos_y = -SEARCH_RANGE; pos_y <= SEARCH_RANGE; pos_y++) {
    const int base_ref_y = frame_y + pos_y;
    for (pos_x = -SEARCH_RANGE; pos_x <= SEARCH_RANGE; pos_x++, pos++) {
      const int base_ref_x = frame_x + pos_x;
      int blky, blkx;

      /* Each 4x4 block in the macroblock */
      for (blky = 0; blky < 4; blky++) {
	for (blkx = 0; blkx < 4; blkx++) {
	  int y, x;
	  unsigned int sad = 0;
	  const int blk_off_y = blky * 4;
	  const int blk_off_x = blkx * 4;

	  /* Each pixel */
	  for (y = 0; y < 4; y++) {
	    const int fy = blk_off_y + y;
	    unsigned short const *frame_line = frame_row[fy];
	    const int ref_base_y = base_ref_y + fy;
	    int ref_y;
	    if (ref_base_y < 0) ref_y = 0;
	    else if (ref_base_y >= height) ref_y = height - 1;
	    else ref_y = ref_base_y;
	    unsigned short const *ref_line = ref + (size_t)ref_y * (size_t)width;

	    for (x = 0; x < 4; x++) {
	      const int fx = blk_off_x + x;
	      const int ref_base_x = base_ref_x + fx;
	      int ref_x;
	      if (ref_base_x < 0) ref_x = 0;
	      else if (ref_base_x >= width) ref_x = width - 1;
	      else ref_x = ref_base_x;

	      const unsigned int a = frame_line[fx];
	      const unsigned int b = ref_line[ref_x];

	      sad += (a > b) ? (a - b) : (b - a);
	    }
	  }

	  /* Save the SAD */
	  macroblock_sad[(4 * blky + blkx) * MAX_POS_PADDED + pos] =
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
  unsigned short *x, *y;	/* inputs to vector addition */
  unsigned short *z;		/* output of vector addition */
  int count;

  const int stride7 = SAD_TYPE_7_CT * MAX_POS_PADDED;
  const int stride6 = SAD_TYPE_6_CT * MAX_POS_PADDED;
  const int stride5 = SAD_TYPE_5_CT * MAX_POS_PADDED;
  const int stride4 = SAD_TYPE_4_CT * MAX_POS_PADDED;
  const int stride3 = SAD_TYPE_3_CT * MAX_POS_PADDED;
  const int stride2 = SAD_TYPE_2_CT * MAX_POS_PADDED;

  const int base7 = SAD_TYPE_7_IX(mbs);
  const int base6 = SAD_TYPE_6_IX(mbs);
  const int base5 = SAD_TYPE_5_IX(mbs);
  const int base4 = SAD_TYPE_4_IX(mbs);
  const int base3 = SAD_TYPE_3_IX(mbs);
  const int base2 = SAD_TYPE_2_IX(mbs);
  const int base1 = SAD_TYPE_1_IX(mbs);

  for (macroblock = 0; macroblock < mbs; macroblock++)
    {
      const int mb_off7 = base7 + macroblock * stride7;
      const int mb_off6 = base6 + macroblock * stride6;
      const int mb_off5 = base5 + macroblock * stride5;
      const int mb_off4 = base4 + macroblock * stride4;
      const int mb_off3 = base3 + macroblock * stride3;
      const int mb_off2 = base2 + macroblock * stride2;
      const int mb_off1 = base1 + macroblock * SAD_TYPE_1_CT * MAX_POS_PADDED;

      /* Block type 6 */
      for (block_y = 0; block_y < 2; block_y++)
	for (block_x = 0; block_x < 4; block_x++)
	  {
	    x = sads + mb_off7 +
	      (8 * block_y + block_x) * MAX_POS_PADDED;
	    y = x + 4 * MAX_POS_PADDED;
	    z = sads + mb_off6 +
	      (4 * block_y + block_x) * MAX_POS_PADDED;

	    for (count = 0; count < MAX_POS; count++) {
	      *z = (unsigned short)((unsigned int)*x + (unsigned int)*y);
	      ++z; ++x; ++y;
	    }
	  }

      /* Block type 5 */
      for (block_y = 0; block_y < 4; block_y++)
	for (block_x = 0; block_x < 2; block_x++)
	  {
	    x = sads + mb_off7 +
	      (4 * block_y + 2 * block_x) * MAX_POS_PADDED;
	    y = x + MAX_POS_PADDED;
	    z = sads + mb_off5 +
	      (2 * block_y + block_x) * MAX_POS_PADDED;

	    for (count = 0; count < MAX_POS; count++) {
	      *z = (unsigned short)((unsigned int)*x + (unsigned int)*y);
	      ++z; ++x; ++y;
	    }
	  }

      /* Block type 4 */
      for (block_y = 0; block_y < 2; block_y++)
	for (block_x = 0; block_x < 2; block_x++)
	  {
	    x = sads + mb_off5 +
	      (4 * block_y + block_x) * MAX_POS_PADDED;
	    y = x + 2 * MAX_POS_PADDED;
	    z = sads + mb_off4 +
	      (2 * block_y + block_x) * MAX_POS_PADDED;
	    
	    for (count = 0; count < MAX_POS; count++) {
	      *z = (unsigned short)((unsigned int)*x + (unsigned int)*y);
	      ++z; ++x; ++y;
	    }
	  }
      
      /* Block type 3 */
      x = sads + mb_off4;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + mb_off3;
      
      for (count = 0; count < MAX_POS; count++) {
	*z = (unsigned short)((unsigned int)*x + (unsigned int)*y);
	++z; ++x; ++y;
      }

      x = sads + mb_off4 +
	MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + mb_off3 +
	MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) {
	*z = (unsigned short)((unsigned int)*x + (unsigned int)*y);
	++z; ++x; ++y;
      }

      /* Block type 2 */
      x = sads + mb_off4;
      y = x + MAX_POS_PADDED;
      z = sads + mb_off2;

      for (count = 0; count < MAX_POS; count++) {
	*z = (unsigned short)((unsigned int)*x + (unsigned int)*y);
	++z; ++x; ++y;
      }

      x = sads + mb_off4 +
	2 * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + mb_off2 +
	MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) {
	*z = (unsigned short)((unsigned int)*x + (unsigned int)*y);
	++z; ++x; ++y;
      }

      /* Block type 1 */
      x = sads + mb_off2;
      y = x + MAX_POS_PADDED;
      z = sads + mb_off1;

      for (count = 0; count < MAX_POS; count++) {
	*z = (unsigned short)((unsigned int)*x + (unsigned int)*y);
	++z; ++x; ++y;
      }
    }
}


