#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
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
  const int mb_count = mb_width * mb_height;
  const int frame_stride = 256 * mb_width;
  const int blk_stride = SAD_TYPE_7_CT * MAX_POS_PADDED;

  /* Go to the starting offset in blk_sad */
  blk_sad += SAD_TYPE_7_IX(mb_count);

  /* Parallelize over macroblocks: each macroblock is independent */
#ifdef _OPENMP
#pragma omp parallel for private(mb_x, frame_yoff) schedule(static)
#endif
  for (mb_y = 0; mb_y < mb_height; mb_y++) {
    frame_yoff = (unsigned int)mb_y * (unsigned int)frame_stride;
    unsigned short *frame_row = frame + frame_yoff;
    unsigned short *blk_row   = blk_sad + mb_y * mb_width * blk_stride;

    for (mb_x = 0; mb_x < mb_width; mb_x++) {
      sad4_one_macroblock(
        blk_row + mb_x * blk_stride,
        frame_row + mb_x * 16,
        ref,
        mb_y * 16,
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
  const int width  = mb_width * 16;
  const int height = mb_height * 16;
  const int frame_y_base = frame_y;
  const int frame_x_base = frame_x;

  int pos_x, pos_y;
  int pos = 0;			/* search position */

  /* Each search position */
  for (pos_y = -SEARCH_RANGE; pos_y <= SEARCH_RANGE; pos_y++) {
    const int pos_y_base = frame_y_base + pos_y;
    for (pos_x = -SEARCH_RANGE; pos_x <= SEARCH_RANGE; pos_x++, pos++) {
      const int pos_x_base = frame_x_base + pos_x;

      int blky, blkx;

      /* Each 4x4 block in the macroblock */
      for (blky = 0; blky < 4; blky++) {
	for (blkx = 0; blkx < 4; blkx++) {
	  int y, x;
	  unsigned int sad = 0;

	  const int blk_y_base = blky * 4;
	  const int blk_x_base = blkx * 4;

	  /* Each pixel */
	  for (y = 0; y < 4; y++) {
	    const int frame_y_off = (blk_y_base + y) * width;
	    const int ref_y_unclipped = pos_y_base + blk_y_base + y;
	    int ref_y = ref_y_unclipped;
	    if (ref_y < 0) ref_y = 0;
	    else if (ref_y >= height) ref_y = height - 1;
	    const int ref_y_off = ref_y * width;

	    for (x = 0; x < 4; x++) {
	      const int frame_x_off = blk_x_base + x;
	      const int ref_x_unclipped = pos_x_base + blk_x_base + x;
	      int ref_x = ref_x_unclipped;
	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      const unsigned int b = ref[ref_y_off + ref_x];
	      const unsigned int a = frame[frame_y_off + frame_x_off];

	      sad += (a > b) ? (a - b) : (b - a);
	    }
	  }

	  /* Save the SAD */
	  macroblock_sad[MAX_POS_PADDED*(4*blky+blkx) + pos] =
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

  const int max_pos = MAX_POS;
  const int max_pos_padded = MAX_POS_PADDED;

#ifdef _OPENMP
#pragma omp parallel for private(block_x, block_y, x, y, z, count) schedule(static)
#endif
  for (macroblock = 0; macroblock < mbs; macroblock++)
    {
      /* Block type 6 */
      for (block_y = 0; block_y < 2; block_y++)
	for (block_x = 0; block_x < 4; block_x++)
	  {
	    x = sads + SAD_TYPE_7_IX(mbs) +
	      macroblock * SAD_TYPE_7_CT * max_pos_padded +
	      (8 * block_y + block_x) * max_pos_padded;
	    y = x + 4 * max_pos_padded;
	    z = sads + SAD_TYPE_6_IX(mbs) +
	      macroblock * SAD_TYPE_6_CT * max_pos_padded +
	      (4 * block_y + block_x) * max_pos_padded;

	    for (count = 0; count < max_pos; count++)
              z[count] = (unsigned short)(x[count] + y[count]);
	  }

      /* Block type 5 */
      for (block_y = 0; block_y < 4; block_y++)
	for (block_x = 0; block_x < 2; block_x++)
	  {
	    x = sads + SAD_TYPE_7_IX(mbs) +
	      macroblock * SAD_TYPE_7_CT * max_pos_padded +
	      (4 * block_y + 2 * block_x) * max_pos_padded;
	    y = x + max_pos_padded;
	    z = sads + SAD_TYPE_5_IX(mbs) +
	      macroblock * SAD_TYPE_6_CT * max_pos_padded +
	      (2 * block_y + block_x) * max_pos_padded;

	    for (count = 0; count < max_pos; count++)
              z[count] = (unsigned short)(x[count] + y[count]);
	  }

      /* Block type 4 */
      for (block_y = 0; block_y < 2; block_y++)
	for (block_x = 0; block_x < 2; block_x++)
	  {
	    x = sads + SAD_TYPE_5_IX(mbs) +
	      macroblock * SAD_TYPE_5_CT * max_pos_padded +
	      (4 * block_y + block_x) * max_pos_padded;
	    y = x + 2 * max_pos_padded;
	    z = sads + SAD_TYPE_4_IX(mbs) +
	      macroblock * SAD_TYPE_4_CT * max_pos_padded +
	      (2 * block_y + block_x) * max_pos_padded;
	    
	    for (count = 0; count < max_pos; count++)
              z[count] = (unsigned short)(x[count] + y[count]);
	  }
      
      /* Block type 3 */
      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * max_pos_padded;
      y = x + 2 * max_pos_padded;
      z = sads + SAD_TYPE_3_IX(mbs) +
	macroblock * SAD_TYPE_3_CT * max_pos_padded;
      
      for (count = 0; count < max_pos; count++)
        z[count] = (unsigned short)(x[count] + y[count]);

      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * max_pos_padded +
	max_pos_padded;
      y = x + 2 * max_pos_padded;
      z = sads + SAD_TYPE_3_IX(mbs) +
	macroblock * SAD_TYPE_3_CT * max_pos_padded +
	max_pos_padded;

      for (count = 0; count < max_pos; count++)
        z[count] = (unsigned short)(x[count] + y[count]);

      /* Block type 2 */
      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * max_pos_padded;
      y = x + max_pos_padded;
      z = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * max_pos_padded;

      for (count = 0; count < max_pos; count++)
        z[count] = (unsigned short)(x[count] + y[count]);

      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * max_pos_padded +
	2 * max_pos_padded;
      y = x + max_pos_padded;
      z = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * max_pos_padded +
	max_pos_padded;

      for (count = 0; count < max_pos; count++)
        z[count] = (unsigned short)(x[count] + y[count]);

      /* Block type 1 */
      x = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * max_pos_padded;
      y = x + max_pos_padded;
      z = sads + SAD_TYPE_1_IX(mbs) +
	macroblock * SAD_TYPE_1_CT * max_pos_padded;

      for (count = 0; count < max_pos; count++)
        z[count] = (unsigned short)(x[count] + y[count]);
    }
}


