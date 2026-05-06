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

  /* Go to the starting offset in blk_sad */
  blk_sad += SAD_TYPE_7_IX(mb_width * mb_height);

  /* For each block */
  /* Parallelize over macroblocks: outer loop over mb_y, inner mb_x kept inside
   * to increase work per thread and improve locality.
   */
#ifdef _OPENMP
#pragma omp parallel for private(mb_x, frame_yoff) schedule(static)
#endif
  for (mb_y = 0; mb_y < mb_height; mb_y++)
    {
      frame_yoff = (unsigned int)mb_y * 256u * (unsigned int)mb_width;
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
  /* Precompute frequently used constants */
  const int width  = mb_width * 16;
  const int height = mb_height * 16;
  const int frame_y_base = frame_y;
  const int frame_x_base = frame_x;

  int pos_y, pos_x;
  int pos = 0;			/* search position */

  /* Each search position */
  for (pos_y = -SEARCH_RANGE; pos_y <= SEARCH_RANGE; pos_y++) {
    const int ref_y_base = frame_y_base + pos_y;
    for (pos_x = -SEARCH_RANGE; pos_x <= SEARCH_RANGE; pos_x++, pos++) {
      const int ref_x_base = frame_x_base + pos_x;

      /* Each 4x4 block in the macroblock */
      int blky;
      for (blky = 0; blky < 4; blky++) {
	const int blk_y_off = blky * 4;
	const int frame_blk_y = blk_y_off;
	int blkx;
	for (blkx = 0; blkx < 4; blkx++) {
	  const int blk_x_off = blkx * 4;
	  unsigned int sad = 0;

	  /* Each pixel in the 4x4 block.
	   * The loops are fully unrolled for better ILP and to avoid
	   * repeated loop-control overheads in the innermost kernel.
	   */

	  /* y = 0 */
	  {
	    const int y = 0;
	    const int frame_row = (frame_blk_y + y) * width;

	    /* x = 0 */
	    {
	      const int x = 0;
	      int ref_x = ref_x_base + blk_x_off + x;
	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      int ref_y = ref_y_base + blk_y_off + y;
	      if (ref_y < 0) ref_y = 0;
	      else if (ref_y >= height) ref_y = height - 1;

	      const unsigned int b = ref[(unsigned int)ref_y * (unsigned int)width + (unsigned int)ref_x];
	      const unsigned int a = frame[frame_row + (blk_x_off + x)];
	      sad += (a > b) ? (a - b) : (b - a);
	    }

	    /* x = 1 */
	    {
	      const int x = 1;
	      int ref_x = ref_x_base + blk_x_off + x;
	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      int ref_y = ref_y_base + blk_y_off + y;
	      if (ref_y < 0) ref_y = 0;
	      else if (ref_y >= height) ref_y = height - 1;

	      const unsigned int b = ref[(unsigned int)ref_y * (unsigned int)width + (unsigned int)ref_x];
	      const unsigned int a = frame[frame_row + (blk_x_off + x)];
	      sad += (a > b) ? (a - b) : (b - a);
	    }

	    /* x = 2 */
	    {
	      const int x = 2;
	      int ref_x = ref_x_base + blk_x_off + x;
	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      int ref_y = ref_y_base + blk_y_off + y;
	      if (ref_y < 0) ref_y = 0;
	      else if (ref_y >= height) ref_y = height - 1;

	      const unsigned int b = ref[(unsigned int)ref_y * (unsigned int)width + (unsigned int)ref_x];
	      const unsigned int a = frame[frame_row + (blk_x_off + x)];
	      sad += (a > b) ? (a - b) : (b - a);
	    }

	    /* x = 3 */
	    {
	      const int x = 3;
	      int ref_x = ref_x_base + blk_x_off + x;
	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      int ref_y = ref_y_base + blk_y_off + y;
	      if (ref_y < 0) ref_y = 0;
	      else if (ref_y >= height) ref_y = height - 1;

	      const unsigned int b = ref[(unsigned int)ref_y * (unsigned int)width + (unsigned int)ref_x];
	      const unsigned int a = frame[frame_row + (blk_x_off + x)];
	      sad += (a > b) ? (a - b) : (b - a);
	    }
	  }

	  /* y = 1 */
	  {
	    const int y = 1;
	    const int frame_row = (frame_blk_y + y) * width;

	    /* x = 0 */
	    {
	      const int x = 0;
	      int ref_x = ref_x_base + blk_x_off + x;
	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      int ref_y = ref_y_base + blk_y_off + y;
	      if (ref_y < 0) ref_y = 0;
	      else if (ref_y >= height) ref_y = height - 1;

	      const unsigned int b = ref[(unsigned int)ref_y * (unsigned int)width + (unsigned int)ref_x];
	      const unsigned int a = frame[frame_row + (blk_x_off + x)];
	      sad += (a > b) ? (a - b) : (b - a);
	    }

	    /* x = 1 */
	    {
	      const int x = 1;
	      int ref_x = ref_x_base + blk_x_off + x;
	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      int ref_y = ref_y_base + blk_y_off + y;
	      if (ref_y < 0) ref_y = 0;
	      else if (ref_y >= height) ref_y = height - 1;

	      const unsigned int b = ref[(unsigned int)ref_y * (unsigned int)width + (unsigned int)ref_x];
	      const unsigned int a = frame[frame_row + (blk_x_off + x)];
	      sad += (a > b) ? (a - b) : (b - a);
	    }

	    /* x = 2 */
	    {
	      const int x = 2;
	      int ref_x = ref_x_base + blk_x_off + x;
	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      int ref_y = ref_y_base + blk_y_off + y;
	      if (ref_y < 0) ref_y = 0;
	      else if (ref_y >= height) ref_y = height - 1;

	      const unsigned int b = ref[(unsigned int)ref_y * (unsigned int)width + (unsigned int)ref_x];
	      const unsigned int a = frame[frame_row + (blk_x_off + x)];
	      sad += (a > b) ? (a - b) : (b - a);
	    }

	    /* x = 3 */
	    {
	      const int x = 3;
	      int ref_x = ref_x_base + blk_x_off + x;
	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      int ref_y = ref_y_base + blk_y_off + y;
	      if (ref_y < 0) ref_y = 0;
	      else if (ref_y >= height) ref_y = height - 1;

	      const unsigned int b = ref[(unsigned int)ref_y * (unsigned int)width + (unsigned int)ref_x];
	      const unsigned int a = frame[frame_row + (blk_x_off + x)];
	      sad += (a > b) ? (a - b) : (b - a);
	    }
	  }

	  /* y = 2 */
	  {
	    const int y = 2;
	    const int frame_row = (frame_blk_y + y) * width;

	    /* x = 0 */
	    {
	      const int x = 0;
	      int ref_x = ref_x_base + blk_x_off + x;
	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      int ref_y = ref_y_base + blk_y_off + y;
	      if (ref_y < 0) ref_y = 0;
	      else if (ref_y >= height) ref_y = height - 1;

	      const unsigned int b = ref[(unsigned int)ref_y * (unsigned int)width + (unsigned int)ref_x];
	      const unsigned int a = frame[frame_row + (blk_x_off + x)];
	      sad += (a > b) ? (a - b) : (b - a);
	    }

	    /* x = 1 */
	    {
	      const int x = 1;
	      int ref_x = ref_x_base + blk_x_off + x;
	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      int ref_y = ref_y_base + blk_y_off + y;
	      if (ref_y < 0) ref_y = 0;
	      else if (ref_y >= height) ref_y = height - 1;

	      const unsigned int b = ref[(unsigned int)ref_y * (unsigned int)width + (unsigned int)ref_x];
	      const unsigned int a = frame[frame_row + (blk_x_off + x)];
	      sad += (a > b) ? (a - b) : (b - a);
	    }

	    /* x = 2 */
	    {
	      const int x = 2;
	      int ref_x = ref_x_base + blk_x_off + x;
	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      int ref_y = ref_y_base + blk_y_off + y;
	      if (ref_y < 0) ref_y = 0;
	      else if (ref_y >= height) ref_y = height - 1;

	      const unsigned int b = ref[(unsigned int)ref_y * (unsigned int)width + (unsigned int)ref_x];
	      const unsigned int a = frame[frame_row + (blk_x_off + x)];
	      sad += (a > b) ? (a - b) : (b - a);
	    }

	    /* x = 3 */
	    {
	      const int x = 3;
	      int ref_x = ref_x_base + blk_x_off + x;
	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      int ref_y = ref_y_base + blk_y_off + y;
	      if (ref_y < 0) ref_y = 0;
	      else if (ref_y >= height) ref_y = height - 1;

	      const unsigned int b = ref[(unsigned int)ref_y * (unsigned int)width + (unsigned int)ref_x];
	      const unsigned int a = frame[frame_row + (blk_x_off + x)];
	      sad += (a > b) ? (a - b) : (b - a);
	    }
	  }

	  /* y = 3 */
	  {
	    const int y = 3;
	    const int frame_row = (frame_blk_y + y) * width;

	    /* x = 0 */
	    {
	      const int x = 0;
	      int ref_x = ref_x_base + blk_x_off + x;
	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      int ref_y = ref_y_base + blk_y_off + y;
	      if (ref_y < 0) ref_y = 0;
	      else if (ref_y >= height) ref_y = height - 1;

	      const unsigned int b = ref[(unsigned int)ref_y * (unsigned int)width + (unsigned int)ref_x];
	      const unsigned int a = frame[frame_row + (blk_x_off + x)];
	      sad += (a > b) ? (a - b) : (b - a);
	    }

	    /* x = 1 */
	    {
	      const int x = 1;
	      int ref_x = ref_x_base + blk_x_off + x;
	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      int ref_y = ref_y_base + blk_y_off + y;
	      if (ref_y < 0) ref_y = 0;
	      else if (ref_y >= height) ref_y = height - 1;

	      const unsigned int b = ref[(unsigned int)ref_y * (unsigned int)width + (unsigned int)ref_x];
	      const unsigned int a = frame[frame_row + (blk_x_off + x)];
	      sad += (a > b) ? (a - b) : (b - a);
	    }

	    /* x = 2 */
	    {
	      const int x = 2;
	      int ref_x = ref_x_base + blk_x_off + x;
	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      int ref_y = ref_y_base + blk_y_off + y;
	      if (ref_y < 0) ref_y = 0;
	      else if (ref_y >= height) ref_y = height - 1;

	      const unsigned int b = ref[(unsigned int)ref_y * (unsigned int)width + (unsigned int)ref_x];
	      const unsigned int a = frame[frame_row + (blk_x_off + x)];
	      sad += (a > b) ? (a - b) : (b - a);
	    }

	    /* x = 3 */
	    {
	      const int x = 3;
	      int ref_x = ref_x_base + blk_x_off + x;
	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      int ref_y = ref_y_base + blk_y_off + y;
	      if (ref_y < 0) ref_y = 0;
	      else if (ref_y >= height) ref_y = height - 1;

	      const unsigned int b = ref[(unsigned int)ref_y * (unsigned int)width + (unsigned int)ref_x];
	      const unsigned int a = frame[frame_row + (blk_x_off + x)];
	      sad += (a > b) ? (a - b) : (b - a);
	    }
	  }

	  /* Save the SAD */
	  macroblock_sad[MAX_POS_PADDED * (4 * blky + blkx) + pos] = (unsigned short)sad;
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

  /* Each macroblock's computations are independent; parallelize over mbs. */
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
	      macroblock * SAD_TYPE_7_CT * MAX_POS_PADDED +
	      (8 * block_y + block_x) * MAX_POS_PADDED;
	    y = x + 4 * MAX_POS_PADDED;
	    z = sads + SAD_TYPE_6_IX(mbs) +
	      macroblock * SAD_TYPE_6_CT * MAX_POS_PADDED +
	      (4 * block_y + block_x) * MAX_POS_PADDED;

	    /* Inner loops are simple vector adds; allow compiler/vectorizer
	     * to work with a straight count-up loop.
	     */
	    for (count = 0; count < MAX_POS; count++) {
	      z[count] = (unsigned short)(x[count] + y[count]);
	    }
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

	    for (count = 0; count < MAX_POS; count++) {
	      z[count] = (unsigned short)(x[count] + y[count]);
	    }
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
	    
	    for (count = 0; count < MAX_POS; count++) {
	      z[count] = (unsigned short)(x[count] + y[count]);
	    }
	  }
      
      /* Block type 3 */
      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + SAD_TYPE_3_IX(mbs) +
	macroblock * SAD_TYPE_3_CT * MAX_POS_PADDED;
      
      for (count = 0; count < MAX_POS; count++) {
	z[count] = (unsigned short)(x[count] + y[count]);
      }

      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED +
	MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + SAD_TYPE_3_IX(mbs) +
	macroblock * SAD_TYPE_3_CT * MAX_POS_PADDED +
	MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) {
	z[count] = (unsigned short)(x[count] + y[count]);
      }

      /* Block type 2 */
      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) {
	z[count] = (unsigned short)(x[count] + y[count]);
      }

      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED +
	2 * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED +
	MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) {
	z[count] = (unsigned short)(x[count] + y[count]);
      }

      /* Block type 1 */
      x = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_1_IX(mbs) +
	macroblock * SAD_TYPE_1_CT * MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) {
	z[count] = (unsigned short)(x[count] + y[count]);
      }
    }
}


