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
  int mb_x, mb_y;
  unsigned int frame_yoff;

  /* Go to the starting offset in blk_sad */
  blk_sad += SAD_TYPE_7_IX(mb_width * mb_height);

  /* For each block */
  /* Parallelize over macroblocks; each macroblock writes to a disjoint region */
#pragma omp parallel for private(mb_x, frame_yoff) schedule(static)
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
  int pos_x, pos_y;
  const int width = mb_width * 16;
  const int height = mb_height * 16;
  int pos;			/* search position */

  /* Each search position */
  pos = 0;
  for (pos_y = -SEARCH_RANGE; pos_y <= SEARCH_RANGE; pos_y++) {
    for (pos_x = -SEARCH_RANGE; pos_x <= SEARCH_RANGE; pos_x++, pos++) {
      int blky, blkx;

      /* Each 4x4 block in the macroblock */
      for (blky = 0; blky < 4; blky++) {
	for (blkx = 0; blkx < 4; blkx++) {

          /* Each pixel: unroll inner loops and hoist invariants */
	  int base_y = blky * 4;
	  int base_x = blkx * 4;
	  unsigned short sad = 0;

	  /* y = 0 */
	  {
	    int y = 0;
	    int fy = base_y + y;
	    int ref_y = frame_y + pos_y + base_y + y;
	    if (ref_y < 0) ref_y = 0;
	    else if (ref_y >= height) ref_y = height - 1;
	    unsigned int frame_row_off = (unsigned int)fy * (unsigned int)width;
	    unsigned int ref_row_off = (unsigned int)ref_y * (unsigned int)width;

	    int x = 0;
	    int fx = base_x + x;
	    int ref_x = frame_x + pos_x + base_x + x;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      unsigned int a = frame[frame_row_off + fx];
	      unsigned int b = ref[ref_row_off + ref_x];
	      sad += (unsigned short)abs((int)a - (int)b);
	    }

	    x = 1;
	    fx = base_x + x;
	    ref_x = frame_x + pos_x + base_x + x;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      unsigned int a = frame[frame_row_off + fx];
	      unsigned int b = ref[ref_row_off + ref_x];
	      sad += (unsigned short)abs((int)a - (int)b);
	    }

	    x = 2;
	    fx = base_x + x;
	    ref_x = frame_x + pos_x + base_x + x;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      unsigned int a = frame[frame_row_off + fx];
	      unsigned int b = ref[ref_row_off + ref_x];
	      sad += (unsigned short)abs((int)a - (int)b);
	    }

	    x = 3;
	    fx = base_x + x;
	    ref_x = frame_x + pos_x + base_x + x;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      unsigned int a = frame[frame_row_off + fx];
	      unsigned int b = ref[ref_row_off + ref_x];
	      sad += (unsigned short)abs((int)a - (int)b);
	    }
	  }

	  /* y = 1 */
	  {
	    int y = 1;
	    int fy = base_y + y;
	    int ref_y = frame_y + pos_y + base_y + y;
	    if (ref_y < 0) ref_y = 0;
	    else if (ref_y >= height) ref_y = height - 1;
	    unsigned int frame_row_off = (unsigned int)fy * (unsigned int)width;
	    unsigned int ref_row_off = (unsigned int)ref_y * (unsigned int)width;

	    int x = 0;
	    int fx = base_x + x;
	    int ref_x = frame_x + pos_x + base_x + x;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      unsigned int a = frame[frame_row_off + fx];
	      unsigned int b = ref[ref_row_off + ref_x];
	      sad += (unsigned short)abs((int)a - (int)b);
	    }

	    x = 1;
	    fx = base_x + x;
	    ref_x = frame_x + pos_x + base_x + x;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      unsigned int a = frame[frame_row_off + fx];
	      unsigned int b = ref[ref_row_off + ref_x];
	      sad += (unsigned short)abs((int)a - (int)b);
	    }

	    x = 2;
	    fx = base_x + x;
	    ref_x = frame_x + pos_x + base_x + x;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      unsigned int a = frame[frame_row_off + fx];
	      unsigned int b = ref[ref_row_off + ref_x];
	      sad += (unsigned short)abs((int)a - (int)b);
	    }

	    x = 3;
	    fx = base_x + x;
	    ref_x = frame_x + pos_x + base_x + x;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      unsigned int a = frame[frame_row_off + fx];
	      unsigned int b = ref[ref_row_off + ref_x];
	      sad += (unsigned short)abs((int)a - (int)b);
	    }
	  }

	  /* y = 2 */
	  {
	    int y = 2;
	    int fy = base_y + y;
	    int ref_y = frame_y + pos_y + base_y + y;
	    if (ref_y < 0) ref_y = 0;
	    else if (ref_y >= height) ref_y = height - 1;
	    unsigned int frame_row_off = (unsigned int)fy * (unsigned int)width;
	    unsigned int ref_row_off = (unsigned int)ref_y * (unsigned int)width;

	    int x = 0;
	    int fx = base_x + x;
	    int ref_x = frame_x + pos_x + base_x + x;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      unsigned int a = frame[frame_row_off + fx];
	      unsigned int b = ref[ref_row_off + ref_x];
	      sad += (unsigned short)abs((int)a - (int)b);
	    }

	    x = 1;
	    fx = base_x + x;
	    ref_x = frame_x + pos_x + base_x + x;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      unsigned int a = frame[frame_row_off + fx];
	      unsigned int b = ref[ref_row_off + ref_x];
	      sad += (unsigned short)abs((int)a - (int)b);
	    }

	    x = 2;
	    fx = base_x + x;
	    ref_x = frame_x + pos_x + base_x + x;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      unsigned int a = frame[frame_row_off + fx];
	      unsigned int b = ref[ref_row_off + ref_x];
	      sad += (unsigned short)abs((int)a - (int)b);
	    }

	    x = 3;
	    fx = base_x + x;
	    ref_x = frame_x + pos_x + base_x + x;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      unsigned int a = frame[frame_row_off + fx];
	      unsigned int b = ref[ref_row_off + ref_x];
	      sad += (unsigned short)abs((int)a - (int)b);
	    }
	  }

	  /* y = 3 */
	  {
	    int y = 3;
	    int fy = base_y + y;
	    int ref_y = frame_y + pos_y + base_y + y;
	    if (ref_y < 0) ref_y = 0;
	    else if (ref_y >= height) ref_y = height - 1;
	    unsigned int frame_row_off = (unsigned int)fy * (unsigned int)width;
	    unsigned int ref_row_off = (unsigned int)ref_y * (unsigned int)width;

	    int x = 0;
	    int fx = base_x + x;
	    int ref_x = frame_x + pos_x + base_x + x;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      unsigned int a = frame[frame_row_off + fx];
	      unsigned int b = ref[ref_row_off + ref_x];
	      sad += (unsigned short)abs((int)a - (int)b);
	    }

	    x = 1;
	    fx = base_x + x;
	    ref_x = frame_x + pos_x + base_x + x;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      unsigned int a = frame[frame_row_off + fx];
	      unsigned int b = ref[ref_row_off + ref_x];
	      sad += (unsigned short)abs((int)a - (int)b);
	    }

	    x = 2;
	    fx = base_x + x;
	    ref_x = frame_x + pos_x + base_x + x;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      unsigned int a = frame[frame_row_off + fx];
	      unsigned int b = ref[ref_row_off + ref_x];
	      sad += (unsigned short)abs((int)a - (int)b);
	    }

	    x = 3;
	    fx = base_x + x;
	    ref_x = frame_x + pos_x + base_x + x;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      unsigned int a = frame[frame_row_off + fx];
	      unsigned int b = ref[ref_row_off + ref_x];
	      sad += (unsigned short)abs((int)a - (int)b);
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
  int block_x, block_y;
  unsigned short *x, *y;	/* inputs to vector addition */
  unsigned short *z;		/* output of vector addition */
  int count;

  /* Parallelize across macroblocks; each macroblock uses disjoint ranges */
#pragma omp parallel for private(block_x, block_y, x, y, z, count) schedule(static)
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

#pragma omp simd
	    for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
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

#pragma omp simd
	    for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
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
	    
#pragma omp simd
	    for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
	  }
      
      /* Block type 3 */
      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + SAD_TYPE_3_IX(mbs) +
	macroblock * SAD_TYPE_3_CT * MAX_POS_PADDED;
      
#pragma omp simd
      for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));

      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED +
	MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + SAD_TYPE_3_IX(mbs) +
	macroblock * SAD_TYPE_3_CT * MAX_POS_PADDED +
	MAX_POS_PADDED;

#pragma omp simd
      for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));

      /* Block type 2 */
      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;

#pragma omp simd
      for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));

      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED +
	2 * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED +
	MAX_POS_PADDED;

#pragma omp simd
      for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));

      /* Block type 1 */
      x = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_1_IX(mbs) +
	macroblock * SAD_TYPE_1_CT * MAX_POS_PADDED;

#pragma omp simd
      for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
    }
}


