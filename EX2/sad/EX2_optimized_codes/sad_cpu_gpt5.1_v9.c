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
#pragma omp parallel for private(mb_x, frame_yoff) schedule(static)
  for (mb_y = 0; mb_y < mb_height; mb_y++) {
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
	  const int base_frame_y = blky * 4;
	  const int base_frame_x = blkx * 4;
	  const int blk_index = 4 * blky + blkx;
	  const int sad_base = MAX_POS_PADDED * blk_index;
	  unsigned short sad0 = 0, sad1 = 0, sad2 = 0, sad3 = 0;

	  /* Each pixel, unrolled for 4x4 block */
	  {
	    int ref_y = frame_y + pos_y + base_frame_y + 0;
	    if (ref_y < 0) ref_y = 0;
	    else if (ref_y >= height) ref_y = height - 1;
	    const int fy0 = base_frame_y * width + base_frame_x;
	    const int ry0 = ref_y * width;

	    int ref_x = frame_x + pos_x + base_frame_x + 0;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      const unsigned int a = frame[fy0 + 0];
	      const unsigned int b = ref[ry0 + ref_x];
	      sad0 += (unsigned short)abs((int)a - (int)b);
	    }

	    ref_x = frame_x + pos_x + base_frame_x + 1;
   	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      const unsigned int a = frame[fy0 + 1];
	      const unsigned int b = ref[ry0 + ref_x];
	      sad0 += (unsigned short)abs((int)a - (int)b);
	    }

	    ref_x = frame_x + pos_x + base_frame_x + 2;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      const unsigned int a = frame[fy0 + 2];
	      const unsigned int b = ref[ry0 + ref_x];
	      sad0 += (unsigned short)abs((int)a - (int)b);
	    }

	    ref_x = frame_x + pos_x + base_frame_x + 3;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      const unsigned int a = frame[fy0 + 3];
	      const unsigned int b = ref[ry0 + ref_x];
	      sad0 += (unsigned short)abs((int)a - (int)b);
	    }
	  }
	  {
	    int ref_y = frame_y + pos_y + base_frame_y + 1;
	    if (ref_y < 0) ref_y = 0;
	    else if (ref_y >= height) ref_y = height - 1;
	    const int fy1 = (base_frame_y + 1) * width + base_frame_x;
	    const int ry1 = ref_y * width;

	    int ref_x = frame_x + pos_x + base_frame_x + 0;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      const unsigned int a = frame[fy1 + 0];
	      const unsigned int b = ref[ry1 + ref_x];
	      sad1 += (unsigned short)abs((int)a - (int)b);
	    }

	    ref_x = frame_x + pos_x + base_frame_x + 1;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      const unsigned int a = frame[fy1 + 1];
	      const unsigned int b = ref[ry1 + ref_x];
	      sad1 += (unsigned short)abs((int)a - (int)b);
	    }

	    ref_x = frame_x + pos_x + base_frame_x + 2;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      const unsigned int a = frame[fy1 + 2];
	      const unsigned int b = ref[ry1 + ref_x];
	      sad1 += (unsigned short)abs((int)a - (int)b);
	    }

	    ref_x = frame_x + pos_x + base_frame_x + 3;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      const unsigned int a = frame[fy1 + 3];
	      const unsigned int b = ref[ry1 + ref_x];
	      sad1 += (unsigned short)abs((int)a - (int)b);
	    }
	  }
	  {
	    int ref_y = frame_y + pos_y + base_frame_y + 2;
	    if (ref_y < 0) ref_y = 0;
	    else if (ref_y >= height) ref_y = height - 1;
	    const int fy2 = (base_frame_y + 2) * width + base_frame_x;
	    const int ry2 = ref_y * width;

	    int ref_x = frame_x + pos_x + base_frame_x + 0;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      const unsigned int a = frame[fy2 + 0];
	      const unsigned int b = ref[ry2 + ref_x];
	      sad2 += (unsigned short)abs((int)a - (int)b);
	    }

	    ref_x = frame_x + pos_x + base_frame_x + 1;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      const unsigned int a = frame[fy2 + 1];
	      const unsigned int b = ref[ry2 + ref_x];
	      sad2 += (unsigned short)abs((int)a - (int)b);
	    }

	    ref_x = frame_x + pos_x + base_frame_x + 2;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      const unsigned int a = frame[fy2 + 2];
	      const unsigned int b = ref[ry2 + ref_x];
	      sad2 += (unsigned short)abs((int)a - (int)b);
	    }

	    ref_x = frame_x + pos_x + base_frame_x + 3;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      const unsigned int a = frame[fy2 + 3];
	      const unsigned int b = ref[ry2 + ref_x];
	      sad2 += (unsigned short)abs((int)a - (int)b);
	    }
	  }
	  {
	    int ref_y = frame_y + pos_y + base_frame_y + 3;
	    if (ref_y < 0) ref_y = 0;
	    else if (ref_y >= height) ref_y = height - 1;
	    const int fy3 = (base_frame_y + 3) * width + base_frame_x;
	    const int ry3 = ref_y * width;

	    int ref_x = frame_x + pos_x + base_frame_x + 0;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      const unsigned int a = frame[fy3 + 0];
	      const unsigned int b = ref[ry3 + ref_x];
	      sad3 += (unsigned short)abs((int)a - (int)b);
	    }

	    ref_x = frame_x + pos_x + base_frame_x + 1;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      const unsigned int a = frame[fy3 + 1];
	      const unsigned int b = ref[ry3 + ref_x];
	      sad3 += (unsigned short)abs((int)a - (int)b);
	    }

	    ref_x = frame_x + pos_x + base_frame_x + 2;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      const unsigned int a = frame[fy3 + 2];
	      const unsigned int b = ref[ry3 + ref_x];
	      sad3 += (unsigned short)abs((int)a - (int)b);
	    }

	    ref_x = frame_x + pos_x + base_frame_x + 3;
	    if (ref_x < 0) ref_x = 0;
	    else if (ref_x >= width) ref_x = width - 1;
	    {
	      const unsigned int a = frame[fy3 + 3];
	      const unsigned int b = ref[ry3 + ref_x];
	      sad3 += (unsigned short)abs((int)a - (int)b);
	    }
	  }

	  macroblock_sad[sad_base + pos] =
	    (unsigned short)(sad0 + sad1 + sad2 + sad3);
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

#pragma omp simd
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

#pragma omp simd
	    for (count = 0; count < MAX_POS; count++) z[count] = x[count] + y[count];
	  }
      
      /* Block type 3 */
      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + SAD_TYPE_3_IX(mbs) +
	macroblock * SAD_TYPE_3_CT * MAX_POS_PADDED;
      
#pragma omp simd
      for (count = 0; count < MAX_POS; count++) z[count] = x[count] + y[count];

      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED +
	MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + SAD_TYPE_3_IX(mbs) +
	macroblock * SAD_TYPE_3_CT * MAX_POS_PADDED +
	MAX_POS_PADDED;

#pragma omp simd
      for (count = 0; count < MAX_POS; count++) z[count] = x[count] + y[count];

      /* Block type 2 */
      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;

#pragma omp simd
      for (count = 0; count < MAX_POS; count++) z[count] = x[count] + y[count];

      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED +
	2 * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED +
	MAX_POS_PADDED;

#pragma omp simd
      for (count = 0; count < MAX_POS; count++) z[count] = x[count] + y[count];

      /* Block type 1 */
      x = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_1_IX(mbs) +
	macroblock * SAD_TYPE_1_CT * MAX_POS_PADDED;

#pragma omp simd
      for (count = 0; count < MAX_POS; count++) z[count] = x[count] + y[count];
    }
}


