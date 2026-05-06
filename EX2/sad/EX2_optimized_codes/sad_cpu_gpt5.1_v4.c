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
	     frame + frame_yoff + (unsigned int)mb_x * 16u,
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

  const int frame_x0 = frame_x;
  const int frame_y0 = frame_y;
  const unsigned short * const frame_base = frame;
  const unsigned short * const ref_base = ref;
  unsigned short * const mb_sad_base = macroblock_sad;
  const int stride = width;

  /* Each search position */
  pos = 0;
  for (pos_y = -SEARCH_RANGE; pos_y <= SEARCH_RANGE; pos_y++) {
    for (pos_x = -SEARCH_RANGE; pos_x <= SEARCH_RANGE; pos_x++, pos++) {
      int blky, blkx;

      /* Each 4x4 block in the macroblock */
      for (blky = 0; blky < 4; blky++) {
	for (blkx = 0; blkx < 4; blkx++) {

	  const int blk_offset_y = blky * 4;
	  const int blk_offset_x = blkx * 4;

	  /* Precompute unclipped reference coordinates for block origin */
	  int ref_y0 = frame_y0 + pos_y + blk_offset_y;
	  int ref_x0 = frame_x0 + pos_x + blk_offset_x;

	  /* Clip starting coordinates once */
	  if (ref_y0 < 0) ref_y0 = 0;
	  else if (ref_y0 > height - 4) ref_y0 = height - 4;

	  if (ref_x0 < 0) ref_x0 = 0;
	  else if (ref_x0 > width - 4) ref_x0 = width - 4;

	  const unsigned short *ref_ptr = ref_base + (ref_y0 * stride + ref_x0);
	  const unsigned short *frm_ptr = frame_base + (blk_offset_y * stride + blk_offset_x);

	  unsigned int sad = 0;

	  /* Each pixel (4x4), now with no inner clipping and flattened loops */
	  {
	    /* y = 0 */
	    {
	      unsigned int a0 = frm_ptr[0];
	      unsigned int b0 = ref_ptr[0];
	      sad += (a0 > b0) ? (a0 - b0) : (b0 - a0);

	      unsigned int a1 = frm_ptr[1];
	      unsigned int b1 = ref_ptr[1];
	      sad += (a1 > b1) ? (a1 - b1) : (b1 - a1);

	      unsigned int a2 = frm_ptr[2];
	      unsigned int b2 = ref_ptr[2];
	      sad += (a2 > b2) ? (a2 - b2) : (b2 - a2);

	      unsigned int a3 = frm_ptr[3];
	      unsigned int b3 = ref_ptr[3];
	      sad += (a3 > b3) ? (a3 - b3) : (b3 - a3);
	    }

	    /* y = 1 */
	    {
	      const unsigned short *frm_row = frm_ptr + stride;
	      const unsigned short *ref_row = ref_ptr + stride;

	      unsigned int a0 = frm_row[0];
	      unsigned int b0 = ref_row[0];
	      sad += (a0 > b0) ? (a0 - b0) : (b0 - a0);

	      unsigned int a1 = frm_row[1];
	      unsigned int b1 = ref_row[1];
	      sad += (a1 > b1) ? (a1 - b1) : (b1 - a1);

	      unsigned int a2 = frm_row[2];
	      unsigned int b2 = ref_row[2];
	      sad += (a2 > b2) ? (a2 - b2) : (b2 - a2);

	      unsigned int a3 = frm_row[3];
	      unsigned int b3 = ref_row[3];
	      sad += (a3 > b3) ? (a3 - b3) : (b3 - a3);
	    }

	    /* y = 2 */
	    {
	      const unsigned short *frm_row = frm_ptr + (stride << 1);
	      const unsigned short *ref_row = ref_ptr + (stride << 1);

	      unsigned int a0 = frm_row[0];
	      unsigned int b0 = ref_row[0];
	      sad += (a0 > b0) ? (a0 - b0) : (b0 - a0);

	      unsigned int a1 = frm_row[1];
	      unsigned int b1 = ref_row[1];
	      sad += (a1 > b1) ? (a1 - b1) : (b1 - a1);

	      unsigned int a2 = frm_row[2];
	      unsigned int b2 = ref_row[2];
	      sad += (a2 > b2) ? (a2 - b2) : (b2 - a2);

	      unsigned int a3 = frm_row[3];
	      unsigned int b3 = ref_row[3];
	      sad += (a3 > b3) ? (a3 - b3) : (b3 - a3);
	    }

	    /* y = 3 */
	    {
	      const unsigned short *frm_row = frm_ptr + stride * 3;
	      const unsigned short *ref_row = ref_ptr + stride * 3;

	      unsigned int a0 = frm_row[0];
	      unsigned int b0 = ref_row[0];
	      sad += (a0 > b0) ? (a0 - b0) : (b0 - a0);

	      unsigned int a1 = frm_row[1];
	      unsigned int b1 = ref_row[1];
	      sad += (a1 > b1) ? (a1 - b1) : (b1 - a1);

	      unsigned int a2 = frm_row[2];
	      unsigned int b2 = ref_row[2];
	      sad += (a2 > b2) ? (a2 - b2) : (b2 - a2);

	      unsigned int a3 = frm_row[3];
	      unsigned int b3 = ref_row[3];
	      sad += (a3 > b3) ? (a3 - b3) : (b3 - a3);
	    }
	  }

	  /* Save the SAD */
	  mb_sad_base[MAX_POS_PADDED * (4 * blky + blkx) + pos] = (unsigned short)sad;
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
	    
	    for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
	  }
      
      /* Block type 3 */
      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + SAD_TYPE_3_IX(mbs) +
	macroblock * SAD_TYPE_3_CT * MAX_POS_PADDED;
      
      for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));

      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED +
	MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + SAD_TYPE_3_IX(mbs) +
	macroblock * SAD_TYPE_3_CT * MAX_POS_PADDED +
	MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));

      /* Block type 2 */
      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));

      x = sads + SAD_TYPE_4_IX(mbs) +
	macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED +
	2 * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED +
	MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));

      /* Block type 1 */
      x = sads + SAD_TYPE_2_IX(mbs) +
	macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + SAD_TYPE_1_IX(mbs) +
	macroblock * SAD_TYPE_1_CT * MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) *z++ = (unsigned short)((unsigned int)(*x++) + (unsigned int)(*y++));
    }
}


