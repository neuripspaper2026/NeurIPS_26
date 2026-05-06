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
      frame_yoff = (unsigned int)mb_y * 256U * (unsigned int)mb_width;
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
    const int base_ref_y = frame_y + pos_y;
    const int base_frame_y = base_ref_y - frame_y; /* == pos_y */

    for (pos_x = -SEARCH_RANGE; pos_x <= SEARCH_RANGE; pos_x++, pos++) {
      const int base_ref_x = frame_x + pos_x;
      const int base_frame_x = base_ref_x - frame_x; /* == pos_x */

      int blky;
      /* Each 4x4 block in the macroblock */
      for (blky = 0; blky < 4; blky++) {
	int blkx;
	const int blk_y_off = blky * 4;
	const int ref_blk_y_off = base_ref_y + blk_y_off;
	const int frm_blk_y_off = base_frame_y + blk_y_off;

	for (blkx = 0; blkx < 4; blkx++) {
	  unsigned int sad = 0;
	  const int blk_x_off = blkx * 4;
	  const int ref_blk_x_off = base_ref_x + blk_x_off;
	  const int frm_blk_x_off = base_frame_x + blk_x_off;

	  int y;
	  /* Each pixel row */
	  for (y = 0; y < 4; y++) {
	    int ref_y = ref_blk_y_off + y;
	    int frame_row = frm_blk_y_off + y;

	    /* Clip reference Y once per row */
	    if (ref_y < 0) ref_y = 0;
	    else if (ref_y >= height) ref_y = height - 1;

	    const int ref_row_off = ref_y * width;
	    const int frm_row_off = frame_row * width;

	    int x;
	    for (x = 0; x < 4; x++) {
	      int ref_x = ref_blk_x_off + x;

	      /* Clip reference X once per pixel */
	      if (ref_x < 0) ref_x = 0;
	      else if (ref_x >= width) ref_x = width - 1;

	      const unsigned int b = ref[ref_row_off + ref_x];
	      const unsigned int a = frame[frm_row_off + (blk_x_off + x)];
	      sad += (unsigned int)abs((int)a - (int)b);
	    }
	  }

	  /* Save the SAD */
	  macroblock_sad[MAX_POS_PADDED*(4*blky+blkx) + pos] = (unsigned short)sad;
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


