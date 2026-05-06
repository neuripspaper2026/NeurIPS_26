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
  blk_sad += SAD_TYPE_7_IX(mb_width * mb_height);

  /* For each block */
  for (mb_y = 0, frame_yoff = 0;
       mb_y < mb_height;
       mb_y++, frame_yoff += 256U * (unsigned int)mb_width)
    {
      for (mb_x = 0; mb_x < mb_width; mb_x++)
	{
	  sad4_one_macroblock
	    (blk_sad + (mb_y * mb_width + mb_x) * (SAD_TYPE_7_CT * MAX_POS_PADDED),
	     frame + frame_yoff + (unsigned int)mb_x * 16U,
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
  const int width  = mb_width * 16;
  const int height = mb_height * 16;
  int pos;			/* search position */

  const int base_y = frame_y;
  const int base_x = frame_x;
  unsigned short *const sad_base = macroblock_sad;

  /* Each search position */
  pos = 0;
  for (pos_y = -SEARCH_RANGE; pos_y <= SEARCH_RANGE; pos_y++) {
    const int off_y = base_y + pos_y;
    for (pos_x = -SEARCH_RANGE; pos_x <= SEARCH_RANGE; pos_x++, pos++) {
      const int off_x = base_x + pos_x;
      int blky, blkx;

      /* Each 4x4 block in the macroblock */
      for (blky = 0; blky < 4; blky++) {
	const int blk_y_off = blky * 4;
	for (blkx = 0; blkx < 4; blkx++) {
	  const int blk_x_off = blkx * 4;
	  int y, x;
	  unsigned short sad = 0;

	  /* Each pixel */
	  for (y = 0; y < 4; y++) {
	    const int fy = blk_y_off + y;
	    const unsigned short *frame_row = frame + fy * width;
	    const int ry_unclamped = off_y + fy;
	    int ry = ry_unclamped;
	    if (ry < 0) {
	      ry = 0;
	    } else if (ry >= height) {
	      ry = height - 1;
	    }
	    const int ref_row_off = ry * width;

	    for (x = 0; x < 4; x++) {
	      const int fx = blk_x_off + x;
	      const int rx_unclamped = off_x + fx;
	      int rx = rx_unclamped;
	      if (rx < 0) {
		rx = 0;
	      } else if (rx >= width) {
		rx = width - 1;
	      }

	      const unsigned int a = frame_row[fx];
	      const unsigned int b = ref[ref_row_off + rx];

	      sad += (unsigned short)((a > b) ? (a - b) : (b - a));
	    }
	  }

	  /* Save the SAD */
	  sad_base[(4 * blky + blkx) * MAX_POS_PADDED + pos] = sad;
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

  const int sad7_off = SAD_TYPE_7_IX(mbs);
  const int sad6_off = SAD_TYPE_6_IX(mbs);
  const int sad5_off = SAD_TYPE_5_IX(mbs);
  const int sad4_off = SAD_TYPE_4_IX(mbs);
  const int sad3_off = SAD_TYPE_3_IX(mbs);
  const int sad2_off = SAD_TYPE_2_IX(mbs);
  const int sad1_off = SAD_TYPE_1_IX(mbs);

  for (macroblock = 0; macroblock < mbs; macroblock++)
    {
      const int mb7 = sad7_off + macroblock * stride7;
      const int mb6 = sad6_off + macroblock * stride6;
      const int mb5 = sad5_off + macroblock * stride5;
      const int mb4 = sad4_off + macroblock * stride4;
      const int mb3 = sad3_off + macroblock * stride3;
      const int mb2 = sad2_off + macroblock * stride2;
      const int mb1 = sad1_off + macroblock * SAD_TYPE_1_CT * MAX_POS_PADDED;

      /* Block type 6 */
      for (block_y = 0; block_y < 2; block_y++)
	for (block_x = 0; block_x < 4; block_x++)
	  {
	    x = sads + mb7 +
	      (8 * block_y + block_x) * MAX_POS_PADDED;
	    y = x + 4 * MAX_POS_PADDED;
	    z = sads + mb6 +
	      (4 * block_y + block_x) * MAX_POS_PADDED;

	    for (count = 0; count < MAX_POS; count++) {
	      z[count] = (unsigned short)(x[count] + y[count]);
	    }
	  }

      /* Block type 5 */
      for (block_y = 0; block_y < 4; block_y++)
	for (block_x = 0; block_x < 2; block_x++)
	  {
	    x = sads + mb7 +
	      (4 * block_y + 2 * block_x) * MAX_POS_PADDED;
	    y = x + MAX_POS_PADDED;
	    z = sads + sad5_off +
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
	    x = sads + mb5 +
	      (4 * block_y + block_x) * MAX_POS_PADDED;
	    y = x + 2 * MAX_POS_PADDED;
	    z = sads + mb4 +
	      (2 * block_y + block_x) * MAX_POS_PADDED;
	    
	    for (count = 0; count < MAX_POS; count++) {
	      z[count] = (unsigned short)(x[count] + y[count]);
	    }
	  }
      
      /* Block type 3 */
      x = sads + mb4;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + mb3;
      
      for (count = 0; count < MAX_POS; count++) {
	z[count] = (unsigned short)(x[count] + y[count]);
      }

      x = sads + mb4 +
	MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + mb3 +
	MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) {
	z[count] = (unsigned short)(x[count] + y[count]);
      }

      /* Block type 2 */
      x = sads + mb4;
      y = x + MAX_POS_PADDED;
      z = sads + mb2;

      for (count = 0; count < MAX_POS; count++) {
	z[count] = (unsigned short)(x[count] + y[count]);
      }

      x = sads + mb4 +
	2 * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + mb2 +
	MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) {
	z[count] = (unsigned short)(x[count] + y[count]);
      }

      /* Block type 1 */
      x = sads + mb2;
      y = x + MAX_POS_PADDED;
      z = sads + mb1;

      for (count = 0; count < MAX_POS; count++) {
	z[count] = (unsigned short)(x[count] + y[count]);
      }
    }
}


