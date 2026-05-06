#include <stdlib.h>
#include "sad.h"

static void sad4_one_macroblock(unsigned short *blk_sad,
				const unsigned short *frame,
				const unsigned short *ref,
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
  const int mb_per_row = mb_width;
  const int mb_per_col = mb_height;
  const int mb_count = mb_per_row * mb_per_col;
  const int mb_stride = SAD_TYPE_7_CT * MAX_POS_PADDED;
  unsigned short *blk_sad_base = blk_sad + SAD_TYPE_7_IX(mb_count);

  for (mb_y = 0, frame_yoff = 0;
       mb_y < mb_height;
       mb_y++, frame_yoff += 256 * mb_width)
    {
      const int mb_y16 = mb_y * 16;
      const unsigned short *frame_row = frame + frame_yoff;
      unsigned short *blk_sad_row = blk_sad_base + mb_y * mb_width * mb_stride;

      for (mb_x = 0; mb_x < mb_width; mb_x++)
	{
	  const int mb_x16 = mb_x * 16;
	  sad4_one_macroblock(
	      blk_sad_row + mb_x * mb_stride,
	      frame_row + mb_x16,
	      ref,
	      mb_y16,
	      mb_x16,
	      mb_width,
	      mb_height);
	}
    }
}

void
sad4_one_macroblock(unsigned short *macroblock_sad,
		    const unsigned short *frame,
		    const unsigned short *ref,
		    int frame_y,
		    int frame_x,
		    int mb_width,
		    int mb_height)
{
  int pos_x, pos_y;
  const int width  = mb_width * 16;
  const int height = mb_height * 16;
  int pos = 0;			/* search position */

  const int frame_x_base = frame_x;
  const int frame_y_base = frame_y;

  for (pos_y = -SEARCH_RANGE; pos_y <= SEARCH_RANGE; pos_y++) {
    const int off_y = frame_y_base + pos_y;
    for (pos_x = -SEARCH_RANGE; pos_x <= SEARCH_RANGE; pos_x++, pos++) {
      const int off_x = frame_x_base + pos_x;
      int blky;

      for (blky = 0; blky < 4; blky++) {
	int blkx;

	const int blk_y_base = blky * 4;
	for (blkx = 0; blkx < 4; blkx++) {
	  int y;
	  const int blk_x_base = blkx * 4;
	  unsigned int sad = 0;

	  for (y = 0; y < 4; y++) {
	    int x;
	    const int fy = blk_y_base + y;
	    const unsigned int frame_row_index = (unsigned int)fy * (unsigned int)width;
	    const int ref_y_unclipped = off_y + fy;
	    int ref_y = ref_y_unclipped;
	    if (ref_y < 0) {
	      ref_y = 0;
	    } else if (ref_y >= height) {
	      ref_y = height - 1;
	    }
	    const unsigned int ref_row_index = (unsigned int)ref_y * (unsigned int)width;

	    for (x = 0; x < 4; x++) {
	      const int fx = blk_x_base + x;
	      const int ref_x_unclipped = off_x + fx;
	      int ref_x = ref_x_unclipped;
	      if (ref_x < 0) {
		ref_x = 0;
	      } else if (ref_x >= width) {
		ref_x = width - 1;
	      }

	      const unsigned int a = frame[frame_row_index + (unsigned int)fx];
	      const unsigned int b = ref[ref_row_index + (unsigned int)ref_x];
	      const int diff = (int)a - (int)b;
	      sad += (unsigned int)(diff >= 0 ? diff : -diff);
	    }
	  }

	  macroblock_sad[(4 * blky + blkx) * MAX_POS_PADDED + pos] = (unsigned short)sad;
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

  const int sad7_ix = SAD_TYPE_7_IX(mbs);
  const int sad6_ix = SAD_TYPE_6_IX(mbs);
  const int sad5_ix = SAD_TYPE_5_IX(mbs);
  const int sad4_ix = SAD_TYPE_4_IX(mbs);
  const int sad3_ix = SAD_TYPE_3_IX(mbs);
  const int sad2_ix = SAD_TYPE_2_IX(mbs);
  const int sad1_ix = SAD_TYPE_1_IX(mbs);

  const int sad7_stride = SAD_TYPE_7_CT * MAX_POS_PADDED;
  const int sad6_stride = SAD_TYPE_6_CT * MAX_POS_PADDED;
  const int sad5_stride = SAD_TYPE_5_CT * MAX_POS_PADDED;
  const int sad4_stride = SAD_TYPE_4_CT * MAX_POS_PADDED;
  const int sad3_stride = SAD_TYPE_3_CT * MAX_POS_PADDED;
  const int sad2_stride = SAD_TYPE_2_CT * MAX_POS_PADDED;
  const int sad1_stride = SAD_TYPE_1_CT * MAX_POS_PADDED;

  for (macroblock = 0; macroblock < mbs; macroblock++)
    {
      const int mb7_off = sad7_ix + macroblock * sad7_stride;
      const int mb6_off = sad6_ix + macroblock * sad6_stride;
      const int mb5_off = sad5_ix + macroblock * sad5_stride;
      const int mb4_off = sad4_ix + macroblock * sad4_stride;
      const int mb3_off = sad3_ix + macroblock * sad3_stride;
      const int mb2_off = sad2_ix + macroblock * sad2_stride;
      const int mb1_off = sad1_ix + macroblock * sad1_stride;

      /* Block type 6 */
      for (block_y = 0; block_y < 2; block_y++)
	for (block_x = 0; block_x < 4; block_x++)
	  {
	    x = sads + mb7_off +
	      (8 * block_y + block_x) * MAX_POS_PADDED;
	    y = x + 4 * MAX_POS_PADDED;
	    z = sads + mb6_off +
	      (4 * block_y + block_x) * MAX_POS_PADDED;

	    for (count = 0; count < MAX_POS; count++) {
	      z[count] = x[count] + y[count];
	    }
	  }

      /* Block type 5 */
      for (block_y = 0; block_y < 4; block_y++)
	for (block_x = 0; block_x < 2; block_x++)
	  {
	    x = sads + mb7_off +
	      (4 * block_y + 2 * block_x) * MAX_POS_PADDED;
	    y = x + MAX_POS_PADDED;
	    z = sads + mb5_off +
	      (2 * block_y + block_x) * MAX_POS_PADDED;

	    for (count = 0; count < MAX_POS; count++) {
	      z[count] = x[count] + y[count];
	    }
	  }

      /* Block type 4 */
      for (block_y = 0; block_y < 2; block_y++)
	for (block_x = 0; block_x < 2; block_x++)
	  {
	    x = sads + mb5_off +
	      (4 * block_y + block_x) * MAX_POS_PADDED;
	    y = x + 2 * MAX_POS_PADDED;
	    z = sads + mb4_off +
	      (2 * block_y + block_x) * MAX_POS_PADDED;
	    
	    for (count = 0; count < MAX_POS; count++) {
	      z[count] = x[count] + y[count];
	    }
	  }
      
      /* Block type 3 */
      x = sads + mb4_off;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + mb3_off;
      
      for (count = 0; count < MAX_POS; count++) {
	z[count] = x[count] + y[count];
      }

      x = sads + mb4_off +
	MAX_POS_PADDED;
      y = x + 2 * MAX_POS_PADDED;
      z = sads + mb3_off +
	MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) {
	z[count] = x[count] + y[count];
      }

      /* Block type 2 */
      x = sads + mb4_off;
      y = x + MAX_POS_PADDED;
      z = sads + mb2_off;

      for (count = 0; count < MAX_POS; count++) {
	z[count] = x[count] + y[count];
      }

      x = sads + mb4_off +
	2 * MAX_POS_PADDED;
      y = x + MAX_POS_PADDED;
      z = sads + mb2_off +
	MAX_POS_PADDED;

      for (count = 0; count < MAX_POS; count++) {
	z[count] = x[count] + y[count];
      }

      /* Block type 1 */
      x = sads + mb2_off;
      y = x + MAX_POS_PADDED;
      z = sads + mb1_off;

      for (count = 0; count < MAX_POS; count++) {
	z[count] = x[count] + y[count];
      }
    }
}


