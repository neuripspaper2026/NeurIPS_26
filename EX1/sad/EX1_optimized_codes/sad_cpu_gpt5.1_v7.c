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
      unsigned short *frame_row = frame + frame_yoff;
      for (mb_x = 0; mb_x < mb_width; mb_x++)
	{
	  sad4_one_macroblock
	    (blk_sad + (mb_y * mb_width + mb_x) * (SAD_TYPE_7_CT * MAX_POS_PADDED),
	     frame_row + (mb_x * 16),
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
  int pos = 0;			/* search position */

  /* Each search position */
  for (pos_y = -SEARCH_RANGE; pos_y <= SEARCH_RANGE; pos_y++) {
    const int base_ref_y = frame_y + pos_y;
    for (pos_x = -SEARCH_RANGE; pos_x <= SEARCH_RANGE; pos_x++, pos++) {
      const int base_ref_x = frame_x + pos_x;
      int blky, blkx;

      /* Each 4x4 block in the macroblock */
      for (blky = 0; blky < 4; blky++) {
	const int blk_y_off = blky * 4;
	for (blkx = 0; blkx < 4; blkx++) {
	  const int blk_x_off = blkx * 4;
	  int y, x;
	  unsigned int sad = 0;
	  const unsigned short *frame_blk = frame + blk_y_off * width + blk_x_off;
	  const int out_index = MAX_POS_PADDED * (4 * blky + blkx) + pos;

	  /* Each pixel */
	  for (y = 0; y < 4; y++) {
	    const unsigned short *frame_row = frame_blk + y * width;
	    const int ref_y0 = base_ref_y + blk_y_off + y;
	    int ref_y = ref_y0;
	    if (ref_y < 0) {
	      ref_y = 0;
	    } else if (ref_y >= height) {
	      ref_y = height - 1;
	    }
	    const int ref_row_off = ref_y * width;
	    for (x = 0; x < 4; x++) {
	      const int ref_x0 = base_ref_x + blk_x_off + x;
	      int ref_x = ref_x0;
	      if (ref_x < 0) {
		ref_x = 0;
	      } else if (ref_x >= width) {
		ref_x = width - 1;
	      }

	      const unsigned int b = ref[ref_row_off + ref_x];
	      const unsigned int a = frame_row[x];

	      sad += (a > b) ? (a - b) : (b - a);
	    }
	  }

	  /* Save the SAD */
	  macroblock_sad[out_index] = (unsigned short)sad;
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

  const int max_pos_local = MAX_POS;
  const int max_pos_padded_local = MAX_POS_PADDED;
  const int sad7_ct_local = SAD_TYPE_7_CT;
  const int sad6_ct_local = SAD_TYPE_6_CT;
  const int sad5_ct_local = SAD_TYPE_5_CT;
  const int sad4_ct_local = SAD_TYPE_4_CT;
  const int sad3_ct_local = SAD_TYPE_3_CT;
  const int sad2_ct_local = SAD_TYPE_2_CT;
  const int sad1_ct_local = SAD_TYPE_1_CT;

  const int sad7_ix_mbs = SAD_TYPE_7_IX(mbs);
  const int sad6_ix_mbs = SAD_TYPE_6_IX(mbs);
  const int sad5_ix_mbs = SAD_TYPE_5_IX(mbs);
  const int sad4_ix_mbs = SAD_TYPE_4_IX(mbs);
  const int sad3_ix_mbs = SAD_TYPE_3_IX(mbs);
  const int sad2_ix_mbs = SAD_TYPE_2_IX(mbs);
  const int sad1_ix_mbs = SAD_TYPE_1_IX(mbs);

  for (macroblock = 0; macroblock < mbs; macroblock++)
    {
      const int mb_sad7_base = sad7_ix_mbs + macroblock * sad7_ct_local * max_pos_padded_local;
      const int mb_sad6_base = sad6_ix_mbs + macroblock * sad6_ct_local * max_pos_padded_local;
      const int mb_sad5_base = sad5_ix_mbs + macroblock * sad5_ct_local * max_pos_padded_local;
      const int mb_sad4_base = sad4_ix_mbs + macroblock * sad4_ct_local * max_pos_padded_local;
      const int mb_sad3_base = sad3_ix_mbs + macroblock * sad3_ct_local * max_pos_padded_local;
      const int mb_sad2_base = sad2_ix_mbs + macroblock * sad2_ct_local * max_pos_padded_local;
      const int mb_sad1_base = sad1_ix_mbs + macroblock * sad1_ct_local * max_pos_padded_local;

      /* Block type 6 */
      for (block_y = 0; block_y < 2; block_y++)
	for (block_x = 0; block_x < 4; block_x++)
	  {
	    const int blk_index_7 = (8 * block_y + block_x) * max_pos_padded_local;
	    const int blk_index_6 = (4 * block_y + block_x) * max_pos_padded_local;

	    x = sads + mb_sad7_base + blk_index_7;
	    y = x + 4 * max_pos_padded_local;
	    z = sads + mb_sad6_base + blk_index_6;

	    for (count = 0; count < max_pos_local; count++) {
	      z[count] = (unsigned short)(x[count] + y[count]);
	    }
	  }

      /* Block type 5 */
      for (block_y = 0; block_y < 4; block_y++)
	for (block_x = 0; block_x < 2; block_x++)
	  {
	    const int blk_index_7 = (4 * block_y + 2 * block_x) * max_pos_padded_local;
	    const int blk_index_5 = (2 * block_y + block_x) * max_pos_padded_local;

	    x = sads + mb_sad7_base + blk_index_7;
	    y = x + max_pos_padded_local;
	    z = sads + mb_sad5_base + blk_index_5;

	    for (count = 0; count < max_pos_local; count++) {
	      z[count] = (unsigned short)(x[count] + y[count]);
	    }
	  }

      /* Block type 4 */
      for (block_y = 0; block_y < 2; block_y++)
	for (block_x = 0; block_x < 2; block_x++)
	  {
	    const int blk_index_5 = (4 * block_y + block_x) * max_pos_padded_local;
	    const int blk_index_4 = (2 * block_y + block_x) * max_pos_padded_local;

	    x = sads + mb_sad5_base + blk_index_5;
	    y = x + 2 * max_pos_padded_local;
	    z = sads + mb_sad4_base + blk_index_4;
	    
	    for (count = 0; count < max_pos_local; count++) {
	      z[count] = (unsigned short)(x[count] + y[count]);
	    }
	  }
      
      /* Block type 3 */
      x = sads + mb_sad4_base;
      y = x + 2 * max_pos_padded_local;
      z = sads + mb_sad3_base;
      
      for (count = 0; count < max_pos_local; count++) {
	z[count] = (unsigned short)(x[count] + y[count]);
      }

      x = sads + mb_sad4_base + max_pos_padded_local;
      y = x + 2 * max_pos_padded_local;
      z = sads + mb_sad3_base + max_pos_padded_local;

      for (count = 0; count < max_pos_local; count++) {
	z[count] = (unsigned short)(x[count] + y[count]);
      }

      /* Block type 2 */
      x = sads + mb_sad4_base;
      y = x + max_pos_padded_local;
      z = sads + mb_sad2_base;

      for (count = 0; count < max_pos_local; count++) {
	z[count] = (unsigned short)(x[count] + y[count]);
      }

      x = sads + mb_sad4_base + 2 * max_pos_padded_local;
      y = x + max_pos_padded_local;
      z = sads + mb_sad2_base + max_pos_padded_local;

      for (count = 0; count < max_pos_local; count++) {
	z[count] = (unsigned short)(x[count] + y[count]);
      }

      /* Block type 1 */
      x = sads + mb_sad2_base;
      y = x + max_pos_padded_local;
      z = sads + mb_sad1_base;

      for (count = 0; count < max_pos_local; count++) {
	z[count] = (unsigned short)(x[count] + y[count]);
      }
    }
}


