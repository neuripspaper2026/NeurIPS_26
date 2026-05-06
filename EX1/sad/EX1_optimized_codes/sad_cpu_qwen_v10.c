<<<CODE>>>
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
       mb_y++, frame_yoff += 256 * mb_width)
    {
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
  const int width = mb_width * 16;
  const int height = mb_height * 16;
  const int frame_base_offset = frame_y * width + frame_x;
  
  int pos_y, pos_x;
  int pos = 0;

  /* Each search position */
  for (pos_y = -SEARCH_RANGE; pos_y <= SEARCH_RANGE; pos_y++) {
    const int ref_y_base = frame_y + pos_y;
    const int clamped_ref_y_base = (ref_y_base < 0) ? 0 : ((ref_y_base >= height) ? height - 1 : ref_y_base);
    
    for (pos_x = -SEARCH_RANGE; pos_x <= SEARCH_RANGE; pos_x++, pos++) {
      const int ref_x_base = frame_x + pos_x;
      const int clamped_ref_x_base = (ref_x_base < 0) ? 0 : ((ref_x_base >= width) ? width - 1 : ref_x_base);
      
      int blky, blkx;

      /* Each 4x4 block in the macroblock */
      for (blky = 0; blky < 4; blky++) {
	const int block_y_offset = blky * 4;
	const int ref_y_offset = ref_y_base + block_y_offset;
	const int clamped_ref_y_offset = (ref_y_offset < 0) ? 0 : ((ref_y_offset >= height) ? height - 1 : ref_y_offset);
	const int frame_y_idx = block_y_offset * width;
	
	for (blkx = 0; blkx < 4; blkx++) {
	  const int block_x_offset = blkx * 4;
	  const int ref_x_offset = ref_x_base + block_x_offset;
	  const int clamped_ref_x_offset = (ref_x_offset < 0) ? 0 : ((ref_x_offset >= width) ? width - 1 : ref_x_offset);
	  const int frame_base_idx = frame_y_idx + block_x_offset;
	  
	  int y, x;
	  unsigned short sad = 0;

	  /* Each pixel */
	  for (y = 0; y < 4; y++) {
	    const int ref_y = ref_y_offset + y;
	    const int clamped_ref_y = (ref_y < 0) ? 0 : ((ref_y >= height) ? height - 1 : ref_y);
	    const int frame_y_idx_offset = frame_base_idx + y * width;
	    
	    for (x = 0; x < 4; x++) {
	      const int ref_x = ref_x_offset + x;
	      const int clamped_ref_x = (ref_x < 0) ? 0 : ((ref_x >= width) ? width - 1 : ref_x);
	      
	      const unsigned int b = ref[clamped_ref_y * width + clamped_ref_x];
	      const unsigned int a = frame[frame_y_idx_offset + x];

	      sad += abs((int)a - (int)b);
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

  const int type7_base = SAD_TYPE_7_IX(mbs);
  const int type6_base = SAD_TYPE_6_IX(mbs);
  const int type5_base = SAD_TYPE_5_IX(mbs);
  const int type4_base = SAD_TYPE_4_IX(mbs);
  const int type3_base = SAD_TYPE_3_IX(mbs);
  const int type2_base = SAD_TYPE_2_IX(mbs);
  const int type1_base = SAD_TYPE_1_IX(mbs);
  
  const int type7_macroblock_stride = SAD_TYPE_7_CT * MAX_POS_PADDED;
  const int type6_macroblock_stride = SAD_TYPE_6_CT * MAX_POS_PADDED;
  const int type5_macroblock_stride = SAD_TYPE_5_CT * MAX_POS_PADDED;
  const int type4_macroblock_stride = SAD_TYPE_4_CT * MAX_POS_PADDED;
  const int type3_macroblock_stride = SAD_TYPE_3_CT * MAX_POS_PADDED;
  const int type2_macroblock_stride = SAD_TYPE_2_CT * MAX_POS_PADDED;
  const int type1_macroblock_stride = SAD_TYPE_1_CT * MAX_POS_PADDED;

  for (macroblock = 0; macroblock < mbs; macroblock++)
    {
      const int macroblock_offset_7 = macroblock * type7_macroblock_stride;
      const int macroblock_offset_6 = macroblock * type6_macroblock_stride;
      const int macroblock_offset_5 = macroblock * type5_macroblock_stride;
      const int macroblock_offset_4 = macroblock * type4_macroblock_stride;
      const int macroblock_offset_3 = macroblock * type3_macroblock_stride;
      const int macroblock_offset_2 = macroblock * type2_macroblock_stride;
      const int macroblock_offset_1 = macroblock * type1_macroblock_stride;
      
      /* Block type 6 */
      for (block_y = 0; block_y < 2; block_y++)
	for (block_x = 0; block_x < 4; block_x++)
	  {
	    const int src_idx = (8 * block_y + block_x) * MAX_POS_PADDED;
	    x = sads + type7_base + macroblock_offset_7 + src_idx;
	    y = x + 4 * MAX_POS_PADDED;
	    z = sads + type6_base + macroblock_offset_6 + (4 * block_y + block_x) * MAX_POS_PADDED;

	    for (count = 0; count < MAX_POS; count++) *z++ = *x++ + *y++;
	  }

      /* Block type 5 */
      for (block_y = 0; block_y < 4; block_y++)
	for (block_x = 0; block_x < 2; block_x++)
	  {
	    const int src_idx = (4 * block_y + 2 * block_x) * MAX_POS_PADDED;
	    x = sads + type7_base + macroblock_offset_7 + src_idx;
	    y = x + MAX_POS_PADDED;
	    z = sads + type5_base + macroblock_offset_6 + (2 * block_y + block_x) * MAX_POS_PADDED;

	    for (count = 0; count < MAX_POS; count++) *z++ = *x++ + *y++;
	  }

      /* Block type 4 */
      for (block_y = 0; block_y < 2; block_y++)
	for (block_x = 0; block_x < 2; block_x++)
	  {
	    const int src_idx = (4 * block_y + block_x) * MAX_POS_PADDED;
	    x = sads + type5_base + macroblock_offset_5 + src_idx;
	    y = x + 2 * MAX_POS_PADDED;
	    z = sads + type4_base + macroblock_offset_4 + (2 * block_y + block_x) * MAX_POS_PADDED;
	    
	    for (count = 0; count < MAX_POS; count++) *z++ = *x++ + *y++;
	  }
      
      /* Block type 3 */
      {
	x = sads + type4_base + macroblock_offset_4;
	y = x + 2 * MAX_POS_PADDED;
	z = sads + type3_base + macroblock_offset_3;
	
	for (count = 0; count < MAX_POS; count++) *z++ = *x++ + *y++;

	x = sads + type4_base + macroblock_offset_


