#include "lbm.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#if !defined(SPEC_CPU)
#ifdef _OPENMP
#include <omp.h>
#endif
#endif

#define DFL1 (1.0f/ 3.0f)
#define DFL2 (1.0f/18.0f)
#define DFL3 (1.0f/36.0f)

void LBM_allocateGrid( float** ptr ) {
	const size_t margin = 2u*SIZE_X*SIZE_Y*N_CELL_ENTRIES;
	const size_t size   = sizeof( LBM_Grid ) + 2u*margin*sizeof( float );

	*ptr = (float*)malloc( size );
	if( !*ptr ) {
		printf( "LBM_allocateGrid: could not allocate %.1f MByte\n",
		        (double)size / (1024.0*1024.0) );
		exit( 1 );
	}
#if !defined(SPEC_CPU)
	printf( "LBM_allocateGrid: allocated %.1f MByte\n",
	        (double)size / (1024.0*1024.0) );
#endif
	*ptr += margin;
}

void LBM_freeGrid( float** ptr ) {
	const size_t margin = 2u*SIZE_X*SIZE_Y*N_CELL_ENTRIES;

	free( *ptr - margin );
	*ptr = NULL;
}

void LBM_initializeGrid( LBM_Grid grid ) {
	SWEEP_VAR

	/*voption indep*/
	SWEEP_START( 0, 0, -2, 0, 0, SIZE_Z+2 )
		LOCAL( grid, C  ) = DFL1;
		LOCAL( grid, N  ) = DFL2;
		LOCAL( grid, S  ) = DFL2;
		LOCAL( grid, E  ) = DFL2;
		LOCAL( grid, W  ) = DFL2;
		LOCAL( grid, T  ) = DFL2;
		LOCAL( grid, B  ) = DFL2;
		LOCAL( grid, NE ) = DFL3;
		LOCAL( grid, NW ) = DFL3;
		LOCAL( grid, SE ) = DFL3;
		LOCAL( grid, SW ) = DFL3;
		LOCAL( grid, NT ) = DFL3;
		LOCAL( grid, NB ) = DFL3;
		LOCAL( grid, ST ) = DFL3;
		LOCAL( grid, SB ) = DFL3;
		LOCAL( grid, ET ) = DFL3;
		LOCAL( grid, EB ) = DFL3;
		LOCAL( grid, WT ) = DFL3;
		LOCAL( grid, WB ) = DFL3;

		CLEAR_ALL_FLAGS_SWEEP( grid );
	SWEEP_END
}

void LBM_swapGrids( LBM_GridPtr* grid1, LBM_GridPtr* grid2 ) {
	LBM_GridPtr aux = *grid1;
	*grid1 = *grid2;
	*grid2 = aux;
}

void LBM_loadObstacleFile( LBM_Grid grid, const char* filename ) {
	int x,  y,  z;

	FILE* file = fopen( filename, "rb" );
	if (!file) {
		perror("LBM_loadObstacleFile");
		exit(1);
	}

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				const int c = fgetc( file );
				if( c != '.' && c != EOF )
					SET_FLAG( grid, x, y, z, OBSTACLE );
			}
			(void)fgetc( file );
		}
		(void)fgetc( file );
	}

	fclose( file );
}

void LBM_initializeSpecialCellsForLDC( LBM_Grid grid ) {
	int x,  y,  z;

	/*voption indep*/
	for( z = -2; z < SIZE_Z+2; z++ ) {
		const int z_is_inner = (z == 1 || z == SIZE_Z-2);
		const int z_is_border = (z == 0 || z == SIZE_Z-1);
		for( y = 0; y < SIZE_Y; y++ ) {
			const int y_is_border = (y == 0 || y == SIZE_Y-1);
			const int y_in_inner_band = (y > 1 && y < SIZE_Y-2);
			for( x = 0; x < SIZE_X; x++ ) {
				const int x_is_border = (x == 0 || x == SIZE_X-1);
				if( x_is_border || y_is_border || z_is_border ) {
					SET_FLAG( grid, x, y, z, OBSTACLE );
				}
				else if( z_is_inner &&
				         x > 1 && x < SIZE_X-2 &&
				         y_in_inner_band ) {
					SET_FLAG( grid, x, y, z, ACCEL );
				}
			}
		}
	}
}

void LBM_initializeSpecialCellsForChannel( LBM_Grid grid ) {
	int x,  y,  z;

	/*voption indep*/
	for( z = -2; z < SIZE_Z+2; z++ ) {
		const int z_is_io = (z == 0 || z == SIZE_Z-1);
		for( y = 0; y < SIZE_Y; y++ ) {
			const int y_is_border = (y == 0 || y == SIZE_Y-1);
			for( x = 0; x < SIZE_X; x++ ) {
				const int x_is_border = (x == 0 || x == SIZE_X-1);
				if( x_is_border || y_is_border ) {
					SET_FLAG( grid, x, y, z, OBSTACLE );

					if( z_is_io &&
					    !TEST_FLAG( grid, x, y, z, OBSTACLE ) )
						SET_FLAG( grid, x, y, z, IN_OUT_FLOW );
				}
			}
		}
	}
}

void LBM_performStreamCollide( LBM_Grid srcGrid, LBM_Grid dstGrid ) {
	SWEEP_VAR

	float ux, uy, uz, u2, rho;

	/*voption indep*/
	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
			const float src_c  = SRC_C ( srcGrid );
			const float src_n  = SRC_N ( srcGrid );
			const float src_s  = SRC_S ( srcGrid );
			const float src_e  = SRC_E ( srcGrid );
			const float src_w  = SRC_W ( srcGrid );
			const float src_t  = SRC_T ( srcGrid );
			const float src_b  = SRC_B ( srcGrid );
			const float src_ne = SRC_NE( srcGrid );
			const float src_nw = SRC_NW( srcGrid );
			const float src_se = SRC_SE( srcGrid );
			const float src_sw = SRC_SW( srcGrid );
			const float src_nt = SRC_NT( srcGrid );
			const float src_nb = SRC_NB( srcGrid );
			const float src_st = SRC_ST( srcGrid );
			const float src_sb = SRC_SB( srcGrid );
			const float src_et = SRC_ET( srcGrid );
			const float src_eb = SRC_EB( srcGrid );
			const float src_wt = SRC_WT( srcGrid );
			const float src_wb = SRC_WB( srcGrid );

			DST_C ( dstGrid ) = src_c;
			DST_S ( dstGrid ) = src_n;
			DST_N ( dstGrid ) = src_s;
			DST_W ( dstGrid ) = src_e;
			DST_E ( dstGrid ) = src_w;
			DST_B ( dstGrid ) = src_t;
			DST_T ( dstGrid ) = src_b;
			DST_SW( dstGrid ) = src_ne;
			DST_SE( dstGrid ) = src_nw;
			DST_NW( dstGrid ) = src_se;
			DST_NE( dstGrid ) = src_sw;
			DST_SB( dstGrid ) = src_nt;
			DST_ST( dstGrid ) = src_nb;
			DST_NB( dstGrid ) = src_st;
			DST_NT( dstGrid ) = src_sb;
			DST_WB( dstGrid ) = src_et;
			DST_WT( dstGrid ) = src_eb;
			DST_EB( dstGrid ) = src_wt;
			DST_ET( dstGrid ) = src_wb;
			continue;
		}

		{
			const float src_c  = SRC_C ( srcGrid );
			const float src_n  = SRC_N ( srcGrid );
			const float src_s  = SRC_S ( srcGrid );
			const float src_e  = SRC_E ( srcGrid );
			const float src_w  = SRC_W ( srcGrid );
			const float src_t  = SRC_T ( srcGrid );
			const float src_b  = SRC_B ( srcGrid );
			const float src_ne = SRC_NE( srcGrid );
			const float src_nw = SRC_NW( srcGrid );
			const float src_se = SRC_SE( srcGrid );
			const float src_sw = SRC_SW( srcGrid );
			const float src_nt = SRC_NT( srcGrid );
			const float src_nb = SRC_NB( srcGrid );
			const float src_st = SRC_ST( srcGrid );
			const float src_sb = SRC_SB( srcGrid );
			const float src_et = SRC_ET( srcGrid );
			const float src_eb = SRC_EB( srcGrid );
			const float src_wt = SRC_WT( srcGrid );
			const float src_wb = SRC_WB( srcGrid );

			rho = src_c + src_n + src_s + src_e + src_w + src_t + src_b +
			      src_ne + src_nw + src_se + src_sw +
			      src_nt + src_nb + src_st + src_sb +
			      src_et + src_eb + src_wt + src_wb;

			ux = + src_e - src_w
			     + src_ne - src_nw
			     + src_se - src_sw
			     + src_et + src_eb
			     - src_wt - src_wb;
			uy = + src_n - src_s
			     + src_ne + src_nw
			     - src_se - src_sw
			     + src_nt + src_nb
			     - src_st - src_sb;
			uz = + src_t - src_b
			     + src_nt - src_nb
			     + src_st - src_sb
			     + src_et - src_eb
			     + src_wt - src_wb;

			const float inv_rho = 1.0f / rho;
			ux *= inv_rho;
			uy *= inv_rho;
			uz *= inv_rho;

			if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
				ux = 0.005f;
				uy = 0.002f;
				uz = 0.000f;
			}

			u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

			const float omega_rho = OMEGA * rho;
			const float om1       = 1.0f - OMEGA;

			DST_C ( dstGrid ) = om1*src_c + DFL1*omega_rho*(1.0f                                 - u2);

			DST_N ( dstGrid ) = om1*src_n + DFL2*omega_rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2);
			DST_S ( dstGrid ) = om1*src_s + DFL2*omega_rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2);
			DST_E ( dstGrid ) = om1*src_e + DFL2*omega_rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2);
			DST_W ( dstGrid ) = om1*src_w + DFL2*omega_rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2);
			DST_T ( dstGrid ) = om1*src_t + DFL2*omega_rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2);
			DST_B ( dstGrid ) = om1*src_b + DFL2*omega_rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2);

			const float uxpuy  = ux + uy;
			const float uxmuy  = ux - uy;
			const float mxpuy  = -ux + uy;
			const float mxmuy  = -ux - uy;
			const float uypuz  = uy + uz;
			const float uymuz  = uy - uz;
			const float mypuz  = -uy + uz;
			const float mymuz  = -uy - uz;
			const float uxpuz  = ux + uz;
			const float uxmuz  = ux - uz;
			const float mxpuz2 = -ux + uz;
			const float mxmuz  = -ux - uz;

			DST_NE( dstGrid ) = om1*src_ne + DFL3*omega_rho*(1.0f + uxpuy *(4.5f*uxpuy  + 3.0f) - u2);
			DST_NW( dstGrid ) = om1*src_nw + DFL3*omega_rho*(1.0f + mxpuy *(4.5f*mxpuy  + 3.0f) - u2);
			DST_SE( dstGrid ) = om1*src_se + DFL3*omega_rho*(1.0f + uxmuy *(4.5f*uxmuy  + 3.0f) - u2);
			DST_SW( dstGrid ) = om1*src_sw + DFL3*omega_rho*(1.0f + mxmuy *(4.5f*mxmuy  + 3.0f) - u2);
			DST_NT( dstGrid ) = om1*src_nt + DFL3*omega_rho*(1.0f + uypuz *(4.5f*uypuz  + 3.0f) - u2);
			DST_NB( dstGrid ) = om1*src_nb + DFL3*omega_rho*(1.0f + uymuz *(4.5f*uymuz  + 3.0f) - u2);
			DST_ST( dstGrid ) = om1*src_st + DFL3*omega_rho*(1.0f + mypuz *(4.5f*mypuz  + 3.0f) - u2);
			DST_SB( dstGrid ) = om1*src_sb + DFL3*omega_rho*(1.0f + mymuz *(4.5f*mymuz  + 3.0f) - u2);
			DST_ET( dstGrid ) = om1*src_et + DFL3*omega_rho*(1.0f + uxpuz *(4.5f*uxpuz  + 3.0f) - u2);
			DST_EB( dstGrid ) = om1*src_eb + DFL3*omega_rho*(1.0f + uxmuz *(4.5f*uxmuz  + 3.0f) - u2);
			DST_WT( dstGrid ) = om1*src_wt + DFL3*omega_rho*(1.0f + mxpuz2*(4.5f*mxpuz2 + 3.0f) - u2);
			DST_WB( dstGrid ) = om1*src_wb + DFL3*omega_rho*(1.0f + mxmuz *(4.5f*mxmuz  + 3.0f) - u2);
		}
	SWEEP_END
}

void LBM_handleInOutFlow( LBM_Grid srcGrid ) {
	float ux , uy , uz , rho ,
	       ux1, uy1, uz1, rho1,
	       ux2, uy2, uz2, rho2,
	       u2, px, py;
	SWEEP_VAR

	/* inflow */
	/*voption indep*/
	SWEEP_START( 0, 0, 0, 0, 0, 1 )
		{
			const float c1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, C  );
			const float n1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, N  );
			const float s1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, S  );
			const float e1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, E  );
			const float w1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, W  );
			const float t1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, T  );
			const float b1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, B  );
			const float ne1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NE );
			const float nw1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NW );
			const float se1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SE );
			const float sw1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SW );
			const float nt1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NT );
			const float nb1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NB );
			const float st1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ST );
			const float sb1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SB );
			const float et1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ET );
			const float eb1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, EB );
			const float wt1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WT );
			const float wb1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WB );

			rho1 = c1 + n1 + s1 + e1 + w1 + t1 + b1 +
			       ne1 + nw1 + se1 + sw1 +
			       nt1 + nb1 + st1 + sb1 +
			       et1 + eb1 + wt1 + wb1;

			const float c2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, C  );
			const float n2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, N  );
			const float s2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, S  );
			const float e2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, E  );
			const float w2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, W  );
			const float t2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, T  );
			const float b2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, B  );
			const float ne2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NE );
			const float nw2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NW );
			const float se2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SE );
			const float sw2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SW );
			const float nt2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NT );
			const float nb2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NB );
			const float st2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ST );
			const float sb2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SB );
			const float et2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ET );
			const float eb2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, EB );
			const float wt2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WT );
			const float wb2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WB );

			rho2 = c2 + n2 + s2 + e2 + w2 + t2 + b2 +
			       ne2 + nw2 + se2 + sw2 +
			       nt2 + nb2 + st2 + sb2 +
			       et2 + eb2 + wt2 + wb2;
		}

		rho = 2.0f*rho1 - rho2;

		px = (float)SWEEP_X / (0.5f*(SIZE_X-1)) - 1.0f;
		py = (float)SWEEP_Y / (0.5f*(SIZE_Y-1)) - 1.0f;
		ux = 0.00f;
		uy = 0.00f;
		uz = 0.01f * (1.0f-px*px) * (1.0f-py*py);

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = DFL1*rho*(1.0f                                 - u2);

		LOCAL( srcGrid, N ) = DFL2*rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2);
		LOCAL( srcGrid, S ) = DFL2*rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2);
		LOCAL( srcGrid, E ) = DFL2*rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2);
		LOCAL( srcGrid, W ) = DFL2*rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2);
		LOCAL( srcGrid, T ) = DFL2*rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2);
		LOCAL( srcGrid, B ) = DFL2*rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2);

		LOCAL( srcGrid, NE) = DFL3*rho*(1.0f + (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
		LOCAL( srcGrid, NW) = DFL3*rho*(1.0f + (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
		LOCAL( srcGrid, SE) = DFL3*rho*(1.0f + (+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
		LOCAL( srcGrid, SW) = DFL3*rho*(1.0f + (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
		LOCAL( srcGrid, NT) = DFL3*rho*(1.0f + (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
		LOCAL( srcGrid, NB) = DFL3*rho*(1.0f + (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
		LOCAL( srcGrid, ST) = DFL3*rho*(1.0f + (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
		LOCAL( srcGrid, SB) = DFL3*rho*(1.0f + (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
		LOCAL( srcGrid, ET) = DFL3*rho*(1.0f + (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
		LOCAL( srcGrid, EB) = DFL3*rho*(1.0f + (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
		LOCAL( srcGrid, WT) = DFL3*rho*(1.0f + (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
		LOCAL( srcGrid, WB) = DFL3*rho*(1.0f + (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
	SWEEP_END

	/* outflow */
	/*voption indep*/
	SWEEP_START( 0, 0, SIZE_Z-1, 0, 0, SIZE_Z )
		{
			const float cm1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, C  );
			const float nm1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, N  );
			const float sm1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, S  );
			const float em1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, E  );
			const float wm1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, W  );
			const float tm1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, T  );
			const float bm1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, B  );
			const float nem1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE );
			const float nwm1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW );
			const float sem1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE );
			const float swm1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW );
			const float ntm1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT );
			const float nbm1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB );
			const float stm1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST );
			const float sbm1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB );
			const float etm1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET );
			const float ebm1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB );
			const float wtm1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT );
			const float wbm1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );

			rho1 = cm1 + nm1 + sm1 + em1 + wm1 + tm1 + bm1 +
			       nem1 + nwm1 + sem1 + swm1 +
			       ntm1 + nbm1 + stm1 + sbm1 +
			       etm1 + ebm1 + wtm1 + wbm1;
			ux1 = + em1 - wm1
			      + nem1 - nwm1
			      + sem1 - swm1
			      + etm1 + ebm1
			      - wtm1 - wbm1;
			uy1 = + nm1 - sm1
			      + nem1 + nwm1
			      - sem1 - swm1
			      + ntm1 + nbm1
			      - stm1 - sbm1;
			uz1 = + tm1 - bm1
			      + ntm1 - nbm1
			      + stm1 - sbm1
			      + etm1 - ebm1
			      + wtm1 - wbm1;

			{
				const float cm2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, C  );
				const float nm2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, N  );
				const float sm2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, S  );
				const float em2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, E  );
				const float wm2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, W  );
				const float tm2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, T  );
				const float bm2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, B  );
				const float nem2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE );
				const float nwm2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW );
				const float sem2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE );
				const float swm2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW );
				const float ntm2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT );
				const float nbm2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB );
				const float stm2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST );
				const float sbm2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB );
				const float etm2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET );
				const float ebm2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB );
				const float wtm2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT );
				const float wbm2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );

				rho2 = cm2 + nm2 + sm2 + em2 + wm2 + tm2 + bm2 +
				       nem2 + nwm2 + sem2 + swm2 +
				       ntm2 + nbm2 + stm2 + sbm2 +
				       etm2 + ebm2 + wtm2 + wbm2;
				ux2 = + em2 - wm2
				      + nem2 - nwm2
				      + sem2 - swm2
				      + etm2 + ebm2
				      - wtm2 - wbm2;
				uy2 = + nm2 - sm2
				      + nem2 + nwm2
				      - sem2 - swm2
				      + ntm2 + nbm2
				      - stm2 - sbm2;
				uz2 = + tm2 - bm2
				      + ntm2 - nbm2
				      + stm2 - sbm2
				      + etm2 - ebm2
				      + wtm2 - wbm2;
			}
		}

		{
			const float inv_rho1 = 1.0f / rho1;
			const float inv_rho2 = 1.0f / rho2;
			ux1 *= inv_rho1;
			uy1 *= inv_rho1;
			uz1 *= inv_rho1;
			ux2 *= inv_rho2;
			uy2 *= inv_rho2;
			uz2 *= inv_rho2;
		}

		rho = 1.0f;

		ux = 2.0f*ux1 - ux2;
		uy = 2.0f*uy1 - uy2;
		uz = 2.0f*uz1 - uz2;

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = DFL1*rho*(1.0f                                 - u2);

		LOCAL( srcGrid, N ) = DFL2*rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2);
		LOCAL( srcGrid, S ) = DFL2*rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2);
		LOCAL( srcGrid, E ) = DFL2*rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2);
		LOCAL( srcGrid, W ) = DFL2*rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2);
		LOCAL( srcGrid, T ) = DFL2*rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2);
		LOCAL( srcGrid, B ) = DFL2*rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2);

		LOCAL( srcGrid, NE) = DFL3*rho*(1.0f + (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
		LOCAL( srcGrid, NW) = DFL3*rho*(1.0f + (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
		LOCAL( srcGrid, SE) = DFL3*rho*(1.0f + (+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
		LOCAL( srcGrid, SW) = DFL3*rho*(1.0f + (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
		LOCAL( srcGrid, NT) = DFL3*rho*(1.0f + (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
		LOCAL( srcGrid, NB) = DFL3*rho*(1.0f + (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
		LOCAL( srcGrid, ST) = DFL3*rho*(1.0f + (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
		LOCAL( srcGrid, SB) = DFL3*rho*(1.0f + (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
		LOCAL( srcGrid, ET) = DFL3*rho*(1.0f + (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
		LOCAL( srcGrid, EB) = DFL3*rho*(1.0f + (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
		LOCAL( srcGrid, WT) = DFL3*rho*(1.0f + (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
		LOCAL( srcGrid, WB) = DFL3*rho*(1.0f + (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
	SWEEP_END
}

/*############################################################################*/

void LBM_showGridStatistics( LBM_Grid grid ) {
	int nObstacleCells = 0,
	    nAccelCells    = 0,
	    nFluidCells    = 0;
	float ux, uy, uz;
	float minU2  = 1e+30f, maxU2  = -1e+30f, u2;
	float minRho = 1e+30f, maxRho = -1e+30f, rho;
	float mass = 0.0f;

	SWEEP_VAR

	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		{
			const float c  = LOCAL( grid, C  );
			const float n  = LOCAL( grid, N  );
			const float s  = LOCAL( grid, S  );
			const float e  = LOCAL( grid, E  );
			const float w  = LOCAL( grid, W  );
			const float t  = LOCAL( grid, T  );
			const float b  = LOCAL( grid, B  );
			const float ne = LOCAL( grid, NE );
			const float nw = LOCAL( grid, NW );
			const float se = LOCAL( grid, SE );
			const float sw = LOCAL( grid, SW );
			const float nt = LOCAL( grid, NT );
			const float nb = LOCAL( grid, NB );
			const float st = LOCAL( grid, ST );
			const float sb = LOCAL( grid, SB );
			const float et = LOCAL( grid, ET );
			const float eb = LOCAL( grid, EB );
			const float wt = LOCAL( grid, WT );
			const float wb = LOCAL( grid, WB );

			rho = c + n + s + e + w + t + b +
			      ne + nw + se + sw +
			      nt + nb + st + sb +
			      et + eb + wt + wb;
		}
		if( rho < minRho ) minRho = rho;
		if( rho > maxRho ) maxRho = rho;
		mass += rho;

		if( TEST_FLAG_SWEEP( grid, OBSTACLE )) {
			nObstacleCells++;
		}
		else {
			if( TEST_FLAG_SWEEP( grid, ACCEL ))
				nAccelCells++;
			else
				nFluidCells++;

			ux = + LOCAL( grid, E  ) - LOCAL( grid, W  )
			     + LOCAL( grid, NE ) - LOCAL( grid, NW )
			     + LOCAL( grid, SE ) - LOCAL( grid, SW )
			     + LOCAL( grid, ET ) + LOCAL( grid, EB )
			     - LOCAL( grid, WT ) - LOCAL( grid, WB );
			uy = + LOCAL( grid, N  ) - LOCAL( grid, S  )
			     + LOCAL( grid, NE ) + LOCAL( grid, NW )
			     - LOCAL( grid, SE ) - LOCAL( grid, SW )
			     + LOCAL( grid, NT ) + LOCAL( grid, NB )
			     - LOCAL( grid, ST ) - LOCAL( grid, SB );
			uz = + LOCAL( grid, T  ) - LOCAL( grid, B  )
			     + LOCAL( grid, NT ) - LOCAL( grid, NB )
			     + LOCAL( grid, ST ) - LOCAL( grid, SB )
			     + LOCAL( grid, ET ) - LOCAL( grid, EB )
			     + LOCAL( grid, WT ) - LOCAL( grid, WB );
			{
				const float inv_rho = 1.0f / rho;
				const float ux_n = ux * inv_rho;
				const float uy_n = uy * inv_rho;
				const float uz_n = uz * inv_rho;
				u2 = (ux_n*ux_n + uy_n*uy_n + uz_n*uz_n);
			}
			if( u2 < minU2 ) minU2 = u2;
			if( u2 > maxU2 ) maxU2 = u2;
		}
	SWEEP_END

        printf( "LBM_showGridStatistics:\n"
        "\tnObstacleCells: %7i nAccelCells: %7i nFluidCells: %7i\n"
        "\tminRho: %8.4f maxRho: %8.4f mass: %e\n"
        "\tminU: %e maxU: %e\n\n",
        nObstacleCells, nAccelCells, nFluidCells,
        minRho, maxRho, (double)mass,
        sqrtf( minU2 ), sqrtf( maxU2 ) );

}

/*############################################################################*/

static void storeValue( FILE* file, OUTPUT_PRECISION* v ) {
	const int litteBigEndianTest = 1;
	if( (*((unsigned char*) &litteBigEndianTest)) == 0 ) {         /* big endian */
		const char* vPtr = (const char*) v;
		char buffer[sizeof( OUTPUT_PRECISION )];
		size_t i;

		for (i = 0; i < sizeof( OUTPUT_PRECISION ); i++)
			buffer[i] = vPtr[sizeof( OUTPUT_PRECISION ) - i - 1];

		fwrite( buffer, sizeof( OUTPUT_PRECISION ), 1, file );
	}
	else {                                                     /* little endian */
		fwrite( v, sizeof( OUTPUT_PRECISION ), 1, file );
	}
}

/*############################################################################*/

static void loadValue( FILE* file, OUTPUT_PRECISION* v ) {
	const int litteBigEndianTest = 1;
	if( (*((unsigned char*) &litteBigEndianTest)) == 0 ) {         /* big endian */
		char* vPtr = (char*) v;
		char buffer[sizeof( OUTPUT_PRECISION )];
		size_t i;

		fread( buffer, sizeof( OUTPUT_PRECISION ), 1, file );

		for (i = 0; i < sizeof( OUTPUT_PRECISION ); i++)
			vPtr[i] = buffer[sizeof( OUTPUT_PRECISION ) - i - 1];
	}
	else {                                                     /* little endian */
		fread( v, sizeof( OUTPUT_PRECISION ), 1, file );
	}
}

void LBM_storeVelocityField( LBM_Grid grid, const char* filename,
                             const int binary ) {
	int x, y, z;
	OUTPUT_PRECISION rho, ux, uy, uz;

	FILE* file = fopen( filename, (binary ? "wb" : "w") );
	if (!file) {
		perror("LBM_storeVelocityField");
		return;
	}

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				const float c  = GRID_ENTRY( grid, x, y, z, C  );
				const float n  = GRID_ENTRY( grid, x, y, z, N  );
				const float s  = GRID_ENTRY( grid, x, y, z, S  );
				const float e  = GRID_ENTRY( grid, x, y, z, E  );
				const float w  = GRID_ENTRY( grid, x, y, z, W  );
				const float t  = GRID_ENTRY( grid, x, y, z, T  );
				const float b  = GRID_ENTRY( grid, x, y, z, B  );
				const float ne = GRID_ENTRY( grid, x, y, z, NE );
				const float nw = GRID_ENTRY( grid, x, y, z, NW );
				const float se = GRID_ENTRY( grid, x, y, z, SE );
				const float sw = GRID_ENTRY( grid, x, y, z, SW );
				const float nt = GRID_ENTRY( grid, x, y, z, NT );
				const float nb = GRID_ENTRY( grid, x, y, z, NB );
				const float st = GRID_ENTRY( grid, x, y, z, ST );
				const float sb = GRID_ENTRY( grid, x, y, z, SB );
				const float et = GRID_ENTRY( grid, x, y, z, ET );
				const float eb = GRID_ENTRY( grid, x, y, z, EB );
				const float wt = GRID_ENTRY( grid, x, y, z, WT );
				const float wb = GRID_ENTRY( grid, x, y, z, WB );

				rho = c + n + s + e + w + t + b +
				      ne + nw + se + sw +
				      nt + nb + st + sb +
				      et + eb + wt + wb;

				ux = + e - w
				     + ne - nw
				     + se - sw
				     + et + eb
				     - wt - wb;
				uy = + n - s
				     + ne + nw
				     - se - sw
				     + nt + nb
				     - st - sb;
				uz = + t - b
				     + nt - nb
				     + st - sb
				     + et - eb
				     + wt - wb;

				{
					const float inv_rho = 1.0f / rho;
					ux *= inv_rho;
					uy *= inv_rho;
					uz *= inv_rho;
				}

				if( binary ) {
					storeValue( file, &ux );
					storeValue( file, &uy );
					storeValue( file, &uz );
				} else
					fprintf( file, "%e %e %e\n",
					         (double)ux, (double)uy, (double)uz );

			}
		}
	}

	fclose( file );
}

void LBM_compareVelocityField( LBM_Grid grid, const char* filename,
                             const int binary ) {
	int x, y, z;
	float rho, ux, uy, uz;
	OUTPUT_PRECISION fileUx, fileUy, fileUz,
	                 dUx, dUy, dUz,
	                 diff2, maxDiff2 = -1e+30f;

	FILE* file = fopen( filename, (binary ? "rb" : "r") );
	if (!file) {
		perror("LBM_compareVelocityField");
		return;
	}

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				const float c  = GRID_ENTRY( grid, x, y, z, C  );
				const float n  = GRID_ENTRY( grid, x, y, z, N  );
				const float s  = GRID_ENTRY( grid, x, y, z, S  );
				const float e  = GRID_ENTRY( grid, x, y, z, E  );
				const float w  = GRID_ENTRY( grid, x, y, z, W  );
				const float t  = GRID_ENTRY( grid, x, y, z, T  );
				const float b  = GRID_ENTRY( grid, x, y, z, B  );
				const float ne = GRID_ENTRY( grid, x, y, z, NE );
				const float nw = GRID_ENTRY( grid, x, y, z, NW );
				const float se = GRID_ENTRY( grid, x, y, z, SE );
				const float sw = GRID_ENTRY( grid, x, y, z, SW );
				const float nt = GRID_ENTRY( grid, x, y, z, NT );
				const float nb = GRID_ENTRY( grid, x, y, z, NB );
				const float st = GRID_ENTRY( grid, x, y, z, ST );
				const float sb = GRID_ENTRY( grid, x, y, z, SB );
				const float et = GRID_ENTRY( grid, x, y, z, ET );
				const float eb = GRID_ENTRY( grid, x, y, z, EB );
				const float wt = GRID_ENTRY( grid, x, y, z, WT );
				const float wb = GRID_ENTRY( grid, x, y, z, WB );

				rho = c + n + s + e + w + t + b +
				      ne + nw + se + sw +
				      nt + nb + st + sb +
				      et + eb + wt + wb;

				ux = + e - w
				     + ne - nw
				     + se - sw
				     + et + eb
				     - wt - wb;
				uy = + n - s
				     + ne + nw
				     - se - sw
				     + nt + nb
				     - st - sb;
				uz = + t - b
				     + nt - nb
				     + st - sb
				     + et - eb
				     + wt - wb;

				{
					const float inv_rho = 1.0f / rho;
					ux *= inv_rho;
					uy *= inv_rho;
					uz *= inv_rho;
				}

				if( binary ) {
					loadValue( file, &fileUx );
					loadValue( file, &fileUy );
					loadValue( file, &fileUz );
				}
				else {
					if( sizeof( OUTPUT_PRECISION ) == sizeof( double )) {
						fscanf( file, "%lf %lf %lf\n",
						        (double*)&fileUx, (double*)&fileUy, (double*)&fileUz );
					}
					else {
						fscanf( file, "%f %f %f\n",
						        (float*)&fileUx, (float*)&fileUy, (float*)&fileUz );
					}
				}

				dUx = ux - fileUx;
				dUy = uy - fileUy;
				dUz = uz - fileUz;
				diff2 = dUx*dUx + dUy*dUy + dUz*dUz;
				if( diff2 > maxDiff2 ) maxDiff2 = diff2;
			}
		}
	}

#if defined(SPEC_CPU)
	printf( "LBM_compareVelocityField: maxDiff = %e  \n\n",
	        sqrtf( maxDiff2 )  );
#else
	printf( "LBM_compareVelocityField: maxDiff = %e  ==>  %s\n\n",
	        sqrtf( maxDiff2 ),
	        sqrtf( maxDiff2 ) > 1e-5f ? "##### ERROR #####" : "OK" );
#endif
	fclose( file );
}

