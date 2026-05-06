#include "lbm.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#if !defined(SPEC_CPU)
#ifdef _OPENMP
#include <omp.h>
#endif
#endif

#define DFL1 (1.0/ 3.0)
#define DFL2 (1.0/18.0)
#define DFL3 (1.0/36.0)

void LBM_allocateGrid( float** ptr ) {
	const size_t margin = 2*SIZE_X*SIZE_Y*N_CELL_ENTRIES,
	             size   = sizeof( LBM_Grid ) + 2*margin*sizeof( float );

	*ptr = (float*)malloc( size );
	if( ! *ptr ) {
		printf( "LBM_allocateGrid: could not allocate %.1f MByte\n",
		        size / (1024.0*1024.0) );
		exit( 1 );
	}
#if !defined(SPEC_CPU)
	printf( "LBM_allocateGrid: allocated %.1f MByte\n",
	        size / (1024.0*1024.0) );
#endif
	*ptr += margin;
}

void LBM_freeGrid( float** ptr ) {
	const size_t margin = 2*SIZE_X*SIZE_Y*N_CELL_ENTRIES;

	free( *ptr-margin );
	*ptr = NULL;
}

void LBM_initializeGrid( LBM_Grid grid ) {
	SWEEP_VAR

	/*voption indep*/
	SWEEP_START( 0, 0, -2, 0, 0, SIZE_Z+2 )
		LOCAL( grid, C  ) = (float)DFL1;
		LOCAL( grid, N  ) = (float)DFL2;
		LOCAL( grid, S  ) = (float)DFL2;
		LOCAL( grid, E  ) = (float)DFL2;
		LOCAL( grid, W  ) = (float)DFL2;
		LOCAL( grid, T  ) = (float)DFL2;
		LOCAL( grid, B  ) = (float)DFL2;
		LOCAL( grid, NE ) = (float)DFL3;
		LOCAL( grid, NW ) = (float)DFL3;
		LOCAL( grid, SE ) = (float)DFL3;
		LOCAL( grid, SW ) = (float)DFL3;
		LOCAL( grid, NT ) = (float)DFL3;
		LOCAL( grid, NB ) = (float)DFL3;
		LOCAL( grid, ST ) = (float)DFL3;
		LOCAL( grid, SB ) = (float)DFL3;
		LOCAL( grid, ET ) = (float)DFL3;
		LOCAL( grid, EB ) = (float)DFL3;
		LOCAL( grid, WT ) = (float)DFL3;
		LOCAL( grid, WB ) = (float)DFL3;

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
		return;
	}

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				int c = fgetc( file );
				if( c != '.' && c != EOF ) SET_FLAG( grid, x, y, z, OBSTACLE );
			}
			fgetc( file );
		}
		fgetc( file );
	}

	fclose( file );
}

void LBM_initializeSpecialCellsForLDC( LBM_Grid grid ) {
	int x,  y,  z;

	/*voption indep*/
	for( z = -2; z < SIZE_Z+2; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				if( x == 0 || x == SIZE_X-1 ||
				    y == 0 || y == SIZE_Y-1 ||
				    z == 0 || z == SIZE_Z-1 ) {
					SET_FLAG( grid, x, y, z, OBSTACLE );
				}
				else {
					if( (z == 1 || z == SIZE_Z-2) &&
					     x > 1 && x < SIZE_X-2 &&
					     y > 1 && y < SIZE_Y-2 ) {
						SET_FLAG( grid, x, y, z, ACCEL );
					}
				}
			}
		}
	}
}

void LBM_initializeSpecialCellsForChannel( LBM_Grid grid ) {
	int x,  y,  z;

	/*voption indep*/
	for( z = -2; z < SIZE_Z+2; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				if( x == 0 || x == SIZE_X-1 ||
				    y == 0 || y == SIZE_Y-1 ) {
					SET_FLAG( grid, x, y, z, OBSTACLE );

					if( (z == 0 || z == SIZE_Z-1) &&
					    ! TEST_FLAG( grid, x, y, z, OBSTACLE ))
						SET_FLAG( grid, x, y, z, IN_OUT_FLOW );
				}
			}
		}
	}
}

void LBM_performStreamCollide( LBM_Grid srcGrid, LBM_Grid dstGrid ) {
	const float omega = OMEGA;
	const float one_minus_omega = 1.0f - OMEGA;
	const float omega_dfl1 = (float)DFL1 * OMEGA;
	const float omega_dfl2 = (float)DFL2 * OMEGA;
	const float omega_dfl3 = (float)DFL3 * OMEGA;

	SWEEP_VAR

	/*voption indep*/
#if defined(_OPENMP) && !defined(SPEC_CPU)
#pragma omp parallel for private(i) schedule(static)
#endif
	for( i = CALC_INDEX(0, 0, 0, 0); i < CALC_INDEX(0, 0, SIZE_Z, 0); i += N_CELL_ENTRIES ) {
		float ux, uy, uz, u2, rho;

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

			rho =  src_c  + src_n  + src_s  + src_e  + src_w  + src_t  + src_b
			     + src_ne + src_nw + src_se + src_sw + src_nt + src_nb
			     + src_st + src_sb + src_et + src_eb + src_wt + src_wb;

			ux =  src_e  - src_w
			   +  src_ne - src_nw
			   +  src_se - src_sw
			   +  src_et + src_eb
			   -  src_wt - src_wb;

			uy =  src_n  - src_s
			   +  src_ne + src_nw
			   -  src_se - src_sw
			   +  src_nt + src_nb
			   -  src_st - src_sb;

			uz =  src_t  - src_b
			   +  src_nt - src_nb
			   +  src_st - src_sb
			   +  src_et - src_eb
			   +  src_wt - src_wb;

			ux /= rho;
			uy /= rho;
			uz /= rho;

			if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
				ux = 0.005f;
				uy = 0.002f;
				uz = 0.000f;
			}

			u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

			const float cu  =       ux * 3.0f;
			const float cv  =       uy * 3.0f;
			const float cw  =       uz * 3.0f;
			const float cux = 4.5f * ux;
			const float cuy = 4.5f * uy;
			const float cuz = 4.5f * uz;

			DST_C ( dstGrid ) = one_minus_omega*src_c  + omega_dfl1*rho*(1.0f         - u2);

			DST_N ( dstGrid ) = one_minus_omega*src_n  + omega_dfl2*rho*(1.0f +       uy*(cuy + cv ) - u2);
			DST_S ( dstGrid ) = one_minus_omega*src_s  + omega_dfl2*rho*(1.0f +       uy*(cuy - cv ) - u2);
			DST_E ( dstGrid ) = one_minus_omega*src_e  + omega_dfl2*rho*(1.0f +       ux*(cux + cu ) - u2);
			DST_W ( dstGrid ) = one_minus_omega*src_w  + omega_dfl2*rho*(1.0f +       ux*(cux - cu ) - u2);
			DST_T ( dstGrid ) = one_minus_omega*src_t  + omega_dfl2*rho*(1.0f +       uz*(cuz + cw ) - u2);
			DST_B ( dstGrid ) = one_minus_omega*src_b  + omega_dfl2*rho*(1.0f +       uz*(cuz - cw ) - u2);

			const float uxy_p = ux + uy;
			const float uxy_m = ux - uy;
			const float uyz_p = uy + uz;
			const float uyz_m = uy - uz;
			const float uxz_p = ux + uz;
			const float uxz_m = ux - uz;

			DST_NE( dstGrid ) = one_minus_omega*src_ne + omega_dfl3*rho*(1.0f + ( uxy_p)*(4.5f*( uxy_p) + 3.0f) - u2);
			DST_NW( dstGrid ) = one_minus_omega*src_nw + omega_dfl3*rho*(1.0f + (-uxy_m)*(4.5f*(-uxy_m) + 3.0f) - u2);
			DST_SE( dstGrid ) = one_minus_omega*src_se + omega_dfl3*rho*(1.0f + ( uxy_m)*(4.5f*( uxy_m) + 3.0f) - u2);
			DST_SW( dstGrid ) = one_minus_omega*src_sw + omega_dfl3*rho*(1.0f + (-uxy_p)*(4.5f*(-uxy_p) + 3.0f) - u2);
			DST_NT( dstGrid ) = one_minus_omega*src_nt + omega_dfl3*rho*(1.0f + ( uyz_p)*(4.5f*( uyz_p) + 3.0f) - u2);
			DST_NB( dstGrid ) = one_minus_omega*src_nb + omega_dfl3*rho*(1.0f + ( uyz_m)*(4.5f*( uyz_m) + 3.0f) - u2);
			DST_ST( dstGrid ) = one_minus_omega*src_st + omega_dfl3*rho*(1.0f + (-uyz_m)*(4.5f*(-uyz_m) + 3.0f) - u2);
			DST_SB( dstGrid ) = one_minus_omega*src_sb + omega_dfl3*rho*(1.0f + (-uyz_p)*(4.5f*(-uyz_p) + 3.0f) - u2);
			DST_ET( dstGrid ) = one_minus_omega*src_et + omega_dfl3*rho*(1.0f + ( uxz_p)*(4.5f*( uxz_p) + 3.0f) - u2);
			DST_EB( dstGrid ) = one_minus_omega*src_eb + omega_dfl3*rho*(1.0f + ( uxz_m)*(4.5f*( uxz_m) + 3.0f) - u2);
			DST_WT( dstGrid ) = one_minus_omega*src_wt + omega_dfl3*rho*(1.0f + (-uxz_m)*(4.5f*(-uxz_m) + 3.0f) - u2);
			DST_WB( dstGrid ) = one_minus_omega*src_wb + omega_dfl3*rho*(1.0f + (-uxz_p)*(4.5f*(-uxz_p) + 3.0f) - u2);
		}
	}
}

void LBM_handleInOutFlow( LBM_Grid srcGrid ) {
	float ux , uy , uz , rho ,
	       ux1, uy1, uz1, rho1,
	       ux2, uy2, uz2, rho2,
	       u2, px, py;
	SWEEP_VAR

	/* inflow */
	/*voption indep*/
#if defined(_OPENMP) && !defined(SPEC_CPU)
#pragma omp parallel for private(i,ux,uy,uz,rho,ux1,uy1,uz1,rho1,ux2,uy2,uz2,rho2,u2,px,py) schedule(static)
#endif
	for( i = CALC_INDEX(0, 0, 0, 0); i < CALC_INDEX(0, 0, 1, 0); i += N_CELL_ENTRIES ) {
		const int sx = SWEEP_X;
		const int sy = SWEEP_Y;

		rho1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, N  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, E  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, T  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ST )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ET )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WB );
		rho2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, N  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, E  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, T  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ST )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ET )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WB );

		rho = 2.0f*rho1 - rho2;

		px = (sx / (0.5f*(SIZE_X-1))) - 1.0f;
		py = (sy / (0.5f*(SIZE_Y-1))) - 1.0f;
		ux = 0.00f;
		uy = 0.00f;
		uz = 0.01f * (1.0f-px*px) * (1.0f-py*py);

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = (float)DFL1*rho*(1.0f                                 - u2);

		LOCAL( srcGrid, N ) = (float)DFL2*rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2);
		LOCAL( srcGrid, S ) = (float)DFL2*rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2);
		LOCAL( srcGrid, E ) = (float)DFL2*rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2);
		LOCAL( srcGrid, W ) = (float)DFL2*rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2);
		LOCAL( srcGrid, T ) = (float)DFL2*rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2);
		LOCAL( srcGrid, B ) = (float)DFL2*rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2);

		LOCAL( srcGrid, NE) = (float)DFL3*rho*(1.0f + (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
		LOCAL( srcGrid, NW) = (float)DFL3*rho*(1.0f + (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
		LOCAL( srcGrid, SE) = (float)DFL3*rho*(1.0f + (+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
		LOCAL( srcGrid, SW) = (float)DFL3*rho*(1.0f + (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
		LOCAL( srcGrid, NT) = (float)DFL3*rho*(1.0f + (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
		LOCAL( srcGrid, NB) = (float)DFL3*rho*(1.0f + (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
		LOCAL( srcGrid, ST) = (float)DFL3*rho*(1.0f + (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
		LOCAL( srcGrid, SB) = (float)DFL3*rho*(1.0f + (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
		LOCAL( srcGrid, ET) = (float)DFL3*rho*(1.0f + (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
		LOCAL( srcGrid, EB) = (float)DFL3*rho*(1.0f + (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
		LOCAL( srcGrid, WT) = (float)DFL3*rho*(1.0f + (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
		LOCAL( srcGrid, WB) = (float)DFL3*rho*(1.0f + (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
	}

	/* outflow */
	/*voption indep*/
#if defined(_OPENMP) && !defined(SPEC_CPU)
#pragma omp parallel for private(i,ux,uy,uz,rho,ux1,uy1,uz1,rho1,ux2,uy2,uz2,rho2,u2) schedule(static)
#endif
	for( i = CALC_INDEX(0, 0, SIZE_Z-1, 0); i < CALC_INDEX(0, 0, SIZE_Z, 0); i += N_CELL_ENTRIES ) {
		rho1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, N  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, E  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, T  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );
		ux1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, E  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, W  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );
		uy1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, N  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, S  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB );
		uz1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, T  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, B  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );

		ux1 /= rho1;
		uy1 /= rho1;
		uz1 /= rho1;

		rho2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, N  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, E  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, T  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );
		ux2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, E  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, W  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );
		uy2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, N  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, S  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB );
		uz2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, T  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, B  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );

		ux2 /= rho2;
		uy2 /= rho2;
		uz2 /= rho2;

		rho = 1.0f;

		ux = 2.0f*ux1 - ux2;
		uy = 2.0f*uy1 - uy2;
		uz = 2.0f*uz1 - uz2;

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = (float)DFL1*rho*(1.0f                                 - u2);

		LOCAL( srcGrid, N ) = (float)DFL2*rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2);
		LOCAL( srcGrid, S ) = (float)DFL2*rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2);
		LOCAL( srcGrid, E ) = (float)DFL2*rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2);
		LOCAL( srcGrid, W ) = (float)DFL2*rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2);
		LOCAL( srcGrid, T ) = (float)DFL2*rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2);
		LOCAL( srcGrid, B ) = (float)DFL2*rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2);

		LOCAL( srcGrid, NE) = (float)DFL3*rho*(1.0f + (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
		LOCAL( srcGrid, NW) = (float)DFL3*rho*(1.0f + (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
		LOCAL( srcGrid, SE) = (float)DFL3*rho*(1.0f + (+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
		LOCAL( srcGrid, SW) = (float)DFL3*rho*(1.0f + (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
		LOCAL( srcGrid, NT) = (float)DFL3*rho*(1.0f + (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
		LOCAL( srcGrid, NB) = (float)DFL3*rho*(1.0f + (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
		LOCAL( srcGrid, ST) = (float)DFL3*rho*(1.0f + (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
		LOCAL( srcGrid, SB) = (float)DFL3*rho*(1.0f + (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
		LOCAL( srcGrid, ET) = (float)DFL3*rho*(1.0f + (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
		LOCAL( srcGrid, EB) = (float)DFL3*rho*(1.0f + (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
		LOCAL( srcGrid, WT) = (float)DFL3*rho*(1.0f + (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
		LOCAL( srcGrid, WB) = (float)DFL3*rho*(1.0f + (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
	}
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

	{
		int nObstacleCellsLocal = 0;
		int nAccelCellsLocal = 0;
		int nFluidCellsLocal = 0;
		float minU2Local  = 1e+30f, maxU2Local  = -1e+30f;
		float minRhoLocal = 1e+30f, maxRhoLocal = -1e+30f;
		float massLocal = 0.0f;

#if defined(_OPENMP) && !defined(SPEC_CPU)
#pragma omp parallel private(i,ux,uy,uz,u2,rho) reduction(+:massLocal,nObstacleCellsLocal,nAccelCellsLocal,nFluidCellsLocal) \
                    reduction(min:minU2Local,minRhoLocal) reduction(max:maxU2Local,maxRhoLocal)
#endif
		{
#if defined(_OPENMP) && !defined(SPEC_CPU)
#pragma omp for schedule(static)
#endif
			for( i = CALC_INDEX(0, 0, 0, 0); i < CALC_INDEX(0, 0, SIZE_Z, 0); i += N_CELL_ENTRIES ) {
				rho = + LOCAL( grid, C  ) + LOCAL( grid, N  )
				      + LOCAL( grid, S  ) + LOCAL( grid, E  )
				      + LOCAL( grid, W  ) + LOCAL( grid, T  )
				      + LOCAL( grid, B  ) + LOCAL( grid, NE )
				      + LOCAL( grid, NW ) + LOCAL( grid, SE )
				      + LOCAL( grid, SW ) + LOCAL( grid, NT )
				      + LOCAL( grid, NB ) + LOCAL( grid, ST )
				      + LOCAL( grid, SB ) + LOCAL( grid, ET )
				      + LOCAL( grid, EB ) + LOCAL( grid, WT )
				      + LOCAL( grid, WB );
				if( rho < minRhoLocal ) minRhoLocal = rho;
				if( rho > maxRhoLocal ) maxRhoLocal = rho;
				massLocal += rho;

				if( TEST_FLAG_SWEEP( grid, OBSTACLE )) {
					nObstacleCellsLocal++;
				}
				else {
					if( TEST_FLAG_SWEEP( grid, ACCEL ))
						nAccelCellsLocal++;
					else
						nFluidCellsLocal++;

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
					u2 = (ux*ux + uy*uy + uz*uz) / (rho*rho);
					if( u2 < minU2Local ) minU2Local = u2;
					if( u2 > maxU2Local ) maxU2Local = u2;
				}
			}
		}

		nObstacleCells = nObstacleCellsLocal;
		nAccelCells    = nAccelCellsLocal;
		nFluidCells    = nFluidCellsLocal;
		minU2          = minU2Local;
		maxU2          = maxU2Local;
		minRho         = minRhoLocal;
		maxRho         = maxRhoLocal;
		mass           = massLocal;
	}

        printf( "LBM_showGridStatistics:\n"
        "\tnObstacleCells: %7i nAccelCells: %7i nFluidCells: %7i\n"
        "\tminRho: %8.4f maxRho: %8.4f mass: %e\n"
        "\tminU: %e maxU: %e\n\n",
        nObstacleCells, nAccelCells, nFluidCells,
        minRho, maxRho, mass,
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
	if (!file) return;

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				rho = + GRID_ENTRY( grid, x, y, z, C  ) + GRID_ENTRY( grid, x, y, z, N  )
				      + GRID_ENTRY( grid, x, y, z, S  ) + GRID_ENTRY( grid, x, y, z, E  )
				      + GRID_ENTRY( grid, x, y, z, W  ) + GRID_ENTRY( grid, x, y, z, T  )
				      + GRID_ENTRY( grid, x, y, z, B  ) + GRID_ENTRY( grid, x, y, z, NE )
				      + GRID_ENTRY( grid, x, y, z, NW ) + GRID_ENTRY( grid, x, y, z, SE )
				      + GRID_ENTRY( grid, x, y, z, SW ) + GRID_ENTRY( grid, x, y, z, NT )
				      + GRID_ENTRY( grid, x, y, z, NB ) + GRID_ENTRY( grid, x, y, z, ST )
				      + GRID_ENTRY( grid, x, y, z, SB ) + GRID_ENTRY( grid, x, y, z, ET )
				      + GRID_ENTRY( grid, x, y, z, EB ) + GRID_ENTRY( grid, x, y, z, WT )
				      + GRID_ENTRY( grid, x, y, z, WB );
				ux = + GRID_ENTRY( grid, x, y, z, E  ) - GRID_ENTRY( grid, x, y, z, W  ) 
				     + GRID_ENTRY( grid, x, y, z, NE ) - GRID_ENTRY( grid, x, y, z, NW ) 
				     + GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW ) 
				     + GRID_ENTRY( grid, x, y, z, ET ) + GRID_ENTRY( grid, x, y, z, EB ) 
				     - GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
				uy = + GRID_ENTRY( grid, x, y, z, N  ) - GRID_ENTRY( grid, x, y, z, S  ) 
				     + GRID_ENTRY( grid, x, y, z, NE ) + GRID_ENTRY( grid, x, y, z, NW ) 
				     - GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW ) 
				     + GRID_ENTRY( grid, x, y, z, NT ) + GRID_ENTRY( grid, x, y, z, NB ) 
				     - GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB );
				uz = + GRID_ENTRY( grid, x, y, z, T  ) - GRID_ENTRY( grid, x, y, z, B  ) 
				     + GRID_ENTRY( grid, x, y, z, NT ) - GRID_ENTRY( grid, x, y, z, NB ) 
				     + GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB ) 
				     + GRID_ENTRY( grid, x, y, z, ET ) - GRID_ENTRY( grid, x, y, z, EB ) 
				     + GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
				ux /= rho;
				uy /= rho;
				uz /= rho;

				if( binary ) {
					storeValue( file, &ux );
					storeValue( file, &uy );
					storeValue( file, &uz );
				} else
					fprintf( file, "%e %e %e\n", (double)ux, (double)uy, (double)uz );

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
	if (!file) return;

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				rho = + GRID_ENTRY( grid, x, y, z, C  ) + GRID_ENTRY( grid, x, y, z, N  )
				      + GRID_ENTRY( grid, x, y, z, S  ) + GRID_ENTRY( grid, x, y, z, E  )
				      + GRID_ENTRY( grid, x, y, z, W  ) + GRID_ENTRY( grid, x, y, z, T  )
				      + GRID_ENTRY( grid, x, y, z, B  ) + GRID_ENTRY( grid, x, y, z, NE )
				      + GRID_ENTRY( grid, x, y, z, NW ) + GRID_ENTRY( grid, x, y, z, SE )
				      + GRID_ENTRY( grid, x, y, z, SW ) + GRID_ENTRY( grid, x, y, z, NT )
				      + GRID_ENTRY( grid, x, y, z, NB ) + GRID_ENTRY( grid, x, y, z, ST )
				      + GRID_ENTRY( grid, x, y, z, SB ) + GRID_ENTRY( grid, x, y, z, ET )
				      + GRID_ENTRY( grid, x, y, z, EB ) + GRID_ENTRY( grid, x, y, z, WT )
				      + GRID_ENTRY( grid, x, y, z, WB );
				ux = + GRID_ENTRY( grid, x, y, z, E  ) - GRID_ENTRY( grid, x, y, z, W  ) 
				     + GRID_ENTRY( grid, x, y, z, NE ) - GRID_ENTRY( grid, x, y, z, NW ) 
				     + GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW ) 
				     + GRID_ENTRY( grid, x, y, z, ET ) + GRID_ENTRY( grid, x, y, z, EB ) 
				     - GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
				uy = + GRID_ENTRY( grid, x, y, z, N  ) - GRID_ENTRY( grid, x, y, z, S  ) 
				     + GRID_ENTRY( grid, x, y, z, NE ) + GRID_ENTRY( grid, x, y, z, NW ) 
				     - GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW ) 
				     + GRID_ENTRY( grid, x, y, z, NT ) + GRID_ENTRY( grid, x, y, z, NB ) 
				     - GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB );
				uz = + GRID_ENTRY( grid, x, y, z, T  ) - GRID_ENTRY( grid, x, y, z, B  ) 
				     + GRID_ENTRY( grid, x, y, z, NT ) - GRID_ENTRY( grid, x, y, z, NB ) 
				     + GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB ) 
				     + GRID_ENTRY( grid, x, y, z, ET ) - GRID_ENTRY( grid, x, y, z, EB ) 
				     + GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
				ux /= rho;
				uy /= rho;
				uz /= rho;

				if( binary ) {
					loadValue( file, &fileUx );
					loadValue( file, &fileUy );
					loadValue( file, &fileUz );
				}
				else {
					if( sizeof( OUTPUT_PRECISION ) == sizeof( double )) {
						fscanf( file, "%lf %lf %lf\n", &fileUx, &fileUy, &fileUz );
					}
					else {
						fscanf( file, "%f %f %f\n", &fileUx, &fileUy, &fileUz );
					}
				}

				dUx = (OUTPUT_PRECISION)ux - fileUx;
				dUy = (OUTPUT_PRECISION)uy - fileUy;
				dUz = (OUTPUT_PRECISION)uz - fileUz;
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

