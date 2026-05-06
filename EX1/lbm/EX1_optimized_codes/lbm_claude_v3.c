<<<CODE>>>
#include "lbm.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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

	*ptr = malloc( size );
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

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				if( fgetc( file ) != '.' ) SET_FLAG( grid, x, y, z, OBSTACLE );
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
	SWEEP_VAR

	float ux, uy, uz, u2, rho;
	const float omega_dfl1 = OMEGA * DFL1;
	const float omega_dfl2 = OMEGA * DFL2;
	const float omega_dfl3 = OMEGA * DFL3;
	const float one_minus_omega = 1.0f - OMEGA;

	/*voption indep*/
	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
			DST_C ( dstGrid ) = SRC_C ( srcGrid );
			DST_S ( dstGrid ) = SRC_N ( srcGrid );
			DST_N ( dstGrid ) = SRC_S ( srcGrid );
			DST_W ( dstGrid ) = SRC_E ( srcGrid );
			DST_E ( dstGrid ) = SRC_W ( srcGrid );
			DST_B ( dstGrid ) = SRC_T ( srcGrid );
			DST_T ( dstGrid ) = SRC_B ( srcGrid );
			DST_SW( dstGrid ) = SRC_NE( srcGrid );
			DST_SE( dstGrid ) = SRC_NW( srcGrid );
			DST_NW( dstGrid ) = SRC_SE( srcGrid );
			DST_NE( dstGrid ) = SRC_SW( srcGrid );
			DST_SB( dstGrid ) = SRC_NT( srcGrid );
			DST_ST( dstGrid ) = SRC_NB( srcGrid );
			DST_NB( dstGrid ) = SRC_ST( srcGrid );
			DST_NT( dstGrid ) = SRC_SB( srcGrid );
			DST_WB( dstGrid ) = SRC_ET( srcGrid );
			DST_WT( dstGrid ) = SRC_EB( srcGrid );
			DST_EB( dstGrid ) = SRC_WT( srcGrid );
			DST_ET( dstGrid ) = SRC_WB( srcGrid );
			continue;
		}

		float src_c = SRC_C ( srcGrid );
		float src_n = SRC_N ( srcGrid );
		float src_s = SRC_S ( srcGrid );
		float src_e = SRC_E ( srcGrid );
		float src_w = SRC_W ( srcGrid );
		float src_t = SRC_T ( srcGrid );
		float src_b = SRC_B ( srcGrid );
		float src_ne = SRC_NE( srcGrid );
		float src_nw = SRC_NW( srcGrid );
		float src_se = SRC_SE( srcGrid );
		float src_sw = SRC_SW( srcGrid );
		float src_nt = SRC_NT( srcGrid );
		float src_nb = SRC_NB( srcGrid );
		float src_st = SRC_ST( srcGrid );
		float src_sb = SRC_SB( srcGrid );
		float src_et = SRC_ET( srcGrid );
		float src_eb = SRC_EB( srcGrid );
		float src_wt = SRC_WT( srcGrid );
		float src_wb = SRC_WB( srcGrid );

		rho = src_c + src_n + src_s + src_e + src_w + src_t + src_b +
		      src_ne + src_nw + src_se + src_sw + src_nt + src_nb +
		      src_st + src_sb + src_et + src_eb + src_wt + src_wb;

		ux = src_e - src_w + src_ne - src_nw + src_se - src_sw +
		     src_et + src_eb - src_wt - src_wb;
		uy = src_n - src_s + src_ne + src_nw - src_se - src_sw +
		     src_nt + src_nb - src_st - src_sb;
		uz = src_t - src_b + src_nt - src_nb + src_st - src_sb +
		     src_et - src_eb + src_wt - src_wb;

		float inv_rho = 1.0f / rho;
		ux *= inv_rho;
		uy *= inv_rho;
		uz *= inv_rho;

		if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
			ux = 0.005f;
			uy = 0.002f;
			uz = 0.000f;
		}

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);
		
		float ux_45 = 4.5f * ux;
		float uy_45 = 4.5f * uy;
		float uz_45 = 4.5f * uz;
		
		float ux_3 = 3.0f * ux;
		float uy_3 = 3.0f * uy;
		float uz_3 = 3.0f * uz;

		DST_C ( dstGrid ) = one_minus_omega * src_c + omega_dfl1 * rho * (1.0f - u2);

		DST_N ( dstGrid ) = one_minus_omega * src_n + omega_dfl2 * rho * (1.0f + uy * (uy_45 + uy_3) - u2);
		DST_S ( dstGrid ) = one_minus_omega * src_s + omega_dfl2 * rho * (1.0f + uy * (uy_45 - uy_3) - u2);
		DST_E ( dstGrid ) = one_minus_omega * src_e + omega_dfl2 * rho * (1.0f + ux * (ux_45 + ux_3) - u2);
		DST_W ( dstGrid ) = one_minus_omega * src_w + omega_dfl2 * rho * (1.0f + ux * (ux_45 - ux_3) - u2);
		DST_T ( dstGrid ) = one_minus_omega * src_t + omega_dfl2 * rho * (1.0f + uz * (uz_45 + uz_3) - u2);
		DST_B ( dstGrid ) = one_minus_omega * src_b + omega_dfl2 * rho * (1.0f + uz * (uz_45 - uz_3) - u2);

		float ux_uy_p = ux + uy;
		float ux_uy_m = ux - uy;
		float ux_uz_p = ux + uz;
		float ux_uz_m = ux - uz;
		float uy_uz_p = uy + uz;
		float uy_uz_m = uy - uz;
		
		DST_NE( dstGrid ) = one_minus_omega * src_ne + omega_dfl3 * rho * (1.0f + ux_uy_p * (4.5f * ux_uy_p + 3.0f) - u2);
		DST_NW( dstGrid ) = one_minus_omega * src_nw + omega_dfl3 * rho * (1.0f + (-ux + uy) * (4.5f * (-ux + uy) + 3.0f) - u2);
		DST_SE( dstGrid ) = one_minus_omega * src_se + omega_dfl3 * rho * (1.0f + ux_uy_m * (4.5f * ux_uy_m + 3.0f) - u2);
		DST_SW( dstGrid ) = one_minus_omega * src_sw + omega_dfl3 * rho * (1.0f + (-ux - uy) * (4.5f * (-ux - uy) + 3.0f) - u2);
		DST_NT( dstGrid ) = one_minus_omega * src_nt + omega_dfl3 * rho * (1.0f + uy_uz_p * (4.5f * uy_uz_p + 3.0f) - u2);
		DST_NB( dstGrid ) = one_minus_omega * src_nb + omega_dfl3 * rho * (1.0f + uy_uz_m * (4.5f * uy_uz_m + 3.0f) - u2);
		DST_ST( dstGrid ) = one_minus_omega * src_st + omega_dfl3 * rho * (1.0f + (-uy + uz) * (4.5f * (-uy + uz) + 3.0f) - u2);
		DST_SB( dstGrid ) = one_minus_omega * src_sb + omega_dfl3 * rho * (1.0f + (-uy - uz) * (4.5f * (-uy - uz) + 3.0f) - u2);
		DST_ET( dstGrid ) = one_minus_omega * src_et + omega_dfl3 * rho * (1.0f + ux_uz_p * (4.5f * ux_uz_p + 3.0f) - u2);
		DST_EB( dstGrid ) = one_minus_omega * src_eb + omega_dfl3 * rho * (1.0f + ux_uz_m * (4.5f * ux_uz_m + 3.0f) - u2);
		DST_WT( dstGrid ) = one_minus_omega * src_wt + omega_dfl3 * rho * (1.0f + (-ux + uz) * (4.5f * (-ux + uz) + 3.0f) - u2);
		DST_WB( dstGrid ) = one_minus_omega * src_wb + omega

