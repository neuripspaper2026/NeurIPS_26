<<<CODE>>>
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
	const int total_cells = SIZE_X * SIZE_Y * (SIZE_Z + 4);
	
#ifdef _OPENMP
	#pragma omp parallel for schedule(static)
#endif
	for (int idx = 0; idx < total_cells; idx++) {
		int i = idx * N_CELL_ENTRIES;
		grid[i + C ] = DFL1;
		grid[i + N ] = DFL2;
		grid[i + S ] = DFL2;
		grid[i + E ] = DFL2;
		grid[i + W ] = DFL2;
		grid[i + T ] = DFL2;
		grid[i + B ] = DFL2;
		grid[i + NE] = DFL3;
		grid[i + NW] = DFL3;
		grid[i + SE] = DFL3;
		grid[i + SW] = DFL3;
		grid[i + NT] = DFL3;
		grid[i + NB] = DFL3;
		grid[i + ST] = DFL3;
		grid[i + SB] = DFL3;
		grid[i + ET] = DFL3;
		grid[i + EB] = DFL3;
		grid[i + WT] = DFL3;
		grid[i + WB] = DFL3;
		
		unsigned int* flag_ptr = (unsigned int*)(&grid[i + FLAGS]);
		*flag_ptr = 0;
	}
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

#ifdef _OPENMP
	#pragma omp parallel for private(x, y) schedule(static)
#endif
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

#ifdef _OPENMP
	#pragma omp parallel for private(x, y) schedule(static)
#endif
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
	const int total_cells = SIZE_X * SIZE_Y * SIZE_Z;
	
#ifdef _OPENMP
	#pragma omp parallel for schedule(static)
#endif
	for (int idx = 0; idx < total_cells; idx++) {
		int i = CALC_INDEX(0, 0, 0, 0) + idx * N_CELL_ENTRIES;
		
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

		float rho = SRC_C ( srcGrid ) + SRC_N ( srcGrid )
		      + SRC_S ( srcGrid ) + SRC_E ( srcGrid )
		      + SRC_W ( srcGrid ) + SRC_T ( srcGrid )
		      + SRC_B ( srcGrid ) + SRC_NE( srcGrid )
		      + SRC_NW( srcGrid ) + SRC_SE( srcGrid )
		      + SRC_SW( srcGrid ) + SRC_NT( srcGrid )
		      + SRC_NB( srcGrid ) + SRC_ST( srcGrid )
		      + SRC_SB( srcGrid ) + SRC_ET( srcGrid )
		      + SRC_EB( srcGrid ) + SRC_WT( srcGrid )
		      + SRC_WB( srcGrid );

		float ux = SRC_E ( srcGrid ) - SRC_W ( srcGrid )
		     + SRC_NE( srcGrid ) - SRC_NW( srcGrid )
		     + SRC_SE( srcGrid ) - SRC_SW( srcGrid )
		     + SRC_ET( srcGrid ) + SRC_EB( srcGrid )
		     - SRC_WT( srcGrid ) - SRC_WB( srcGrid );
		float uy = SRC_N ( srcGrid ) - SRC_S ( srcGrid )
		     + SRC_NE( srcGrid ) + SRC_NW( srcGrid )
		     - SRC_SE( srcGrid ) - SRC_SW( srcGrid )
		     + SRC_NT( srcGrid ) + SRC_NB( srcGrid )
		     - SRC_ST( srcGrid ) - SRC_SB( srcGrid );
		float uz = SRC_T ( srcGrid ) - SRC_B ( srcGrid )
		     + SRC_NT( srcGrid ) - SRC_NB( srcGrid )
		     + SRC_ST( srcGrid ) - SRC_SB( srcGrid )
		     + SRC_ET( srcGrid ) - SRC_EB( srcGrid )
		     + SRC_WT( srcGrid ) - SRC_WB( srcGrid );

		float inv_rho = 1.0f / rho;
		ux *= inv_rho;
		uy *= inv_rho;
		uz *= inv_rho;

		if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
			ux = 0.005f;
			uy = 0.002f;
			uz = 0.000f;
		}

		float u2 = 1.5f * (ux*ux + uy*uy + uz*uz);
		float omega_rho = OMEGA * rho;
		float one_minus_omega = 1.0f - OMEGA;
		
		DST_C ( dstGrid ) = one_minus_omega*SRC_C ( srcGrid ) + DFL1*omega_rho*(1.0f - u2);

		float uy_term = 4.5f*uy;
		DST_N ( dstGrid ) = one_minus_omega*SRC_N ( srcGrid ) + DFL2*omega_rho*(1.0f + uy*(uy_term + 3.0f) - u2);
		DST_S ( dstGrid ) = one_minus_omega*SRC_S ( srcGrid ) + DFL2*omega_rho*(1.0f + uy*(uy_term - 3.0f) - u2);
		
		float ux_term = 4.5f*ux;
		DST_E ( dstGrid ) = one_minus_omega*SRC_E ( srcGrid ) + DFL2*omega_rho*(1.0f + ux*(ux_term + 3.0f) - u2);
		DST_W ( dstGrid ) = one_minus_omega*SRC_W ( srcGrid ) + DFL2*omega_rho*(1.0f + ux*(ux_term - 3.0f) - u2);
		
		float uz_term = 4.5f*uz;
		DST_T ( dstGrid ) = one_minus_omega*SRC_T ( srcGrid ) + DFL2*omega_rho*(1.0f + uz*(uz_term + 3.0f) - u2);
		DST_B ( dstGrid ) = one_minus_omega*SRC_B ( srcGrid ) + DFL2*omega_rho*(1.0f + uz*(uz_term - 3.0f) - u2);

		float ux_plus_uy = ux + uy;
		DST_NE( dstGrid ) = one_minus_omega*SRC_NE( srcGrid ) + DFL3*omega_rho*(1.0f + ux_plus_uy*(4.5f*ux_plus_uy + 3.0f) - u2);
		float neg_ux_plus_uy = -ux + uy;
		DST_NW( dstGrid ) = one_minus_omega*SRC_NW( srcGrid ) + DFL3*omega_rho*(1.0f + neg_ux_plus_uy*(4.5f*neg_ux_plus_uy + 3.0f) - u2);
		float ux_minus_uy = ux - uy;
		DST_SE( dstGrid ) = one_minus_omega*SRC_SE( srcGrid ) + DFL3*omega_rho*(1.0f + ux_minus_uy*(4.5f*ux_minus_uy + 3.0f) - u2);
		float neg_ux_minus_uy = -ux - uy;
		DST_SW( dstGrid ) = one_minus_omega*SRC_SW( srcGrid ) + DFL3*omega_rho*(1.0f + neg_ux_minus_uy*(4.5f*neg_ux_minus_uy + 3.0f) - u2);
		
		float uy_plus_uz = uy + uz;
		DST_NT( dstGrid ) = one_minus_omega*SRC_NT( srcGrid ) + DFL3*omega_rho*(1.0f + uy_plus_uz*(4.5f*uy_plus_uz + 3.0f) - u2);
		float uy_minus_uz = uy - uz;
		DST_NB( dstGrid ) = one_minus_omega*SRC_NB( srcGrid ) + DFL3*omega_rho*(1.0f + uy_minus_uz*(4.5f*uy_minus_uz + 3.0f) - u2);
		float neg_uy_plus_uz = -uy + uz;
		DST_ST( dstGrid ) = one_minus_omega*SRC_ST( srcGrid ) + DFL3*omega_rho*(1.0f + neg_uy_plus_uz*(4.5f*neg_uy_plus_uz + 3.0f) - u2);
		float neg_uy_minus_uz = -uy - uz;
		DST_SB( dstGrid ) = one_minus_omega*SRC_SB( srcGrid ) + DFL3*omega_rho*(1.0f + neg_uy_minus_uz*(4.5f*neg_uy_minus_uz + 3.0f) - u2);
		
		float ux_plus_uz = ux + uz;
		DST_ET( dstGrid ) = one_minus_omega*SRC_ET( srcGrid ) + DFL3*omega_rho*(1.0f + ux_plus_uz*(4.5f*ux_plus_uz + 3.0f) - u2);
		float ux_minus_uz = ux -

