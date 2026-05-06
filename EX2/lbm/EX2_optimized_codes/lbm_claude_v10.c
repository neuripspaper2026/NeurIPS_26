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
	for(int idx = 0; idx < total_cells; idx++) {
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
	const float omega = OMEGA;
	const float one_minus_omega = 1.0f - omega;
	const float omega_dfl1 = DFL1 * omega;
	const float omega_dfl2 = DFL2 * omega;
	const float omega_dfl3 = DFL3 * omega;
	
	const int total_cells = SIZE_X * SIZE_Y * SIZE_Z;
	
#ifdef _OPENMP
	#pragma omp parallel for schedule(static)
#endif
	for(int cell_idx = 0; cell_idx < total_cells; cell_idx++) {
		int z = cell_idx / (SIZE_X * SIZE_Y);
		int rem = cell_idx % (SIZE_X * SIZE_Y);
		int y = rem / SIZE_X;
		int x = rem % SIZE_X;
		
		int i = CALC_INDEX(x, y, z, 0);
		
		unsigned int flags = *((unsigned int*)(&srcGrid[i + FLAGS]));
		
		if(flags & OBSTACLE) {
			dstGrid[i + C ] = srcGrid[i + C ];
			dstGrid[CALC_INDEX(x, y-1, z, N)] = srcGrid[i + N];
			dstGrid[CALC_INDEX(x, y+1, z, S)] = srcGrid[i + S];
			dstGrid[CALC_INDEX(x-1, y, z, E)] = srcGrid[i + E];
			dstGrid[CALC_INDEX(x+1, y, z, W)] = srcGrid[i + W];
			dstGrid[CALC_INDEX(x, y, z-1, T)] = srcGrid[i + T];
			dstGrid[CALC_INDEX(x, y, z+1, B)] = srcGrid[i + B];
			dstGrid[CALC_INDEX(x-1, y-1, z, NE)] = srcGrid[i + NE];
			dstGrid[CALC_INDEX(x+1, y-1, z, NW)] = srcGrid[i + NW];
			dstGrid[CALC_INDEX(x-1, y+1, z, SE)] = srcGrid[i + SE];
			dstGrid[CALC_INDEX(x+1, y+1, z, SW)] = srcGrid[i + SW];
			dstGrid[CALC_INDEX(x, y-1, z-1, NT)] = srcGrid[i + NT];
			dstGrid[CALC_INDEX(x, y-1, z+1, NB)] = srcGrid[i + NB];
			dstGrid[CALC_INDEX(x, y+1, z-1, ST)] = srcGrid[i + ST];
			dstGrid[CALC_INDEX(x, y+1, z+1, SB)] = srcGrid[i + SB];
			dstGrid[CALC_INDEX(x-1, y, z-1, ET)] = srcGrid[i + ET];
			dstGrid[CALC_INDEX(x-1, y, z+1, EB)] = srcGrid[i + EB];
			dstGrid[CALC_INDEX(x+1, y, z-1, WT)] = srcGrid[i + WT];
			dstGrid[CALC_INDEX(x+1, y, z+1, WB)] = srcGrid[i + WB];
			continue;
		}

		float src_C  = srcGrid[i + C ];
		float src_N  = srcGrid[i + N ];
		float src_S  = srcGrid[i + S ];
		float src_E  = srcGrid[i + E ];
		float src_W  = srcGrid[i + W ];
		float src_T  = srcGrid[i + T ];
		float src_B  = srcGrid[i + B ];
		float src_NE = srcGrid[i + NE];
		float src_NW = srcGrid[i + NW];
		float src_SE = srcGrid[i + SE];
		float src_SW = srcGrid[i + SW];
		float src_NT = srcGrid[i + NT];
		float src_NB = srcGrid[i + NB];
		float src_ST = srcGrid[i + ST];
		float src_SB = srcGrid[i + SB];
		float src_ET = srcGrid[i + ET];
		float src_EB = srcGrid[i + EB];
		float src_WT = srcGrid[i + WT];
		float src_WB = srcGrid[i + WB];

		float rho = src_C  + src_N  + src_S  + src_E  + src_W  + src_T  + src_B  +
		            src_NE + src_NW + src_SE + src_SW + src_NT + src_NB + src_ST +
		            src_SB + src_ET + src_EB + src_WT + src_WB;

		float ux = (src_E  - src_W ) + (src_NE - src_NW) + (src_SE - src_SW) +
		           (src_ET + src_EB) - (src_WT + src_WB);
		float uy = (src_N  - src_S ) + (src_NE + src_NW) - (src_SE + src_SW) +
		           (src_NT + src_NB) - (src_ST + src_SB);
		float uz = (src_T  - src_B ) + (src_NT - src_NB) + (src_ST - src_SB) +
		           (src_ET - src_EB) + (src_WT - src_WB);

		float inv_rho = 1.0f / rho;
		ux *= inv_rho;
		uy *= inv_rho;
		uz *= inv_rho;

		if(flags & ACCEL) {
			ux = 0.005f;
			uy = 0.002f;
			uz = 0.000f;
		}

		float u2 = 1.5f * (ux*ux + uy*uy + uz*uz);
		
		float ux_45 = 4.5f * ux;
		float uy_45 = 4.5f * uy;
		float uz_45 = 4.5f * uz;
		
		float ux_3 = 3.0f * ux;
		float uy_3 = 3.0f * uy;
		float uz_3 = 3.0f * uz;

		dstGrid[i + C] = one_minus_omega * src_C + omega_dfl1 * rho * (1.0f - u2);

		dstGrid[CALC_INDEX(x, y+1, z, N)] = one_minus_omega * src_N + omega_dfl2 * rho * (1.0f + uy * (uy_45 + uy_3) - u2);
		dstGrid[CALC_INDEX(x, y-1, z, S)] = one_minus_omega * src_S + omega_dfl2 * rho * (1.0f + uy * (uy_45 - uy_3) - u2);
		dstGrid[CALC_INDEX(x+1, y, z, E)] = one_minus_omega * src_E + omega_dfl2 * rho * (1.0f + ux * (ux_45 + ux_3) - u2);
		dstGrid[CALC_INDEX(x-1, y, z, W)] = one_minus_omega * src_W + omega_dfl2 * rho * (1.0f + ux * (ux_45 - ux_3) - u2);
		dstGrid[CALC_INDEX(x, y, z+1, T)] = one_minus_omega * src_T + omega_dfl2 * rho * (1.0f + uz * (uz_45 + uz_3) - u2);
		dstGrid[CALC_INDEX(x, y, z-1, B)] = one_minus_omega * src_B + omega_dfl2 * rho * (1.0f + uz * (uz_45 - uz_3) - u2);

		float ux_uy_p = ux + uy;
		float ux_uy_m = ux - uy;
		float ux_uz_p = ux + uz;
		float ux_uz_m = ux - uz;
		float uy_uz_p = uy + uz;
		float uy_uz_m = uy - uz;
		
		dstGrid[CALC_INDEX(x+1, y+1, z, NE)] = one_minus_omega * src_NE + omega_dfl3 * rho * (1.0f + ux_uy_p * (4.5f * ux_uy_p + 3.0f) - u2);
		dstGrid[CALC_INDEX(x-1, y+1, z, NW)] = one_minus_omega * src_NW + omega_dfl3 * rho * (1.0f + (-ux + uy) * (4.5f * (-ux + uy) + 3.0f) - u2);
		dstGrid[CALC_INDEX(x+1, y-1, z, SE)] = one_minus_omega * src_SE + omega_dfl3 * rho * (1.0f + ux_uy_m * (4.5f * ux_uy_m + 3.0f) - u2);
		dstGrid[CALC_INDEX(x-1, y-1, z, SW)] = one_minus_omega * src_SW + omega_dfl3 * rho * (1.0f + (-ux - uy) * (4.5f * (-

