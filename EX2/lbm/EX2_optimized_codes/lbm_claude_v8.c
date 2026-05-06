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

#define DFL1 (1.0f/ 3.0f)
#define DFL2 (1.0f/18.0f)
#define DFL3 (1.0f/36.0f)

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
	for(int cell = 0; cell < total_cells; cell++) {
		int base_idx = cell * N_CELL_ENTRIES;
		grid[base_idx + C ] = DFL1;
		grid[base_idx + N ] = DFL2;
		grid[base_idx + S ] = DFL2;
		grid[base_idx + E ] = DFL2;
		grid[base_idx + W ] = DFL2;
		grid[base_idx + T ] = DFL2;
		grid[base_idx + B ] = DFL2;
		grid[base_idx + NE] = DFL3;
		grid[base_idx + NW] = DFL3;
		grid[base_idx + SE] = DFL3;
		grid[base_idx + SW] = DFL3;
		grid[base_idx + NT] = DFL3;
		grid[base_idx + NB] = DFL3;
		grid[base_idx + ST] = DFL3;
		grid[base_idx + SB] = DFL3;
		grid[base_idx + ET] = DFL3;
		grid[base_idx + EB] = DFL3;
		grid[base_idx + WT] = DFL3;
		grid[base_idx + WB] = DFL3;
		
		unsigned int* flag_ptr = (unsigned int*)(&grid[base_idx + FLAGS]);
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
	const float dfl1_omega = DFL1 * omega;
	const float dfl2_omega = DFL2 * omega;
	const float dfl3_omega = DFL3 * omega;
	
	const int xy_size = SIZE_X * SIZE_Y;
	const int total_cells = xy_size * SIZE_Z;

#ifdef _OPENMP
	#pragma omp parallel for schedule(static)
#endif
	for(int cell = 0; cell < total_cells; cell++) {
		int z = cell / xy_size;
		int rem = cell % xy_size;
		int y = rem / SIZE_X;
		int x = rem % SIZE_X;
		
		int i = CALC_INDEX(x, y, z, 0);
		
		unsigned int flags = *((unsigned int*)(&srcGrid[i + FLAGS]));
		
		if( flags & OBSTACLE ) {
			dstGrid[i + C ] = srcGrid[i + C ];
			dstGrid[i + S ] = srcGrid[i + N ];
			dstGrid[i + N ] = srcGrid[i + S ];
			dstGrid[i + W ] = srcGrid[i + E ];
			dstGrid[i + E ] = srcGrid[i + W ];
			dstGrid[i + B ] = srcGrid[i + T ];
			dstGrid[i + T ] = srcGrid[i + B ];
			dstGrid[i + SW] = srcGrid[i + NE];
			dstGrid[i + SE] = srcGrid[i + NW];
			dstGrid[i + NW] = srcGrid[i + SE];
			dstGrid[i + NE] = srcGrid[i + SW];
			dstGrid[i + SB] = srcGrid[i + NT];
			dstGrid[i + ST] = srcGrid[i + NB];
			dstGrid[i + NB] = srcGrid[i + ST];
			dstGrid[i + NT] = srcGrid[i + SB];
			dstGrid[i + WB] = srcGrid[i + ET];
			dstGrid[i + WT] = srcGrid[i + EB];
			dstGrid[i + EB] = srcGrid[i + WT];
			dstGrid[i + ET] = srcGrid[i + WB];
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

		float rho = src_C + src_N + src_S + src_E + src_W + src_T + src_B +
		            src_NE + src_NW + src_SE + src_SW + src_NT + src_NB +
		            src_ST + src_SB + src_ET + src_EB + src_WT + src_WB;

		float ux = (src_E - src_W + src_NE - src_NW + src_SE - src_SW +
		            src_ET + src_EB - src_WT - src_WB) / rho;
		float uy = (src_N - src_S + src_NE + src_NW - src_SE - src_SW +
		            src_NT + src_NB - src_ST - src_SB) / rho;
		float uz = (src_T - src_B + src_NT - src_NB + src_ST - src_SB +
		            src_ET - src_EB + src_WT - src_WB) / rho;

		if( flags & ACCEL ) {
			ux = 0.005f;
			uy = 0.002f;
			uz = 0.000f;
		}

		float u2 = 1.5f * (ux*ux + uy*uy + uz*uz);
		
		float ux_45 = 4.5f * ux;
		float uy_45 = 4.5f * uy;
		float uz_45 = 4.5f * uz;
		
		float ux_term = ux_45 * ux;
		float uy_term = uy_45 * uy;
		float uz_term = uz_45 * uz;

		dstGrid[i + C ] = one_minus_omega * src_C  + dfl1_omega * rho * (1.0f - u2);

		dstGrid[i + N ] = one_minus_omega * src_N  + dfl2_omega * rho * (1.0f + uy_term + 3.0f * uy - u2);
		dstGrid[i + S ] = one_minus_omega * src_S  + dfl2_omega * rho * (1.0f + uy_term - 3.0f * uy - u2);
		dstGrid[i + E ] = one_minus_omega * src_E  + dfl2_omega * rho * (1.0f + ux_term + 3.0f * ux - u2);
		dstGrid[i + W ] = one_minus_omega * src_W  + dfl2_omega * rho * (1.0f + ux_term - 3.0f * ux - u2);
		dstGrid[i + T ] = one_minus_omega * src_T  + dfl2_omega * rho * (1.0f + uz_term + 3.0f * uz - u2);
		dstGrid[i + B ] = one_minus_omega * src_B  + dfl2_omega * rho * (1.0f + uz_term - 3.0f * uz - u2);

		float ux_uy_p = ux + uy;
		float ux_uy_m = ux - uy;
		float ux_uz_p = ux + uz;
		float ux_uz_m = ux - uz;
		float uy_uz_p = uy + uz;
		float uy_uz_m = uy - uz;

		dstGrid[i + NE] = one_minus_omega * src_NE + dfl3_omega * rho * (1.0f + ux_uy_p * (4.5f * ux_uy_p + 3.0f) - u2);
		dstGrid[i + NW] = one_minus_omega * src_NW + dfl3_omega * rho * (1.0f + (-ux + uy) * (4.5f * (-ux + uy) + 3.0f) - u2);
		dstGrid[i + SE] = one_minus_omega * src_SE + dfl3_omega * rho * (1.0f + ux_uy_m * (4.5f * ux_uy_m + 3.0f) - u2);
		dstGrid[i + SW] = one_minus_omega * src_SW + dfl3_omega * rho * (1.0f + (-ux - uy) * (4.5f * (-ux - uy) + 3.0f) - u2);
		dstGrid[i + NT] = one_minus_omega * src_NT + dfl3_omega * rho * (1.0f + uy_uz_p * (4.5f * uy_uz_p + 3.0f) - u2);
		dstGrid[i + NB] = one_minus_omega * src_NB + dfl3_omega * rho * (1.0f + uy_uz_m * (4.5f * uy_uz_m + 3.0f) - u2);
		dstGrid[i + ST] = one_minus_omega * src_ST + dfl3_omega * rho * (1.0f + (-uy + uz) * (4.5f * (-uy + uz) + 3.0f) - u2);
		dstGrid[i + SB] = one_minus_omega * src_SB + dfl3_omega * rho * (1.0f + (-uy - uz) * (4.5f * (-uy - uz) + 3.0f) - u2);
		dstGrid[i + ET] = one_minus_omega * src_ET + dfl3_omega * rho * (1.0f + ux_uz_p * (4.5f * ux_uz_p + 3.0f) - u2);
		dstGrid[i + EB] = one_minus_omega * src_EB + dfl3_omega * rho * (1.

