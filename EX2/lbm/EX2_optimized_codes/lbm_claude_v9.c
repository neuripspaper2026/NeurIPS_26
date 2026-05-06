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
	for(int cell = 0; cell < total_cells; cell++) {
		int i = cell * N_CELL_ENTRIES;
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
	const int start_z = 0;
	const int end_z = SIZE_Z;
	const int total_cells = SIZE_X * SIZE_Y * (end_z - start_z);

#ifdef _OPENMP
	#pragma omp parallel for schedule(static)
#endif
	for(int cell = 0; cell < total_cells; cell++) {
		int z = cell / (SIZE_X * SIZE_Y) + start_z;
		int rem = cell % (SIZE_X * SIZE_Y);
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

		float rho = src_C  + src_N  + src_S  + src_E  + src_W  + src_T  + src_B  +
		            src_NE + src_NW + src_SE + src_SW + src_NT + src_NB + src_ST +
		            src_SB + src_ET + src_EB + src_WT + src_WB;

		float ux = src_E  - src_W  + src_NE - src_NW + src_SE - src_SW +
		           src_ET + src_EB - src_WT - src_WB;
		float uy = src_N  - src_S  + src_NE + src_NW - src_SE - src_SW +
		           src_NT + src_NB - src_ST - src_SB;
		float uz = src_T  - src_B  + src_NT - src_NB + src_ST - src_SB +
		           src_ET - src_EB + src_WT - src_WB;

		float inv_rho = 1.0f / rho;
		ux *= inv_rho;
		uy *= inv_rho;
		uz *= inv_rho;

		if( flags & ACCEL ) {
			ux = 0.005f;
			uy = 0.002f;
			uz = 0.000f;
		}

		float u2 = 1.5f * (ux*ux + uy*uy + uz*uz);
		
		float omega_rho = OMEGA * rho;
		float one_minus_omega = 1.0f - OMEGA;
		
		float term_C = DFL1 * omega_rho * (1.0f - u2);
		dstGrid[i + C] = one_minus_omega * src_C + term_C;

		float uy_45 = 4.5f * uy;
		float uy_term = uy_45 * uy;
		float term_N = DFL2 * omega_rho * (1.0f + uy_term + 3.0f * uy - u2);
		float term_S = DFL2 * omega_rho * (1.0f + uy_term - 3.0f * uy - u2);
		dstGrid[i + N] = one_minus_omega * src_N + term_N;
		dstGrid[i + S] = one_minus_omega * src_S + term_S;

		float ux_45 = 4.5f * ux;
		float ux_term = ux_45 * ux;
		float term_E = DFL2 * omega_rho * (1.0f + ux_term + 3.0f * ux - u2);
		float term_W = DFL2 * omega_rho * (1.0f + ux_term - 3.0f * ux - u2);
		dstGrid[i + E] = one_minus_omega * src_E + term_E;
		dstGrid[i + W] = one_minus_omega * src_W + term_W;

		float uz_45 = 4.5f * uz;
		float uz_term = uz_45 * uz;
		float term_T = DFL2 * omega_rho * (1.0f + uz_term + 3.0f * uz - u2);
		float term_B = DFL2 * omega_rho * (1.0f + uz_term - 3.0f * uz - u2);
		dstGrid[i + T] = one_minus_omega * src_T + term_T;
		dstGrid[i + B] = one_minus_omega * src_B + term_B;

		float ux_uy_p = ux + uy;
		float ux_uy_p_term = 4.5f * ux_uy_p * ux_uy_p;
		float term_NE = DFL3 * omega_rho * (1.0f + ux_uy_p_term + 3.0f * ux_uy_p - u2);
		dstGrid[i + NE] = one_minus_omega * src_NE + term_NE;

		float ux_uy_m = -ux + uy;
		float ux_uy_m_term = 4.5f * ux_uy_m * ux_uy_m;
		float term_NW = DFL3 * omega_rho * (1.0f + ux_uy_m_term + 3.0f * ux_uy_m - u2);
		dstGrid[i + NW] = one_minus_omega * src_NW + term_NW;

		float ux_m_uy = ux - uy;
		float ux_m_uy_term = 4.5f * ux_m_uy * ux_m_uy;
		float term_SE = DFL3 * omega_rho * (1.0f + ux_m_uy_term + 3.0f * ux_m_uy - u2);
		dstGrid[i + SE] = one_minus_omega * src_SE + term_SE;

		float ux_uy_mm = -ux - uy;
		float ux_uy_mm_term = 4.5f * ux_uy_mm * ux_uy_mm;
		float term_SW = DFL3 * omega_rho * (1.0f + ux_uy_mm_term + 3.0f * ux_uy_mm - u2);
		dstGrid[i + SW] = one_minus_omega * src_SW + term_SW;

		float uy_uz_p = uy + uz;
		float uy_uz_p_term = 4.5f * uy_uz_p * uy_uz_p;
		float term_NT = DFL3 * omega_rho * (1.0f + uy_uz_p_term + 3.0f * uy_uz_p - u2);
		dstGrid[i + NT] = one_minus_omega * src_NT + term_NT;

		float uy_m_uz = uy - uz;
		float uy_m_uz_term = 4.5f * uy_m_uz * uy_m_uz;
		float term_N

