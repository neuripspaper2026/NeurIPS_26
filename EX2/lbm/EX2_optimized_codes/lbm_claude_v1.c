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
	#pragma omp parallel
	{
		#pragma omp for schedule(static) nowait
		for (int idx = 0; idx < total_cells; idx++) {
			int base = idx * N_CELL_ENTRIES;
			grid[base + C ] = DFL1;
			grid[base + N ] = DFL2;
			grid[base + S ] = DFL2;
			grid[base + E ] = DFL2;
			grid[base + W ] = DFL2;
			grid[base + T ] = DFL2;
			grid[base + B ] = DFL2;
			grid[base + NE] = DFL3;
			grid[base + NW] = DFL3;
			grid[base + SE] = DFL3;
			grid[base + SW] = DFL3;
			grid[base + NT] = DFL3;
			grid[base + NB] = DFL3;
			grid[base + ST] = DFL3;
			grid[base + SB] = DFL3;
			grid[base + ET] = DFL3;
			grid[base + EB] = DFL3;
			grid[base + WT] = DFL3;
			grid[base + WB] = DFL3;
			unsigned int* flag_ptr = (unsigned int*)(&grid[base + FLAGS]);
			*flag_ptr = 0;
		}
	}
#else
	SWEEP_VAR
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
#endif
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
	#pragma omp parallel for collapse(3) schedule(static) private(x, y, z)
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
	#pragma omp parallel for collapse(3) schedule(static) private(x, y, z)
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
	const float omega = OMEGA;
	const float one_minus_omega = 1.0f - omega;
	
#ifdef _OPENMP
	#pragma omp parallel
	{
		#pragma omp for schedule(static) nowait
		for (int idx = 0; idx < total_cells; idx++) {
			int z = idx / (SIZE_X * SIZE_Y);
			int rem = idx % (SIZE_X * SIZE_Y);
			int y = rem / SIZE_X;
			int x = rem % SIZE_X;
			
			int i = CALC_INDEX(x, y, z, 0);
			
			unsigned int flags = *((unsigned int*)(&srcGrid[i + FLAGS]));
			
			if (flags & OBSTACLE) {
				dstGrid[CALC_INDEX(x, y, z, C )] = srcGrid[i + C ];
				dstGrid[CALC_INDEX(x, y+1, z, S )] = srcGrid[i + N ];
				dstGrid[CALC_INDEX(x, y-1, z, N )] = srcGrid[i + S ];
				dstGrid[CALC_INDEX(x-1, y, z, E )] = srcGrid[i + W ];
				dstGrid[CALC_INDEX(x+1, y, z, W )] = srcGrid[i + E ];
				dstGrid[CALC_INDEX(x, y, z-1, T )] = srcGrid[i + B ];
				dstGrid[CALC_INDEX(x, y, z+1, B )] = srcGrid[i + T ];
				dstGrid[CALC_INDEX(x-1, y-1, z, NE)] = srcGrid[i + SW];
				dstGrid[CALC_INDEX(x-1, y+1, z, SE)] = srcGrid[i + NW];
				dstGrid[CALC_INDEX(x+1, y+1, z, SW)] = srcGrid[i + NE];
				dstGrid[CALC_INDEX(x+1, y-1, z, NW)] = srcGrid[i + SE];
				dstGrid[CALC_INDEX(x, y-1, z-1, NT)] = srcGrid[i + SB];
				dstGrid[CALC_INDEX(x, y-1, z+1, NB)] = srcGrid[i + ST];
				dstGrid[CALC_INDEX(x, y+1, z+1, SB)] = srcGrid[i + NT];
				dstGrid[CALC_INDEX(x, y+1, z-1, ST)] = srcGrid[i + NB];
				dstGrid[CALC_INDEX(x-1, y, z-1, ET)] = srcGrid[i + WB];
				dstGrid[CALC_INDEX(x-1, y, z+1, EB)] = srcGrid[i + WT];
				dstGrid[CALC_INDEX(x+1, y, z+1, WB)] = srcGrid[i + ET];
				dstGrid[CALC_INDEX(x+1, y, z-1, WT)] = srcGrid[i + EB];
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

			if (flags & ACCEL) {
				ux = 0.005f;
				uy = 0.002f;
				uz = 0.000f;
			}

			float u2 = 1.5f * (ux*ux + uy*uy + uz*uz);
			
			float uy_4_5 = 4.5f * uy;
			float ux_4_5 = 4.5f * ux;
			float uz_4_5 = 4.5f * uz;
			
			float term_uy_p = uy_4_5 * uy + 3.0f;
			float term_uy_m = uy_4_5 * uy - 3.0f;
			float term_ux_p = ux_4_5 * ux + 3.0f;
			float term_ux_m = ux_4_5 * ux - 3.0f;
			float term_uz_p = uz_4_5 * uz + 3.0f;
			float term_uz_m = uz_4_5 * uz - 3.0f;
			
			float base_1_u2 = 1.0f - u2;
			float rho_omega_dfl1 = DFL1 * omega * rho;
			float rho_omega_dfl2 = DFL2 * omega * rho;
			float rho_omega_dfl3 = DFL3 * omega * rho;

			dstGrid[CALC_INDEX(x, y, z, C)] = one_minus_omega * src_C + rho_omega_dfl1 * base_1_u2;

			dstGrid[CALC_INDEX(x, y+1, z, N)] = one_minus_omega * src_N + rho_omega_dfl2 * (base_1_u2 + uy * term_uy_p);
			dstGrid[CALC_INDEX(x, y-1, z, S)] = one_minus_omega * src_S + rho_omega_dfl2 * (base_1_u2 + uy * term_uy_m);
			dstGrid[CALC_INDEX(x+1, y, z, E)] = one_minus_omega * src_E + rho_omega_dfl2 * (base_1_u2 + ux * term_ux_p);
			dstGrid[CALC_INDEX(x-1, y, z, W)] = one_minus_omega * src_W + rho_omega_dfl2 * (base_1_u2 + ux * term_ux_m);
			dstGrid[CALC_INDEX(x, y, z+1, T)] = one_minus_omega * src_T + rho_omega_dfl2 * (base_1_u2 + uz * term_uz_p);
			dstGrid[CALC_INDEX(x, y, z-1, B)] = one_minus_omega * src_B + rho_omega_dfl2 * (base_1_u

