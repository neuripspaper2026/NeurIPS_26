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
	const int total_cells = SIZE_X * SIZE_Y * (SIZE_Z + 4);
	
#if !defined(SPEC_CPU) && defined(_OPENMP)
	#pragma omp parallel for schedule(static)
#endif
	for(int cell_idx = 0; cell_idx < total_cells; cell_idx++) {
		int base = cell_idx * N_CELL_ENTRIES;
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
		*((unsigned int*)(&grid[base + FLAGS])) = 0;
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

#if !defined(SPEC_CPU) && defined(_OPENMP)
	#pragma omp parallel for private(x,y) schedule(static)
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

#if !defined(SPEC_CPU) && defined(_OPENMP)
	#pragma omp parallel for private(x,y) schedule(static)
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
	
	const int plane_size = SIZE_X * SIZE_Y;
	const int total_cells = SIZE_Z * plane_size;

#if !defined(SPEC_CPU) && defined(_OPENMP)
	#pragma omp parallel for schedule(static)
#endif
	for(int cell_idx = 0; cell_idx < total_cells; cell_idx++) {
		int i = cell_idx * N_CELL_ENTRIES;
		
		unsigned int flags = *((unsigned int*)(&srcGrid[i + FLAGS]));
		
		if(flags & OBSTACLE) {
			dstGrid[i + C ] = srcGrid[i + C ];
			dstGrid[i + N ] = srcGrid[i + S ];
			dstGrid[i + S ] = srcGrid[i + N ];
			dstGrid[i + E ] = srcGrid[i + W ];
			dstGrid[i + W ] = srcGrid[i + E ];
			dstGrid[i + T ] = srcGrid[i + B ];
			dstGrid[i + B ] = srcGrid[i + T ];
			dstGrid[i + NE] = srcGrid[i + SW];
			dstGrid[i + NW] = srcGrid[i + SE];
			dstGrid[i + SE] = srcGrid[i + NW];
			dstGrid[i + SW] = srcGrid[i + NE];
			dstGrid[i + NT] = srcGrid[i + SB];
			dstGrid[i + NB] = srcGrid[i + ST];
			dstGrid[i + ST] = srcGrid[i + NB];
			dstGrid[i + SB] = srcGrid[i + NT];
			dstGrid[i + ET] = srcGrid[i + WB];
			dstGrid[i + EB] = srcGrid[i + WT];
			dstGrid[i + WT] = srcGrid[i + EB];
			dstGrid[i + WB] = srcGrid[i + ET];
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

		if(flags & ACCEL) {
			ux = 0.005f;
			uy = 0.002f;
			uz = 0.000f;
		}

		float u2 = 1.5f * (ux*ux + uy*uy + uz*uz);
		
		dstGrid[i + C ] = one_minus_omega * src_C  + dfl1_omega * rho * (1.0f - u2);

		float uy_term = 4.5f * uy;
		dstGrid[i + N ] = one_minus_omega * src_N  + dfl2_omega * rho * (1.0f + uy * (uy_term + 3.0f) - u2);
		dstGrid[i + S ] = one_minus_omega * src_S  + dfl2_omega * rho * (1.0f + uy * (uy_term - 3.0f) - u2);
		
		float ux_term = 4.5f * ux;
		dstGrid[i + E ] = one_minus_omega * src_E  + dfl2_omega * rho * (1.0f + ux * (ux_term + 3.0f) - u2);
		dstGrid[i + W ] = one_minus_omega * src_W  + dfl2_omega * rho * (1.0f + ux * (ux_term - 3.0f) - u2);
		
		float uz_term = 4.5f * uz;
		dstGrid[i + T ] = one_minus_omega * src_T  + dfl2_omega * rho * (1.0f + uz * (uz_term + 3.0f) - u2);
		dstGrid[i + B ] = one_minus_omega * src_B  + dfl2_omega * rho * (1.0f + uz * (uz_term - 3.0f) - u2);

		float ux_uy_p = ux + uy;
		float ux_uy_p_term = 4.5f * ux_uy_p;
		dstGrid[i + NE] = one_minus_omega * src_NE + dfl3_omega * rho * (1.0f + ux_uy_p * (ux_uy_p_term + 3.0f) - u2);
		
		float ux_uy_n = -ux + uy;
		float ux_uy_n_term = 4.5f * ux_uy_n;
		dstGrid[i + NW] = one_minus_omega * src_NW + dfl3_omega * rho * (1.0f + ux_uy_n * (ux_uy_n_term + 3.0f) - u2);
		
		float ux_uy_pn = ux - uy;
		float ux_uy_pn_term = 4.5f * ux_uy_pn;
		dstGrid[i + SE] = one_minus_omega * src_SE + dfl3_omega * rho * (1.0f + ux_uy_pn * (ux_uy_pn_term + 3.0f) - u2);
		
		float ux_uy_nn = -ux - uy;
		float ux_uy_nn_term = 4.5f * ux_uy_nn;
		dstGrid[i + SW] = one_minus_omega * src_SW + dfl3_omega * rho * (1.0f + ux_uy_nn * (ux_uy_nn_term + 3.0f) - u2);
		
		float uy_uz_p = uy + uz;
		float uy_uz_p_term = 4.5f * uy_uz_p;
		dstGrid[i + NT] = one_minus_omega * src_NT + dfl3_omega * rho * (1.0f + uy_uz_p * (uy_uz_p_term + 3.0f) - u2);
		
		float uy_uz_pn = uy - uz;
		float uy_uz_pn_term = 4.5f * uy_uz_pn;
		dstGrid[i + NB] = one_minus_omega * src_NB + dfl3_omega * rho * (1.0f + uy_uz_pn * (uy_uz_pn_term + 3.0f) - u2);
		
		float uy_uz_np = -uy + uz;
		float uy_uz_np_term = 4.5f * uy_uz_np;
		dstGrid[i + ST] = one_minus_omega * src_ST + dfl3_omega * rho * (1.0f + uy_uz_np * (uy_uz_np_term + 3.0f) - u2);
		
		float uy_uz_nn = -uy - uz;
		float uy_uz_nn_term = 4.5f * uy_uz_nn;
		dstGrid[i +

