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
	for(int idx = 0; idx < total_cells; idx++) {
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
		*((unsigned int*)&grid[base + FLAGS]) = 0;
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
	#pragma omp parallel for collapse(3) private(x,y,z) schedule(static)
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
	#pragma omp parallel for collapse(3) private(x,y,z) schedule(static)
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
	
#ifdef _OPENMP
	#pragma omp parallel
#endif
	{
		int i_start = CALC_INDEX(0, 0, 0, 0);
		int i_end = CALC_INDEX(0, 0, SIZE_Z, 0);
		
#ifdef _OPENMP
		int num_threads = omp_get_num_threads();
		int tid = omp_get_thread_num();
		int total_work = (i_end - i_start) / N_CELL_ENTRIES;
		int work_per_thread = (total_work + num_threads - 1) / num_threads;
		int my_start = i_start + tid * work_per_thread * N_CELL_ENTRIES;
		int my_end = i_start + (tid + 1) * work_per_thread * N_CELL_ENTRIES;
		if(my_end > i_end) my_end = i_end;
#else
		int my_start = i_start;
		int my_end = i_end;
#endif
		
		for(int i = my_start; i < my_end; i += N_CELL_ENTRIES) {
			unsigned int flags = *((unsigned int*)&srcGrid[i + FLAGS]);
			
			if(flags & OBSTACLE) {
				dstGrid[i + C ] = srcGrid[i + C ];
				dstGrid[i + N ] = srcGrid[CALC_INDEX(0, -1, 0, N ) + i];
				dstGrid[i + S ] = srcGrid[CALC_INDEX(0, +1, 0, S ) + i];
				dstGrid[i + E ] = srcGrid[CALC_INDEX(-1, 0, 0, E ) + i];
				dstGrid[i + W ] = srcGrid[CALC_INDEX(+1, 0, 0, W ) + i];
				dstGrid[i + T ] = srcGrid[CALC_INDEX(0, 0, -1, T ) + i];
				dstGrid[i + B ] = srcGrid[CALC_INDEX(0, 0, +1, B ) + i];
				dstGrid[i + NE] = srcGrid[CALC_INDEX(-1, -1, 0, NE) + i];
				dstGrid[i + NW] = srcGrid[CALC_INDEX(+1, -1, 0, NW) + i];
				dstGrid[i + SE] = srcGrid[CALC_INDEX(-1, +1, 0, SE) + i];
				dstGrid[i + SW] = srcGrid[CALC_INDEX(+1, +1, 0, SW) + i];
				dstGrid[i + NT] = srcGrid[CALC_INDEX(0, -1, -1, NT) + i];
				dstGrid[i + NB] = srcGrid[CALC_INDEX(0, -1, +1, NB) + i];
				dstGrid[i + ST] = srcGrid[CALC_INDEX(0, +1, -1, ST) + i];
				dstGrid[i + SB] = srcGrid[CALC_INDEX(0, +1, +1, SB) + i];
				dstGrid[i + ET] = srcGrid[CALC_INDEX(-1, 0, -1, ET) + i];
				dstGrid[i + EB] = srcGrid[CALC_INDEX(-1, 0, +1, EB) + i];
				dstGrid[i + WT] = srcGrid[CALC_INDEX(+1, 0, -1, WT) + i];
				dstGrid[i + WB] = srcGrid[CALC_INDEX(+1, 0, +1, WB) + i];
				continue;
			}

			float src_c  = srcGrid[i + C ];
			float src_n  = srcGrid[i + N ];
			float src_s  = srcGrid[i + S ];
			float src_e  = srcGrid[i + E ];
			float src_w  = srcGrid[i + W ];
			float src_t  = srcGrid[i + T ];
			float src_b  = srcGrid[i + B ];
			float src_ne = srcGrid[i + NE];
			float src_nw = srcGrid[i + NW];
			float src_se = srcGrid[i + SE];
			float src_sw = srcGrid[i + SW];
			float src_nt = srcGrid[i + NT];
			float src_nb = srcGrid[i + NB];
			float src_st = srcGrid[i + ST];
			float src_sb = srcGrid[i + SB];
			float src_et = srcGrid[i + ET];
			float src_eb = srcGrid[i + EB];
			float src_wt = srcGrid[i + WT];
			float src_wb = srcGrid[i + WB];

			float rho = src_c + src_n + src_s + src_e + src_w + src_t + src_b +
			            src_ne + src_nw + src_se + src_sw + src_nt + src_nb +
			            src_st + src_sb + src_et + src_eb + src_wt + src_wb;

			float ux = (src_e - src_w + src_ne - src_nw + src_se - src_sw +
			            src_et + src_eb - src_wt - src_wb) / rho;
			float uy = (src_n - src_s + src_ne + src_nw - src_se - src_sw +
			            src_nt + src_nb - src_st - src_sb) / rho;
			float uz = (src_t - src_b + src_nt - src_nb + src_st - src_sb +
			            src_et - src_eb + src_wt - src_wb) / rho;

			if(flags & ACCEL) {
				ux = 0.005f;
				uy = 0.002f;
				uz = 0.000f;
			}

			float u2 = 1.5f * (ux*ux + uy*uy + uz*uz);
			
			float ux3 = 3.0f * ux;
			float uy3 = 3.0f * uy;
			float uz3 = 3.0f * uz;
			float ux_45 = 4.5f * ux;
			float uy_45 = 4.5f * uy;
			float uz_45 = 4.5f * uz;

			dstGrid[CALC_INDEX(0, 0, 0, C) + i] = one_minus_omega * src_c + omega_dfl1 * rho * (1.0f - u2);

			float term_n = uy_45 * uy + uy3;
			dstGrid[CALC_INDEX(0, +1, 0, N) + i] = one_minus_omega * src_n + omega_dfl2 * rho * (1.0f + term_n - u2);
			dstGrid[CALC_INDEX(0, -1, 0, S) + i] = one_minus_omega * src_s + omega_dfl2 * rho * (1.0f - term_n - u2);

			float term_e = ux_45 * ux + ux3;
			dstGrid[CALC_INDEX(+1, 0, 0, E) + i] = one_minus_omega * src_e + omega_dfl2 * rho * (1.0f + term_e - u2);
			dstGrid[CALC_INDEX(-1, 0, 0, W) + i] = one_minus_omega * src_w + omega_dfl2 * rho * (1.0f - term_e - u2);

			float term_t = uz_45 * uz + uz3;
			dstGrid[CALC_INDEX(0, 0, +1, T) + i] = one_minus_omega * src_t + omega_dfl2 * rho * (1.0f + term_t - u2);
			dstGrid[CALC_INDEX(0, 0, -1, B) + i] = one_minus_omega * src_b + omega_dfl2 * rho * (1.0f - term_t - u2);

			float ux_uy_p = ux + uy;
			float term_ne = 4.5f * ux_uy_p * ux_uy_p + 3.0f * ux_uy_p;
			dstGrid[CALC_INDEX(+1, +1, 0, NE) + i] = one_minus_omega * src_ne + omega_dfl3 * rho * (1.0f + term_ne - u2);

			float ux_uy_m = -ux + uy;
			float term_nw = 4.5f * ux_uy_m * ux_uy_m + 3.0f * ux_uy_m;
			dstGrid[CALC_INDEX(-1, +1, 0, NW) + i] = one_minus_omega * src_nw + omega_dfl3 * rho * (1.0f + term_nw - u2);

			float ux_m_uy = ux - uy;
			float term_se = 4.5f * ux_m_uy * ux_m_

