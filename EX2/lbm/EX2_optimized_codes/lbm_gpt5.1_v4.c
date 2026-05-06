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

	/*voption indep*/
#if defined(_OPENMP) && !defined(SPEC_CPU)
#pragma omp parallel for private(i,ux,uy,uz,u2,rho) schedule(static)
#endif
	for( i = CALC_INDEX( 0, 0, 0, 0 ); i < CALC_INDEX( 0, 0, SIZE_Z, 0 ); i += N_CELL_ENTRIES ) {
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

		rho = + SRC_C ( srcGrid ) + SRC_N ( srcGrid )
		      + SRC_S ( srcGrid ) + SRC_E ( srcGrid )
		      + SRC_W ( srcGrid ) + SRC_T ( srcGrid )
		      + SRC_B ( srcGrid ) + SRC_NE( srcGrid )
		      + SRC_NW( srcGrid ) + SRC_SE( srcGrid )
		      + SRC_SW( srcGrid ) + SRC_NT( srcGrid )
		      + SRC_NB( srcGrid ) + SRC_ST( srcGrid )
		      + SRC_SB( srcGrid ) + SRC_ET( srcGrid )
		      + SRC_EB( srcGrid ) + SRC_WT( srcGrid )
		      + SRC_WB( srcGrid );

		ux = + SRC_E ( srcGrid ) - SRC_W ( srcGrid )
		     + SRC_NE( srcGrid ) - SRC_NW( srcGrid )
		     + SRC_SE( srcGrid ) - SRC_SW( srcGrid )
		     + SRC_ET( srcGrid ) + SRC_EB( srcGrid )
		     - SRC_WT( srcGrid ) - SRC_WB( srcGrid );
		uy = + SRC_N ( srcGrid ) - SRC_S ( srcGrid )
		     + SRC_NE( srcGrid ) + SRC_NW( srcGrid )
		     - SRC_SE( srcGrid ) - SRC_SW( srcGrid )
		     + SRC_NT( srcGrid ) + SRC_NB( srcGrid )
		     - SRC_ST( srcGrid ) - SRC_SB( srcGrid );
		uz = + SRC_T ( srcGrid ) - SRC_B ( srcGrid )
		     + SRC_NT( srcGrid ) - SRC_NB( srcGrid )
		     + SRC_ST( srcGrid ) - SRC_SB( srcGrid )
		     + SRC_ET( srcGrid ) - SRC_EB( srcGrid )
		     + SRC_WT( srcGrid ) - SRC_WB( srcGrid );

		ux /= rho;
		uy /= rho;
		uz /= rho;

		if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
			ux = 0.005f;
			uy = 0.002f;
			uz = 0.000f;
		}

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);
		{
			const float omega1 = 1.0f - OMEGA;
			const float omega_dfl1 = (float)DFL1 * OMEGA;
			const float omega_dfl2 = (float)DFL2 * OMEGA;
			const float omega_dfl3 = (float)DFL3 * OMEGA;

			DST_C ( dstGrid ) = omega1*SRC_C ( srcGrid ) + omega_dfl1*rho*(1.0f                                 - u2);

			DST_N ( dstGrid ) = omega1*SRC_N ( srcGrid ) + omega_dfl2*rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2);
			DST_S ( dstGrid ) = omega1*SRC_S ( srcGrid ) + omega_dfl2*rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2);
			DST_E ( dstGrid ) = omega1*SRC_E ( srcGrid ) + omega_dfl2*rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2);
			DST_W ( dstGrid ) = omega1*SRC_W ( srcGrid ) + omega_dfl2*rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2);
			DST_T ( dstGrid ) = omega1*SRC_T ( srcGrid ) + omega_dfl2*rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2);
			DST_B ( dstGrid ) = omega1*SRC_B ( srcGrid ) + omega_dfl2*rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2);

			DST_NE( dstGrid ) = omega1*SRC_NE( srcGrid ) + omega_dfl3*rho*(1.0f + (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
			DST_NW( dstGrid ) = omega1*SRC_NW( srcGrid ) + omega_dfl3*rho*(1.0f + (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
			DST_SE( dstGrid ) = omega1*SRC_SE( srcGrid ) + omega_dfl3*rho*(1.0f + (+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
			DST_SW( dstGrid ) = omega1*SRC_SW( srcGrid ) + omega_dfl3*rho*(1.0f + (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
			DST_NT( dstGrid ) = omega1*SRC_NT( srcGrid ) + omega_dfl3*rho*(1.0f + (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
			DST_NB( dstGrid ) = omega1*SRC_NB( srcGrid ) + omega_dfl3*rho*(1.0f + (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
			DST_ST( dstGrid ) = omega1*SRC_ST( srcGrid ) + omega_dfl3*rho*(1.0f + (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
			DST_SB( dstGrid ) = omega1*SRC_SB( srcGrid ) + omega_dfl3*rho*(1.0f + (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
			DST_ET( dstGrid ) = omega1*SRC_ET( srcGrid ) + omega_dfl3*rho*(1.0f + (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
			DST_EB( dstGrid ) = omega1*SRC_EB( srcGrid ) + omega_dfl3*rho*(1.0f + (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
			DST_WT( dstGrid ) = omega1*SRC_WT( srcGrid ) + omega_dfl3*rho*(1.0f + (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
			DST_WB( dstGrid ) = omega1*SRC_WB( srcGrid ) + omega_dfl3*rho*(1.0f + (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
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
	for( i = CALC_INDEX( 0, 0, 0, 0 ); i < CALC_INDEX( 0, 0, 1, 0 ); i += N_CELL_ENTRIES ) {
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

		px = (SWEEP_X / (0.5f*(SIZE_X-1))) - 1.0f;
		py = (SWEEP_Y / (0.5f*(SIZE_Y-1))) - 1.0f;
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
#pragma omp parallel for private(i,ux,uy,uz,rho,ux1,uy1,uz1,rho1,ux2,uy2,uz2,rho2,u2,px,py) schedule(static)
#endif
	for( i = CALC_INDEX( 0, 0, SIZE_Z-1, 0 ); i < CALC_INDEX( 0, 0, SIZE_Z, 0 ); i += N_CELL_ENTRIES ) {
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
	float mass = 0;

	SWEEP_VAR

#if defined(_OPENMP) && !defined(SPEC_CPU)
	int nObstacleCells_private, nAccelCells_private, nFluidCells_private;
	float minU2_private, maxU2_private, minRho_private, maxRho_private, mass_private;

	nObstacleCells = 0;
	nAccelCells = 0;
	nFluidCells = 0;

#pragma omp parallel private(i,ux,uy,uz,u2,rho,nObstacleCells_private,nAccelCells_private,nFluidCells_private,minU2_private,maxU2_private,minRho_private,maxRho_private,mass_private)
	{
		nObstacleCells_private = 0;
		nAccelCells_private = 0;
		nFluidCells_private = 0;
		minU2_private = 1e+30f;
		maxU2_private = -1e+30f;
		minRho_private = 1e+30f;
		maxRho_private = -1e+30f;
		mass_private = 0.0f;

#pragma omp for nowait schedule(static)
		for( i = CALC_INDEX( 0, 0, 0, 0 ); i < CALC_INDEX( 0, 0, SIZE_Z, 0 ); i += N_CELL_ENTRIES ) {
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
			if( rho < minRho_private ) minRho_private = rho;
			if( rho > maxRho_private ) maxRho_private = rho;
			mass_private += rho;

			if( TEST_FLAG_SWEEP( grid, OBSTACLE )) {
				nObstacleCells_private++;
			}
			else {
				if( TEST_FLAG_SWEEP( grid, ACCEL ))
					nAccelCells_private++;
				else
					nFluidCells_private++;

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
				if( u2 < minU2_private ) minU2_private = u2;
				if( u2 > maxU2_private ) maxU2_private = u2;
			}
		}

#pragma omp critical
		{
			nObstacleCells += nObstacleCells_private;
			nAccelCells    += nAccelCells_private;
			nFluidCells    += nFluidCells_private;
			if( minRho_private < minRho ) minRho = minRho_private;
			if( maxRho_private > maxRho ) maxRho = maxRho_private;
			if( minU2_private < minU2 )   minU2 = minU2_private;
			if( maxU2_private > maxU2 )   maxU2 = maxU2_private;
			mass += mass_private;
		}
	}
#else
	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
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
			u2 = (ux*ux + uy*uy + uz*uz) / (rho*rho);
			if( u2 < minU2 ) minU2 = u2;
			if( u2 > maxU2 ) maxU2 = u2;
		}
	SWEEP_END
#endif

        printf( "LBM_showGridStatistics:\n"
        "\tnObstacleCells: %7i nAccelCells: %7i nFluidCells: %7i\n"
        "\tminRho: %8.4f maxRho: %8.4f mass: %e\n"
        "\tminU: %e maxU: %e\n\n",
        nObstacleCells, nAccelCells, nFluidCells,
        minRho, maxRho, mass,
        sqrt( minU2 ), sqrt( maxU2 ) );

}

/*############################################################################*/

static void storeValue( FILE* file, OUTPUT_PRECISION* v ) {
	const int litteBigEndianTest = 1;
	if( (*((unsigned char*) &litteBigEndianTest)) == 0 ) {         /* big endian */
		const char* vPtr = (char*) v;
		char buffer[sizeof( OUTPUT_PRECISION )];
		int i;

		for (i = 0; i < (int)sizeof( OUTPUT_PRECISION ); i++)
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
		int i;

		fread( buffer, sizeof( OUTPUT_PRECISION ), 1, file );

		for (i = 0; i < (int)sizeof( OUTPUT_PRECISION ); i++)
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
					fprintf( file, "%e %e %e\n", ux, uy, uz );

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
	        sqrt( maxDiff2 )  );
#else
	printf( "LBM_compareVelocityField: maxDiff = %e  ==>  %s\n\n",
	        sqrt( maxDiff2 ),
	        sqrt( maxDiff2 ) > 1e-5 ? "##### ERROR #####" : "OK" );
#endif
	fclose( file );
}

