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
#ifdef _OPENMP
#pragma omp parallel for private(i) schedule(static)
#endif
	for( i = CALC_INDEX( 0, 0, -2, 0 ); \
	     i < CALC_INDEX( 0, 0, SIZE_Z+2, 0 ); \
	     i += N_CELL_ENTRIES ) {
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

	/*voption indep*/
#ifdef _OPENMP
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

	/*voption indep*/
#ifdef _OPENMP
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
	SWEEP_VAR

	float ux, uy, uz, u2, rho;

	/*voption indep*/
#ifdef _OPENMP
#pragma omp parallel for private(i,ux,uy,uz,u2,rho) schedule(static)
#endif
	for( i = CALC_INDEX( 0, 0, 0, 0 ); \
	     i < CALC_INDEX( 0, 0, SIZE_Z, 0 ); \
	     i += N_CELL_ENTRIES ) {
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
		DST_C ( dstGrid ) = (1.0f-OMEGA)*SRC_C ( srcGrid ) + (float)(DFL1*OMEGA)*rho*(1.0f                                 - u2);

		DST_N ( dstGrid ) = (1.0f-OMEGA)*SRC_N ( srcGrid ) + (float)(DFL2*OMEGA)*rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2);
		DST_S ( dstGrid ) = (1.0f-OMEGA)*SRC_S ( srcGrid ) + (float)(DFL2*OMEGA)*rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2);
		DST_E ( dstGrid ) = (1.0f-OMEGA)*SRC_E ( srcGrid ) + (float)(DFL2*OMEGA)*rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2);
		DST_W ( dstGrid ) = (1.0f-OMEGA)*SRC_W ( srcGrid ) + (float)(DFL2*OMEGA)*rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2);
		DST_T ( dstGrid ) = (1.0f-OMEGA)*SRC_T ( srcGrid ) + (float)(DFL2*OMEGA)*rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2);
		DST_B ( dstGrid ) = (1.0f-OMEGA)*SRC_B ( srcGrid ) + (float)(DFL2*OMEGA)*rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2);

		DST_NE( dstGrid ) = (1.0f-OMEGA)*SRC_NE( srcGrid ) + (float)(DFL3*OMEGA)*rho*(1.0f + (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
		DST_NW( dstGrid ) = (1.0f-OMEGA)*SRC_NW( srcGrid ) + (float)(DFL3*OMEGA)*rho*(1.0f + (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
		DST_SE( dstGrid ) = (1.0f-OMEGA)*SRC_SE( srcGrid ) + (float)(DFL3*OMEGA)*rho*(1.0f + (+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
		DST_SW( dstGrid ) = (1.0f-OMEGA)*SRC_SW( srcGrid ) + (float)(DFL3*OMEGA)*rho*(1.0f + (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
		DST_NT( dstGrid ) = (1.0f-OMEGA)*SRC_NT( srcGrid ) + (float)(DFL3*OMEGA)*rho*(1.0f + (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
		DST_NB( dstGrid ) = (1.0f-OMEGA)*SRC_NB( srcGrid ) + (float)(DFL3*OMEGA)*rho*(1.0f + (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
		DST_ST( dstGrid ) = (1.0f-OMEGA)*SRC_ST( srcGrid ) + (float)(DFL3*OMEGA)*rho*(1.0f + (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
		DST_SB( dstGrid ) = (1.0f-OMEGA)*SRC_SB( srcGrid ) + (float)(DFL3*OMEGA)*rho*(1.0f + (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
		DST_ET( dstGrid ) = (1.0f-OMEGA)*SRC_ET( srcGrid ) + (float)(DFL3*OMEGA)*rho*(1.0f + (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
		DST_EB( dstGrid ) = (1.0f-OMEGA)*SRC_EB( srcGrid ) + (float)(DFL3*OMEGA)*rho*(1.0f + (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
		DST_WT( dstGrid ) = (1.0f-OMEGA)*SRC_WT( srcGrid ) + (float)(DFL3*OMEGA)*rho*(1.0f + (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
		DST_WB( dstGrid ) = (1.0f-OMEGA)*SRC_WB( srcGrid ) + (float)(DFL3*OMEGA)*rho*(1.0f + (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
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
#ifdef _OPENMP
#pragma omp parallel for private(i,ux,uy,uz,rho,ux1,uy1,uz1,rho1,ux2,uy2,uz2,rho2,u2,px,py) schedule(static)
#endif
	for( i = CALC_INDEX( 0, 0, 0, 0 ); \
	     i < CALC_INDEX( 0, 0, 1, 0 ); \
	     i += N_CELL_ENTRIES ) {
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

		LOCAL( srcGrid, C ) = (OUTPUT_PRECISION)(DFL1*rho*(1.0f                                 - u2));

		LOCAL( srcGrid, N ) = (OUTPUT_PRECISION)(DFL2*rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2));
		LOCAL( srcGrid, S ) = (OUTPUT_PRECISION)(DFL2*rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2));
		LOCAL( srcGrid, E ) = (OUTPUT_PRECISION)(DFL2*rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2));
		LOCAL( srcGrid, W ) = (OUTPUT_PRECISION)(DFL2*rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2));
		LOCAL( srcGrid, T ) = (OUTPUT_PRECISION)(DFL2*rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2));
		LOCAL( srcGrid, B ) = (OUTPUT_PRECISION)(DFL2*rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2));

		LOCAL( srcGrid, NE) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2));
		LOCAL( srcGrid, NW) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2));
		LOCAL( srcGrid, SE) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2));
		LOCAL( srcGrid, SW) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2));
		LOCAL( srcGrid, NT) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2));
		LOCAL( srcGrid, NB) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2));
		LOCAL( srcGrid, ST) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2));
		LOCAL( srcGrid, SB) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2));
		LOCAL( srcGrid, ET) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2));
		LOCAL( srcGrid, EB) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2));
		LOCAL( srcGrid, WT) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2));
		LOCAL( srcGrid, WB) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2));
	}

	/* outflow */
	/*voption indep*/
#ifdef _OPENMP
#pragma omp parallel for private(i,ux,uy,uz,rho,ux1,uy1,uz1,rho1,ux2,uy2,uz2,rho2,u2) schedule(static)
#endif
	for( i = CALC_INDEX( 0, 0, SIZE_Z-1, 0 ); \
	     i < CALC_INDEX( 0, 0, SIZE_Z, 0 ); \
	     i += N_CELL_ENTRIES ) {
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

		LOCAL( srcGrid, C ) = (OUTPUT_PRECISION)(DFL1*rho*(1.0f                                 - u2));

		LOCAL( srcGrid, N ) = (OUTPUT_PRECISION)(DFL2*rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2));
		LOCAL( srcGrid, S ) = (OUTPUT_PRECISION)(DFL2*rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2));
		LOCAL( srcGrid, E ) = (OUTPUT_PRECISION)(DFL2*rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2));
		LOCAL( srcGrid, W ) = (OUTPUT_PRECISION)(DFL2*rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2));
		LOCAL( srcGrid, T ) = (OUTPUT_PRECISION)(DFL2*rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2));
		LOCAL( srcGrid, B ) = (OUTPUT_PRECISION)(DFL2*rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2));

		LOCAL( srcGrid, NE) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2));
		LOCAL( srcGrid, NW) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2));
		LOCAL( srcGrid, SE) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2));
		LOCAL( srcGrid, SW) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2));
		LOCAL( srcGrid, NT) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2));
		LOCAL( srcGrid, NB) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2));
		LOCAL( srcGrid, ST) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2));
		LOCAL( srcGrid, SB) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2));
		LOCAL( srcGrid, ET) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2));
		LOCAL( srcGrid, EB) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2));
		LOCAL( srcGrid, WT) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2));
		LOCAL( srcGrid, WB) = (OUTPUT_PRECISION)(DFL3*rho*(1.0f + (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2));
	}
}

/*############################################################################*/

void LBM_showGridStatistics( LBM_Grid grid ) {
	int nObstacleCells = 0,
	    nAccelCells    = 0,
	    nFluidCells    = 0;
	float minU2  = 1e+30f, maxU2  = -1e+30f;
	float minRho = 1e+30f, maxRho = -1e+30f;
	float mass = 0.0f;

	SWEEP_VAR

#ifdef _OPENMP
#pragma omp parallel private(i)
	{
		int nObstacleCells_local = 0;
		int nAccelCells_local = 0;
		int nFluidCells_local = 0;
		float minU2_local  = 1e+30f, maxU2_local  = -1e+30f;
		float minRho_local = 1e+30f, maxRho_local = -1e+30f;
		float mass_local = 0.0f;
#pragma omp for schedule(static)
#endif
		for( i = CALC_INDEX( 0, 0, 0, 0 ); \
		     i < CALC_INDEX( 0, 0, SIZE_Z, 0 ); \
		     i += N_CELL_ENTRIES ) {
			float rho, ux, uy, uz, u2;
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
			if( rho < minRho_local ) minRho_local = rho;
			if( rho > maxRho_local ) maxRho_local = rho;
			mass_local += rho;

			if( TEST_FLAG_SWEEP( grid, OBSTACLE )) {
				nObstacleCells_local++;
			}
			else {
				if( TEST_FLAG_SWEEP( grid, ACCEL ))
					nAccelCells_local++;
				else
					nFluidCells_local++;

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
				if( u2 < minU2_local ) minU2_local = u2;
				if( u2 > maxU2_local ) maxU2_local = u2;
			}
		}
#ifdef _OPENMP
#pragma omp critical
		{
			nObstacleCells += nObstacleCells_local;
			nAccelCells    += nAccelCells_local;
			nFluidCells    += nFluidCells_local;
			if (minRho_local < minRho) minRho = minRho_local;
			if (maxRho_local > maxRho) maxRho = maxRho_local;
			if (minU2_local  < minU2 ) minU2  = minU2_local;
			if (maxU2_local  > maxU2 ) maxU2  = maxU2_local;
			mass += mass_local;
		}
	}
#else
	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		{
			float rho, ux, uy, uz, u2;
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
		const char* vPtr = (const char*) v;
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

