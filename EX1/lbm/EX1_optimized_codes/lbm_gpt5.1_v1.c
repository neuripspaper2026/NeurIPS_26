#include "lbm.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#if !defined(SPEC_CPU)
#ifdef _OPENMP
#include <omp.h>
#endif
#endif

#define DFL1 (1.0f/ 3.0f)
#define DFL2 (1.0f/18.0f)
#define DFL3 (1.0f/36.0f)

void LBM_allocateGrid( float** ptr ) {
	const size_t margin = 2u * (size_t)SIZE_X * (size_t)SIZE_Y * (size_t)N_CELL_ENTRIES;
	const size_t size   = sizeof( LBM_Grid ) + 2u * margin * sizeof( float );

	*ptr = (float*)malloc( size );
	if( !*ptr ) {
		printf( "LBM_allocateGrid: could not allocate %.1f MByte\n",
		        (double)size / (1024.0*1024.0) );
		exit( 1 );
	}
#if !defined(SPEC_CPU)
	printf( "LBM_allocateGrid: allocated %.1f MByte\n",
	        (double)size / (1024.0*1024.0) );
#endif
	*ptr += margin;
}

void LBM_freeGrid( float** ptr ) {
	const size_t margin = 2u * (size_t)SIZE_X * (size_t)SIZE_Y * (size_t)N_CELL_ENTRIES;

	free( *ptr - margin );
	*ptr = NULL;
}

void LBM_initializeGrid( LBM_Grid grid ) {
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
}

void LBM_swapGrids( LBM_GridPtr* grid1, LBM_GridPtr* grid2 ) {
	LBM_GridPtr aux = *grid1;
	*grid1 = *grid2;
	*grid2 = aux;
}

void LBM_loadObstacleFile( LBM_Grid grid, const char* filename ) {
	int x,  y,  z;

	FILE* file = fopen( filename, "rb" );
	if (!file) {
		perror("LBM_loadObstacleFile");
		exit(1);
	}

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				int c = fgetc( file );
				if( c != '.' && c != EOF )
					SET_FLAG( grid, x, y, z, OBSTACLE );
			}
			(void)fgetc( file );
		}
		(void)fgetc( file );
	}

	fclose( file );
}

void LBM_initializeSpecialCellsForLDC( LBM_Grid grid ) {
	int x,  y,  z;

	for( z = -2; z < SIZE_Z+2; z++ ) {
		const int z_is_wall = (z == 0 || z == SIZE_Z-1);
		const int z_is_accel = (z == 1 || z == SIZE_Z-2);
		for( y = 0; y < SIZE_Y; y++ ) {
			const int y_is_wall = (y == 0 || y == SIZE_Y-1);
			const int y_in_accel = (y > 1 && y < SIZE_Y-2);
			for( x = 0; x < SIZE_X; x++ ) {
				const int x_is_wall = (x == 0 || x == SIZE_X-1);
				if( x_is_wall || y_is_wall || z_is_wall ) {
					SET_FLAG( grid, x, y, z, OBSTACLE );
				}
				else if( z_is_accel &&
				         x > 1 && x < SIZE_X-2 &&
				         y_in_accel ) {
					SET_FLAG( grid, x, y, z, ACCEL );
				}
			}
		}
	}
}

void LBM_initializeSpecialCellsForChannel( LBM_Grid grid ) {
	int x,  y,  z;

	for( z = -2; z < SIZE_Z+2; z++ ) {
		const int z_is_inout = (z == 0 || z == SIZE_Z-1);
		for( y = 0; y < SIZE_Y; y++ ) {
			const int y_is_wall = (y == 0 || y == SIZE_Y-1);
			for( x = 0; x < SIZE_X; x++ ) {
				const int x_is_wall = (x == 0 || x == SIZE_X-1);
				if( x_is_wall || y_is_wall ) {
					SET_FLAG( grid, x, y, z, OBSTACLE );

					if( z_is_inout &&
					    !TEST_FLAG( grid, x, y, z, OBSTACLE ))
						SET_FLAG( grid, x, y, z, IN_OUT_FLOW );
				}
			}
		}
	}
}

void LBM_performStreamCollide( LBM_Grid srcGrid, LBM_Grid dstGrid ) {
	SWEEP_VAR

	float ux, uy, uz, u2, rho;

	const float omega   = OMEGA;
	const float omega1m = 1.0f - omega;
	const float dfl1o   = DFL1 * omega;
	const float dfl2o   = DFL2 * omega;
	const float dfl3o   = DFL3 * omega;

	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
			const float srcC  = SRC_C ( srcGrid );
			const float srcN  = SRC_N ( srcGrid );
			const float srcS  = SRC_S ( srcGrid );
			const float srcE  = SRC_E ( srcGrid );
			const float srcW  = SRC_W ( srcGrid );
			const float srcT  = SRC_T ( srcGrid );
			const float srcB  = SRC_B ( srcGrid );
			const float srcNE = SRC_NE( srcGrid );
			const float srcNW = SRC_NW( srcGrid );
			const float srcSE = SRC_SE( srcGrid );
			const float srcSW = SRC_SW( srcGrid );
			const float srcNT = SRC_NT( srcGrid );
			const float srcNB = SRC_NB( srcGrid );
			const float srcST = SRC_ST( srcGrid );
			const float srcSB = SRC_SB( srcGrid );
			const float srcET = SRC_ET( srcGrid );
			const float srcEB = SRC_EB( srcGrid );
			const float srcWT = SRC_WT( srcGrid );
			const float srcWB = SRC_WB( srcGrid );

			DST_C ( dstGrid ) = srcC;
			DST_S ( dstGrid ) = srcN;
			DST_N ( dstGrid ) = srcS;
			DST_W ( dstGrid ) = srcE;
			DST_E ( dstGrid ) = srcW;
			DST_B ( dstGrid ) = srcT;
			DST_T ( dstGrid ) = srcB;
			DST_SW( dstGrid ) = srcNE;
			DST_SE( dstGrid ) = srcNW;
			DST_NW( dstGrid ) = srcSE;
			DST_NE( dstGrid ) = srcSW;
			DST_SB( dstGrid ) = srcNT;
			DST_ST( dstGrid ) = srcNB;
			DST_NB( dstGrid ) = srcST;
			DST_NT( dstGrid ) = srcSB;
			DST_WB( dstGrid ) = srcET;
			DST_WT( dstGrid ) = srcEB;
			DST_EB( dstGrid ) = srcWT;
			DST_ET( dstGrid ) = srcWB;
			continue;
		}

		{
			const float srcC  = SRC_C ( srcGrid );
			const float srcN  = SRC_N ( srcGrid );
			const float srcS  = SRC_S ( srcGrid );
			const float srcE  = SRC_E ( srcGrid );
			const float srcW  = SRC_W ( srcGrid );
			const float srcT  = SRC_T ( srcGrid );
			const float srcB  = SRC_B ( srcGrid );
			const float srcNE = SRC_NE( srcGrid );
			const float srcNW = SRC_NW( srcGrid );
			const float srcSE = SRC_SE( srcGrid );
			const float srcSW = SRC_SW( srcGrid );
			const float srcNT = SRC_NT( srcGrid );
			const float srcNB = SRC_NB( srcGrid );
			const float srcST = SRC_ST( srcGrid );
			const float srcSB = SRC_SB( srcGrid );
			const float srcET = SRC_ET( srcGrid );
			const float srcEB = SRC_EB( srcGrid );
			const float srcWT = SRC_WT( srcGrid );
			const float srcWB = SRC_WB( srcGrid );

			rho = srcC + srcN + srcS + srcE + srcW + srcT + srcB +
			      srcNE + srcNW + srcSE + srcSW +
			      srcNT + srcNB + srcST + srcSB +
			      srcET + srcEB + srcWT + srcWB;

			ux = + srcE - srcW
			     + srcNE - srcNW
			     + srcSE - srcSW
			     + srcET + srcEB
			     - srcWT - srcWB;
			uy = + srcN - srcS
			     + srcNE + srcNW
			     - srcSE - srcSW
			     + srcNT + srcNB
			     - srcST - srcSB;
			uz = + srcT - srcB
			     + srcNT - srcNB
			     + srcST - srcSB
			     + srcET - srcEB
			     + srcWT - srcWB;

			ux /= rho;
			uy /= rho;
			uz /= rho;

			if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
				ux = 0.005f;
				uy = 0.002f;
				uz = 0.000f;
			}

			u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

			const float cuN  = uy;
			const float cuS  = uy;
			const float cuE  = ux;
			const float cuW  = ux;
			const float cuT  = uz;
			const float cuB  = uz;
			const float cuNE = ux + uy;
			const float cuNW = -ux + uy;
			const float cuSE = ux - uy;
			const float cuSW = -ux - uy;
			const float cuNT = uy + uz;
			const float cuNB = uy - uz;
			const float cuST = -uy + uz;
			const float cuSB = -uy - uz;
			const float cuET = ux + uz;
			const float cuEB = ux - uz;
			const float cuWT = -ux + uz;
			const float cuWB = -ux - uz;

			DST_C ( dstGrid ) = omega1m*srcC + dfl1o*rho*(1.0f          - u2);

			DST_N ( dstGrid ) = omega1m*srcN + dfl2o*rho*(1.0f + cuN *(4.5f*cuN  + 3.0f) - u2);
			DST_S ( dstGrid ) = omega1m*srcS + dfl2o*rho*(1.0f + cuS *(4.5f*cuS  - 3.0f) - u2);
			DST_E ( dstGrid ) = omega1m*srcE + dfl2o*rho*(1.0f + cuE *(4.5f*cuE  + 3.0f) - u2);
			DST_W ( dstGrid ) = omega1m*srcW + dfl2o*rho*(1.0f + cuW *(4.5f*cuW  - 3.0f) - u2);
			DST_T ( dstGrid ) = omega1m*srcT + dfl2o*rho*(1.0f + cuT *(4.5f*cuT  + 3.0f) - u2);
			DST_B ( dstGrid ) = omega1m*srcB + dfl2o*rho*(1.0f + cuB *(4.5f*cuB  - 3.0f) - u2);

			DST_NE( dstGrid ) = omega1m*srcNE + dfl3o*rho*(1.0f + cuNE*(4.5f*cuNE + 3.0f) - u2);
			DST_NW( dstGrid ) = omega1m*srcNW + dfl3o*rho*(1.0f + cuNW*(4.5f*cuNW + 3.0f) - u2);
			DST_SE( dstGrid ) = omega1m*srcSE + dfl3o*rho*(1.0f + cuSE*(4.5f*cuSE + 3.0f) - u2);
			DST_SW( dstGrid ) = omega1m*srcSW + dfl3o*rho*(1.0f + cuSW*(4.5f*cuSW + 3.0f) - u2);
			DST_NT( dstGrid ) = omega1m*srcNT + dfl3o*rho*(1.0f + cuNT*(4.5f*cuNT + 3.0f) - u2);
			DST_NB( dstGrid ) = omega1m*srcNB + dfl3o*rho*(1.0f + cuNB*(4.5f*cuNB + 3.0f) - u2);
			DST_ST( dstGrid ) = omega1m*srcST + dfl3o*rho*(1.0f + cuST*(4.5f*cuST + 3.0f) - u2);
			DST_SB( dstGrid ) = omega1m*srcSB + dfl3o*rho*(1.0f + cuSB*(4.5f*cuSB + 3.0f) - u2);
			DST_ET( dstGrid ) = omega1m*srcET + dfl3o*rho*(1.0f + cuET*(4.5f*cuET + 3.0f) - u2);
			DST_EB( dstGrid ) = omega1m*srcEB + dfl3o*rho*(1.0f + cuEB*(4.5f*cuEB + 3.0f) - u2);
			DST_WT( dstGrid ) = omega1m*srcWT + dfl3o*rho*(1.0f + cuWT*(4.5f*cuWT + 3.0f) - u2);
			DST_WB( dstGrid ) = omega1m*srcWB + dfl3o*rho*(1.0f + cuWB*(4.5f*cuWB + 3.0f) - u2);
		}
	SWEEP_END
}

void LBM_handleInOutFlow( LBM_Grid srcGrid ) {
	float ux , uy , uz , rho ,
	       ux1, uy1, uz1, rho1,
	       ux2, uy2, uz2, rho2,
	       u2, px, py;
	SWEEP_VAR

	const float dfl1 = DFL1;
	const float dfl2 = DFL2;
	const float dfl3 = DFL3;

	/* inflow */
	SWEEP_START( 0, 0, 0, 0, 0, 1 )
		rho1 =
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, N  ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, E  ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, T  ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NE ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SE ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NT ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ST ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ET ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WT ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WB );

		rho2 =
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, N  ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, E  ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, T  ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NE ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SE ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NT ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ST ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ET ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WT ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WB );

		rho = 2.0f * rho1 - rho2;

		px = (float)SWEEP_X / (0.5f*(float)(SIZE_X-1)) - 1.0f;
		py = (float)SWEEP_Y / (0.5f*(float)(SIZE_Y-1)) - 1.0f;
		ux = 0.00f;
		uy = 0.00f;
		uz = 0.01f * (1.0f-px*px) * (1.0f-py*py);

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = dfl1*rho*(1.0f          - u2);

		LOCAL( srcGrid, N ) = dfl2*rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2);
		LOCAL( srcGrid, S ) = dfl2*rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2);
		LOCAL( srcGrid, E ) = dfl2*rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2);
		LOCAL( srcGrid, W ) = dfl2*rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2);
		LOCAL( srcGrid, T ) = dfl2*rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2);
		LOCAL( srcGrid, B ) = dfl2*rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2);

		{
			const float cuNE = ux + uy;
			const float cuNW = -ux + uy;
			const float cuSE = ux - uy;
			const float cuSW = -ux - uy;
			const float cuNT = uy + uz;
			const float cuNB = uy - uz;
			const float cuST = -uy + uz;
			const float cuSB = -uy - uz;
			const float cuET = ux + uz;
			const float cuEB = ux - uz;
			const float cuWT = -ux + uz;
			const float cuWB = -ux - uz;

			LOCAL( srcGrid, NE) = dfl3*rho*(1.0f + cuNE*(4.5f*cuNE + 3.0f) - u2);
			LOCAL( srcGrid, NW) = dfl3*rho*(1.0f + cuNW*(4.5f*cuNW + 3.0f) - u2);
			LOCAL( srcGrid, SE) = dfl3*rho*(1.0f + cuSE*(4.5f*cuSE + 3.0f) - u2);
			LOCAL( srcGrid, SW) = dfl3*rho*(1.0f + cuSW*(4.5f*cuSW + 3.0f) - u2);
			LOCAL( srcGrid, NT) = dfl3*rho*(1.0f + cuNT*(4.5f*cuNT + 3.0f) - u2);
			LOCAL( srcGrid, NB) = dfl3*rho*(1.0f + cuNB*(4.5f*cuNB + 3.0f) - u2);
			LOCAL( srcGrid, ST) = dfl3*rho*(1.0f + cuST*(4.5f*cuST + 3.0f) - u2);
			LOCAL( srcGrid, SB) = dfl3*rho*(1.0f + cuSB*(4.5f*cuSB + 3.0f) - u2);
			LOCAL( srcGrid, ET) = dfl3*rho*(1.0f + cuET*(4.5f*cuET + 3.0f) - u2);
			LOCAL( srcGrid, EB) = dfl3*rho*(1.0f + cuEB*(4.5f*cuEB + 3.0f) - u2);
			LOCAL( srcGrid, WT) = dfl3*rho*(1.0f + cuWT*(4.5f*cuWT + 3.0f) - u2);
			LOCAL( srcGrid, WB) = dfl3*rho*(1.0f + cuWB*(4.5f*cuWB + 3.0f) - u2);
		}
	SWEEP_END

	/* outflow */
	SWEEP_START( 0, 0, SIZE_Z-1, 0, 0, SIZE_Z )
		rho1 =
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, N  ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, E  ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, T  ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );

		ux1 =
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, E  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, W  ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB ) -
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );

		uy1 =
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, N  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, S  ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW ) -
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB ) -
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB );

		uz1 =
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, T  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, B  ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );

		ux1 /= rho1;
		uy1 /= rho1;
		uz1 /= rho1;

		rho2 =
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, N  ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, E  ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, T  ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );

		ux2 =
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, E  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, W  ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB ) -
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );

		uy2 =
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, N  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, S  ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW ) -
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB ) -
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB );

		uz2 =
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, T  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, B  ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB ) +
			GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );

		ux2 /= rho2;
		uy2 /= rho2;
		uz2 /= rho2;

		rho = 1.0f;

		ux = 2.0f*ux1 - ux2;
		uy = 2.0f*uy1 - uy2;
		uz = 2.0f*uz1 - uz2;

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = dfl1*rho*(1.0f          - u2);

		LOCAL( srcGrid, N ) = dfl2*rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2);
		LOCAL( srcGrid, S ) = dfl2*rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2);
		LOCAL( srcGrid, E ) = dfl2*rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2);
		LOCAL( srcGrid, W ) = dfl2*rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2);
		LOCAL( srcGrid, T ) = dfl2*rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2);
		LOCAL( srcGrid, B ) = dfl2*rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2);

		{
			const float cuNE = ux + uy;
			const float cuNW = -ux + uy;
			const float cuSE = ux - uy;
			const float cuSW = -ux - uy;
			const float cuNT = uy + uz;
			const float cuNB = uy - uz;
			const float cuST = -uy + uz;
			const float cuSB = -uy - uz;
			const float cuET = ux + uz;
			const float cuEB = ux - uz;
			const float cuWT = -ux + uz;
			const float cuWB = -ux - uz;

			LOCAL( srcGrid, NE) = dfl3*rho*(1.0f + cuNE*(4.5f*cuNE + 3.0f) - u2);
			LOCAL( srcGrid, NW) = dfl3*rho*(1.0f + cuNW*(4.5f*cuNW + 3.0f) - u2);
			LOCAL( srcGrid, SE) = dfl3*rho*(1.0f + cuSE*(4.5f*cuSE + 3.0f) - u2);
			LOCAL( srcGrid, SW) = dfl3*rho*(1.0f + cuSW*(4.5f*cuSW + 3.0f) - u2);
			LOCAL( srcGrid, NT) = dfl3*rho*(1.0f + cuNT*(4.5f*cuNT + 3.0f) - u2);
			LOCAL( srcGrid, NB) = dfl3*rho*(1.0f + cuNB*(4.5f*cuNB + 3.0f) - u2);
			LOCAL( srcGrid, ST) = dfl3*rho*(1.0f + cuST*(4.5f*cuST + 3.0f) - u2);
			LOCAL( srcGrid, SB) = dfl3*rho*(1.0f + cuSB*(4.5f*cuSB + 3.0f) - u2);
			LOCAL( srcGrid, ET) = dfl3*rho*(1.0f + cuET*(4.5f*cuET + 3.0f) - u2);
			LOCAL( srcGrid, EB) = dfl3*rho*(1.0f + cuEB*(4.5f*cuEB + 3.0f) - u2);
			LOCAL( srcGrid, WT) = dfl3*rho*(1.0f + cuWT*(4.5f*cuWT + 3.0f) - u2);
			LOCAL( srcGrid, WB) = dfl3*rho*(1.0f + cuWB*(4.5f*cuWB + 3.0f) - u2);
		}
	SWEEP_END
}

/*############################################################################*/

void LBM_showGridStatistics( LBM_Grid grid ) {
	int nObstacleCells = 0,
	    nAccelCells    = 0,
	    nFluidCells    = 0;
	float ux, uy, uz;
	float minU2  = 1e+30f, maxU2  = -1e+30f, u2;
	float minRho = 1e+30f, maxRho = -1e+30f, rho;
	float mass = 0.0f;

	SWEEP_VAR

	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		rho = LOCAL( grid, C  ) + LOCAL( grid, N  ) +
		      LOCAL( grid, S  ) + LOCAL( grid, E  ) +
		      LOCAL( grid, W  ) + LOCAL( grid, T  ) +
		      LOCAL( grid, B  ) + LOCAL( grid, NE ) +
		      LOCAL( grid, NW ) + LOCAL( grid, SE ) +
		      LOCAL( grid, SW ) + LOCAL( grid, NT ) +
		      LOCAL( grid, NB ) + LOCAL( grid, ST ) +
		      LOCAL( grid, SB ) + LOCAL( grid, ET ) +
		      LOCAL( grid, EB ) + LOCAL( grid, WT ) +
		      LOCAL( grid, WB );

		if( rho < minRho ) minRho = rho;
		if( rho > maxRho ) maxRho = rho;
		mass += rho;

		if( TEST_FLAG_SWEEP( grid, OBSTACLE )) {
			++nObstacleCells;
		}
		else {
			if( TEST_FLAG_SWEEP( grid, ACCEL ))
				++nAccelCells;
			else
				++nFluidCells;

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

        printf( "LBM_showGridStatistics:\n"
        "\tnObstacleCells: %7i nAccelCells: %7i nFluidCells: %7i\n"
        "\tminRho: %8.4f maxRho: %8.4f mass: %e\n"
        "\tminU: %e maxU: %e\n\n",
        nObstacleCells, nAccelCells, nFluidCells,
        minRho, maxRho, mass,
        sqrtf( minU2 ), sqrtf( maxU2 ) );

}

/*############################################################################*/

static void storeValue( FILE* file, OUTPUT_PRECISION* v ) {
	const int litteBigEndianTest = 1;
	if( (*((const unsigned char*) &litteBigEndianTest)) == 0 ) {         /* big endian */
		const char* vPtr = (const char*) v;
		char buffer[sizeof( OUTPUT_PRECISION )];
		size_t i;

		for (i = 0; i < sizeof( OUTPUT_PRECISION ); i++)
			buffer[i] = vPtr[sizeof( OUTPUT_PRECISION ) - i - 1];

		(void)fwrite( buffer, sizeof( OUTPUT_PRECISION ), 1, file );
	}
	else {                                                     /* little endian */
		(void)fwrite( v, sizeof( OUTPUT_PRECISION ), 1, file );
	}
}

/*############################################################################*/

static void loadValue( FILE* file, OUTPUT_PRECISION* v ) {
	const int litteBigEndianTest = 1;
	if( (*((const unsigned char*) &litteBigEndianTest)) == 0 ) {         /* big endian */
		char* vPtr = (char*) v;
		char buffer[sizeof( OUTPUT_PRECISION )];
		size_t i;

		(void)fread( buffer, sizeof( OUTPUT_PRECISION ), 1, file );

		for (i = 0; i < sizeof( OUTPUT_PRECISION ); i++)
			vPtr[i] = buffer[sizeof( OUTPUT_PRECISION ) - i - 1];
	}
	else {                                                     /* little endian */
		(void)fread( v, sizeof( OUTPUT_PRECISION ), 1, file );
	}
}

void LBM_storeVelocityField( LBM_Grid grid, const char* filename,
                             const int binary ) {
	int x, y, z;
	OUTPUT_PRECISION rho, ux, uy, uz;

	FILE* file = fopen( filename, (binary ? "wb" : "w") );
	if (!file) {
		perror("LBM_storeVelocityField");
		exit(1);
	}

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				const float c  = GRID_ENTRY( grid, x, y, z, C  );
				const float n  = GRID_ENTRY( grid, x, y, z, N  );
				const float s  = GRID_ENTRY( grid, x, y, z, S  );
				const float e  = GRID_ENTRY( grid, x, y, z, E  );
				const float w  = GRID_ENTRY( grid, x, y, z, W  );
				const float t  = GRID_ENTRY( grid, x, y, z, T  );
				const float b  = GRID_ENTRY( grid, x, y, z, B  );
				const float ne = GRID_ENTRY( grid, x, y, z, NE );
				const float nw = GRID_ENTRY( grid, x, y, z, NW );
				const float se = GRID_ENTRY( grid, x, y, z, SE );
				const float sw = GRID_ENTRY( grid, x, y, z, SW );
				const float nt = GRID_ENTRY( grid, x, y, z, NT );
				const float nb = GRID_ENTRY( grid, x, y, z, NB );
				const float st = GRID_ENTRY( grid, x, y, z, ST );
				const float sb = GRID_ENTRY( grid, x, y, z, SB );
				const float et = GRID_ENTRY( grid, x, y, z, ET );
				const float eb = GRID_ENTRY( grid, x, y, z, EB );
				const float wt = GRID_ENTRY( grid, x, y, z, WT );
				const float wb = GRID_ENTRY( grid, x, y, z, WB );

				rho = c + n + s + e + w + t + b +
				      ne + nw + se + sw +
				      nt + nb + st + sb +
				      et + eb + wt + wb;

				ux = + e - w
				     + ne - nw
				     + se - sw
				     + et + eb
				     - wt - wb;
				uy = + n - s
				     + ne + nw
				     - se - sw
				     + nt + nb
				     - st - sb;
				uz = + t - b
				     + nt - nb
				     + st - sb
				     + et - eb
				     + wt - wb;
				ux /= rho;
				uy /= rho;
				uz /= rho;

				if( binary ) {
					storeValue( file, &ux );
					storeValue( file, &uy );
					storeValue( file, &uz );
				} else {
					fprintf( file, "%e %e %e\n",
					         (double)ux, (double)uy, (double)uz );
				}

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
	if (!file) {
		perror("LBM_compareVelocityField");
		exit(1);
	}

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				const float c  = GRID_ENTRY( grid, x, y, z, C  );
				const float n  = GRID_ENTRY( grid, x, y, z, N  );
				const float s  = GRID_ENTRY( grid, x, y, z, S  );
				const float e  = GRID_ENTRY( grid, x, y, z, E  );
				const float w  = GRID_ENTRY( grid, x, y, z, W  );
				const float t  = GRID_ENTRY( grid, x, y, z, T  );
				const float b  = GRID_ENTRY( grid, x, y, z, B  );
				const float ne = GRID_ENTRY( grid, x, y, z, NE );
				const float nw = GRID_ENTRY( grid, x, y, z, NW );
				const float se = GRID_ENTRY( grid, x, y, z, SE );
				const float sw = GRID_ENTRY( grid, x, y, z, SW );
				const float nt = GRID_ENTRY( grid, x, y, z, NT );
				const float nb = GRID_ENTRY( grid, x, y, z, NB );
				const float st = GRID_ENTRY( grid, x, y, z, ST );
				const float sb = GRID_ENTRY( grid, x, y, z, SB );
				const float et = GRID_ENTRY( grid, x, y, z, ET );
				const float eb = GRID_ENTRY( grid, x, y, z, EB );
				const float wt = GRID_ENTRY( grid, x, y, z, WT );
				const float wb = GRID_ENTRY( grid, x, y, z, WB );

				rho = c + n + s + e + w + t + b +
				      ne + nw + se + sw +
				      nt + nb + st + sb +
				      et + eb + wt + wb;

				ux = + e - w
				     + ne - nw
				     + se - sw
				     + et + eb
				     - wt - wb;
				uy = + n - s
				     + ne + nw
				     - se - sw
				     + nt + nb
				     - st - sb;
				uz = + t - b
				     + nt - nb
				     + st - sb
				     + et - eb
				     + wt - wb;
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
						fscanf( file, "%lf %lf %lf\n",
						        (double*)&fileUx,
						        (double*)&fileUy,
						        (double*)&fileUz );
					}
					else {
						fscanf( file, "%f %f %f\n",
						        (float*)&fileUx,
						        (float*)&fileUy,
						        (float*)&fileUz );
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

