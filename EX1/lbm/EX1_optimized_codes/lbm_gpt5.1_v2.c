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

	/*voption indep*/
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
		perror("LBM_loadObstacleFile: fopen failed");
		exit(1);
	}

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				int c = fgetc( file );
				if( c != '.' && c != EOF )
					SET_FLAG( grid, x, y, z, OBSTACLE );
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
		const int zIsFace = (z == 0 || z == SIZE_Z-1);
		const int zIsAccel = (z == 1 || z == SIZE_Z-2);
		for( y = 0; y < SIZE_Y; y++ ) {
			const int yIsBorder = (y == 0 || y == SIZE_Y-1);
			const int yInAccel  = (y > 1 && y < SIZE_Y-2);
			for( x = 0; x < SIZE_X; x++ ) {
				const int xIsBorder = (x == 0 || x == SIZE_X-1);
				if( xIsBorder || yIsBorder || zIsFace ) {
					SET_FLAG( grid, x, y, z, OBSTACLE );
				} else if( zIsAccel && x > 1 && x < SIZE_X-2 && yInAccel ) {
					SET_FLAG( grid, x, y, z, ACCEL );
				}
			}
		}
	}
}

void LBM_initializeSpecialCellsForChannel( LBM_Grid grid ) {
	int x,  y,  z;

	/*voption indep*/
	for( z = -2; z < SIZE_Z+2; z++ ) {
		const int zIsInOut = (z == 0 || z == SIZE_Z-1);
		for( y = 0; y < SIZE_Y; y++ ) {
			const int yIsBorder = (y == 0 || y == SIZE_Y-1);
			for( x = 0; x < SIZE_X; x++ ) {
				const int xIsBorder = (x == 0 || x == SIZE_X-1);
				if( xIsBorder || yIsBorder ) {
					SET_FLAG( grid, x, y, z, OBSTACLE );

					if( zIsInOut &&
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

	const float omega      = OMEGA;
	const float oneMinusOm = 1.0f - omega;
	const float dfl1Omega  = DFL1 * omega;
	const float dfl2Omega  = DFL2 * omega;
	const float dfl3Omega  = DFL3 * omega;

	/*voption indep*/
	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
			const float cC  = SRC_C ( srcGrid );
			const float cN  = SRC_N ( srcGrid );
			const float cS  = SRC_S ( srcGrid );
			const float cE  = SRC_E ( srcGrid );
			const float cW  = SRC_W ( srcGrid );
			const float cT  = SRC_T ( srcGrid );
			const float cB  = SRC_B ( srcGrid );
			const float cNE = SRC_NE( srcGrid );
			const float cNW = SRC_NW( srcGrid );
			const float cSE = SRC_SE( srcGrid );
			const float cSW = SRC_SW( srcGrid );
			const float cNT = SRC_NT( srcGrid );
			const float cNB = SRC_NB( srcGrid );
			const float cST = SRC_ST( srcGrid );
			const float cSB = SRC_SB( srcGrid );
			const float cET = SRC_ET( srcGrid );
			const float cEB = SRC_EB( srcGrid );
			const float cWT = SRC_WT( srcGrid );
			const float cWB = SRC_WB( srcGrid );

			DST_C ( dstGrid ) = cC;
			DST_S ( dstGrid ) = cN;
			DST_N ( dstGrid ) = cS;
			DST_W ( dstGrid ) = cE;
			DST_E ( dstGrid ) = cW;
			DST_B ( dstGrid ) = cT;
			DST_T ( dstGrid ) = cB;
			DST_SW( dstGrid ) = cNE;
			DST_SE( dstGrid ) = cNW;
			DST_NW( dstGrid ) = cSE;
			DST_NE( dstGrid ) = cSW;
			DST_SB( dstGrid ) = cNT;
			DST_ST( dstGrid ) = cNB;
			DST_NB( dstGrid ) = cST;
			DST_NT( dstGrid ) = cSB;
			DST_WB( dstGrid ) = cET;
			DST_WT( dstGrid ) = cEB;
			DST_EB( dstGrid ) = cWT;
			DST_ET( dstGrid ) = cWB;
			continue;
		}

		{
			const float cC  = SRC_C ( srcGrid );
			const float cN  = SRC_N ( srcGrid );
			const float cS  = SRC_S ( srcGrid );
			const float cE  = SRC_E ( srcGrid );
			const float cW  = SRC_W ( srcGrid );
			const float cT  = SRC_T ( srcGrid );
			const float cB  = SRC_B ( srcGrid );
			const float cNE = SRC_NE( srcGrid );
			const float cNW = SRC_NW( srcGrid );
			const float cSE = SRC_SE( srcGrid );
			const float cSW = SRC_SW( srcGrid );
			const float cNT = SRC_NT( srcGrid );
			const float cNB = SRC_NB( srcGrid );
			const float cST = SRC_ST( srcGrid );
			const float cSB = SRC_SB( srcGrid );
			const float cET = SRC_ET( srcGrid );
			const float cEB = SRC_EB( srcGrid );
			const float cWT = SRC_WT( srcGrid );
			const float cWB = SRC_WB( srcGrid );

			rho =  cC  + cN  + cS  + cE  + cW  + cT  + cB  +
			       cNE + cNW + cSE + cSW + cNT + cNB + cST +
			       cSB + cET + cEB + cWT + cWB;

			ux =  cE - cW +
			      cNE - cNW +
			      cSE - cSW +
			      cET + cEB -
			      cWT - cWB;
			uy =  cN - cS +
			      cNE + cNW -
			      cSE - cSW +
			      cNT + cNB -
			      cST - cSB;
			uz =  cT - cB +
			      cNT - cNB +
			      cST - cSB +
			      cET - cEB +
			      cWT - cWB;

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
				const float cu, cu2;

				DST_C ( dstGrid ) = oneMinusOm*cC +
				                    dfl1Omega*rho*(1.0f - u2);

				cu  = uy;
				cu2 = 4.5f*cu*cu;
				DST_N ( dstGrid ) = oneMinusOm*cN +
				                    dfl2Omega*rho*(1.0f + cu2 + 3.0f*cu - u2);
				DST_S ( dstGrid ) = oneMinusOm*cS +
				                    dfl2Omega*rho*(1.0f + cu2 - 3.0f*cu - u2);

				cu  = ux;
				cu2 = 4.5f*cu*cu;
				DST_E ( dstGrid ) = oneMinusOm*cE +
				                    dfl2Omega*rho*(1.0f + cu2 + 3.0f*cu - u2);
				DST_W ( dstGrid ) = oneMinusOm*cW +
				                    dfl2Omega*rho*(1.0f + cu2 - 3.0f*cu - u2);

				cu  = uz;
				cu2 = 4.5f*cu*cu;
				DST_T ( dstGrid ) = oneMinusOm*cT +
				                    dfl2Omega*rho*(1.0f + cu2 + 3.0f*cu - u2);
				DST_B ( dstGrid ) = oneMinusOm*cB +
				                    dfl2Omega*rho*(1.0f + cu2 - 3.0f*cu - u2);
			}

			{
				const float uxy_p = ux + uy;
				const float uxy_m = ux - uy;
				const float uyz_p = uy + uz;
				const float uyz_m = uy - uz;
				const float uxz_p = ux + uz;
				const float uxz_m = ux - uz;

				DST_NE( dstGrid ) = oneMinusOm*cNE +
				                    dfl3Omega*rho*(1.0f + (uxy_p)*(4.5f*(uxy_p) + 3.0f) - u2);
				DST_NW( dstGrid ) = oneMinusOm*cNW +
				                    dfl3Omega*rho*(1.0f + (-uxy_m)*(4.5f*(-uxy_m) + 3.0f) - u2);
				DST_SE( dstGrid ) = oneMinusOm*cSE +
				                    dfl3Omega*rho*(1.0f + (uxy_m)*(4.5f*(uxy_m) + 3.0f) - u2);
				DST_SW( dstGrid ) = oneMinusOm*cSW +
				                    dfl3Omega*rho*(1.0f + (-uxy_p)*(4.5f*(-uxy_p) + 3.0f) - u2);

				DST_NT( dstGrid ) = oneMinusOm*cNT +
				                    dfl3Omega*rho*(1.0f + (uyz_p)*(4.5f*(uyz_p) + 3.0f) - u2);
				DST_NB( dstGrid ) = oneMinusOm*cNB +
				                    dfl3Omega*rho*(1.0f + (uyz_m)*(4.5f*(uyz_m) + 3.0f) - u2);
				DST_ST( dstGrid ) = oneMinusOm*cST +
				                    dfl3Omega*rho*(1.0f + (-uyz_m)*(4.5f*(-uyz_m) + 3.0f) - u2);
				DST_SB( dstGrid ) = oneMinusOm*cSB +
				                    dfl3Omega*rho*(1.0f + (-uyz_p)*(4.5f*(-uyz_p) + 3.0f) - u2);

				DST_ET( dstGrid ) = oneMinusOm*cET +
				                    dfl3Omega*rho*(1.0f + (uxz_p)*(4.5f*(uxz_p) + 3.0f) - u2);
				DST_EB( dstGrid ) = oneMinusOm*cEB +
				                    dfl3Omega*rho*(1.0f + (uxz_m)*(4.5f*(uxz_m) + 3.0f) - u2);
				DST_WT( dstGrid ) = oneMinusOm*cWT +
				                    dfl3Omega*rho*(1.0f + (-uxz_m)*(4.5f*(-uxz_m) + 3.0f) - u2);
				DST_WB( dstGrid ) = oneMinusOm*cWB +
				                    dfl3Omega*rho*(1.0f + (-uxz_p)*(4.5f*(-uxz_p) + 3.0f) - u2);
			}
		}
	SWEEP_END
}

void LBM_handleInOutFlow( LBM_Grid srcGrid ) {
	float ux , uy , uz , rho ,
	       ux1, uy1, uz1, rho1,
	       ux2, uy2, uz2, rho2,
	       u2, px, py;
	SWEEP_VAR

	const float two = 2.0f;
	const float one = 1.0f;
	const float coeff = 0.01f;
	const float half = 0.5f;

	/* inflow */
	/*voption indep*/
	SWEEP_START( 0, 0, 0, 0, 0, 1 )
		{
			const float c1C  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, C  );
			const float c1N  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, N  );
			const float c1S  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, S  );
			const float c1E  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, E  );
			const float c1W  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, W  );
			const float c1T  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, T  );
			const float c1B  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, B  );
			const float c1NE = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NE );
			const float c1NW = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NW );
			const float c1SE = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SE );
			const float c1SW = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SW );
			const float c1NT = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NT );
			const float c1NB = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NB );
			const float c1ST = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ST );
			const float c1SB = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SB );
			const float c1ET = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ET );
			const float c1EB = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, EB );
			const float c1WT = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WT );
			const float c1WB = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WB );

			const float c2C  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, C  );
			const float c2N  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, N  );
			const float c2S  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, S  );
			const float c2E  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, E  );
			const float c2W  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, W  );
			const float c2T  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, T  );
			const float c2B  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, B  );
			const float c2NE = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NE );
			const float c2NW = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NW );
			const float c2SE = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SE );
			const float c2SW = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SW );
			const float c2NT = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NT );
			const float c2NB = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NB );
			const float c2ST = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ST );
			const float c2SB = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SB );
			const float c2ET = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ET );
			const float c2EB = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, EB );
			const float c2WT = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WT );
			const float c2WB = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WB );

			rho1 =  c1C  + c1N  + c1S  + c1E  + c1W  + c1T  + c1B  +
			        c1NE + c1NW + c1SE + c1SW + c1NT + c1NB + c1ST +
			        c1SB + c1ET + c1EB + c1WT + c1WB;
			rho2 =  c2C  + c2N  + c2S  + c2E  + c2W  + c2T  + c2B  +
			        c2NE + c2NW + c2SE + c2SW + c2NT + c2NB + c2ST +
			        c2SB + c2ET + c2EB + c2WT + c2WB;
		}

		rho = two*rho1 - rho2;

		px = (SWEEP_X / (half*(SIZE_X-1))) - one;
		py = (SWEEP_Y / (half*(SIZE_Y-1))) - one;
		ux = 0.00f;
		uy = 0.00f;
		uz = coeff * (one-px*px) * (one-py*py);

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = DFL1*rho*(1.0f - u2);

		LOCAL( srcGrid, N ) = DFL2*rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2);
		LOCAL( srcGrid, S ) = DFL2*rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2);
		LOCAL( srcGrid, E ) = DFL2*rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2);
		LOCAL( srcGrid, W ) = DFL2*rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2);
		LOCAL( srcGrid, T ) = DFL2*rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2);
		LOCAL( srcGrid, B ) = DFL2*rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2);

		LOCAL( srcGrid, NE) = DFL3*rho*(1.0f + (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
		LOCAL( srcGrid, NW) = DFL3*rho*(1.0f + (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
		LOCAL( srcGrid, SE) = DFL3*rho*(1.0f + (+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
		LOCAL( srcGrid, SW) = DFL3*rho*(1.0f + (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
		LOCAL( srcGrid, NT) = DFL3*rho*(1.0f + (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
		LOCAL( srcGrid, NB) = DFL3*rho*(1.0f + (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
		LOCAL( srcGrid, ST) = DFL3*rho*(1.0f + (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
		LOCAL( srcGrid, SB) = DFL3*rho*(1.0f + (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
		LOCAL( srcGrid, ET) = DFL3*rho*(1.0f + (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
		LOCAL( srcGrid, EB) = DFL3*rho*(1.0f + (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
		LOCAL( srcGrid, WT) = DFL3*rho*(1.0f + (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
		LOCAL( srcGrid, WB) = DFL3*rho*(1.0f + (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
	SWEEP_END

	/* outflow */
	/*voption indep*/
	SWEEP_START( 0, 0, SIZE_Z-1, 0, 0, SIZE_Z )
		{
			const float cm1C  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, C  );
			const float cm1N  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, N  );
			const float cm1S  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, S  );
			const float cm1E  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, E  );
			const float cm1W  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, W  );
			const float cm1T  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, T  );
			const float cm1B  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, B  );
			const float cm1NE = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE );
			const float cm1NW = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW );
			const float cm1SE = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE );
			const float cm1SW = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW );
			const float cm1NT = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT );
			const float cm1NB = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB );
			const float cm1ST = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST );
			const float cm1SB = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB );
			const float cm1ET = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET );
			const float cm1EB = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB );
			const float cm1WT = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT );
			const float cm1WB = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );

			rho1 =  cm1C  + cm1N  + cm1S  + cm1E  + cm1W  + cm1T  + cm1B  +
			        cm1NE + cm1NW + cm1SE + cm1SW + cm1NT + cm1NB + cm1ST +
			        cm1SB + cm1ET + cm1EB + cm1WT + cm1WB;

			ux1 =  cm1E - cm1W +
			       cm1NE - cm1NW +
			       cm1SE - cm1SW +
			       cm1ET + cm1EB -
			       cm1WT - cm1WB;
			uy1 =  cm1N - cm1S +
			       cm1NE + cm1NW -
			       cm1SE - cm1SW +
			       cm1NT + cm1NB -
			       cm1ST - cm1SB;
			uz1 =  cm1T - cm1B +
			       cm1NT - cm1NB +
			       cm1ST - cm1SB +
			       cm1ET - cm1EB +
			       cm1WT - cm1WB;

			ux1 /= rho1;
			uy1 /= rho1;
			uz1 /= rho1;
		}

		{
			const float cm2C  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, C  );
			const float cm2N  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, N  );
			const float cm2S  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, S  );
			const float cm2E  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, E  );
			const float cm2W  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, W  );
			const float cm2T  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, T  );
			const float cm2B  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, B  );
			const float cm2NE = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE );
			const float cm2NW = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW );
			const float cm2SE = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE );
			const float cm2SW = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW );
			const float cm2NT = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT );
			const float cm2NB = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB );
			const float cm2ST = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST );
			const float cm2SB = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB );
			const float cm2ET = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET );
			const float cm2EB = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB );
			const float cm2WT = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT );
			const float cm2WB = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );

			rho2 =  cm2C  + cm2N  + cm2S  + cm2E  + cm2W  + cm2T  + cm2B  +
			        cm2NE + cm2NW + cm2SE + cm2SW + cm2NT + cm2NB + cm2ST +
			        cm2SB + cm2ET + cm2EB + cm2WT + cm2WB;

			ux2 =  cm2E - cm2W +
			       cm2NE - cm2NW +
			       cm2SE - cm2SW +
			       cm2ET + cm2EB -
			       cm2WT - cm2WB;
			uy2 =  cm2N - cm2S +
			       cm2NE + cm2NW -
			       cm2SE - cm2SW +
			       cm2NT + cm2NB -
			       cm2ST - cm2SB;
			uz2 =  cm2T - cm2B +
			       cm2NT - cm2NB +
			       cm2ST - cm2SB +
			       cm2ET - cm2EB +
			       cm2WT - cm2WB;

			ux2 /= rho2;
			uy2 /= rho2;
			uz2 /= rho2;
		}

		rho = 1.0f;

		ux = two*ux1 - ux2;
		uy = two*uy1 - uy2;
		uz = two*uz1 - uz2;

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = DFL1*rho*(1.0f - u2);

		LOCAL( srcGrid, N ) = DFL2*rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2);
		LOCAL( srcGrid, S ) = DFL2*rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2);
		LOCAL( srcGrid, E ) = DFL2*rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2);
		LOCAL( srcGrid, W ) = DFL2*rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2);
		LOCAL( srcGrid, T ) = DFL2*rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2);
		LOCAL( srcGrid, B ) = DFL2*rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2);

		LOCAL( srcGrid, NE) = DFL3*rho*(1.0f + (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
		LOCAL( srcGrid, NW) = DFL3*rho*(1.0f + (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
		LOCAL( srcGrid, SE) = DFL3*rho*(1.0f + (+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
		LOCAL( srcGrid, SW) = DFL3*rho*(1.0f + (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
		LOCAL( srcGrid, NT) = DFL3*rho*(1.0f + (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
		LOCAL( srcGrid, NB) = DFL3*rho*(1.0f + (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
		LOCAL( srcGrid, ST) = DFL3*rho*(1.0f + (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
		LOCAL( srcGrid, SB) = DFL3*rho*(1.0f + (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
		LOCAL( srcGrid, ET) = DFL3*rho*(1.0f + (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
		LOCAL( srcGrid, EB) = DFL3*rho*(1.0f + (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
		LOCAL( srcGrid, WT) = DFL3*rho*(1.0f + (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
		LOCAL( srcGrid, WB) = DFL3*rho*(1.0f + (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
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
		{
			const float cC  = LOCAL( grid, C  );
			const float cN  = LOCAL( grid, N  );
			const float cS  = LOCAL( grid, S  );
			const float cE  = LOCAL( grid, E  );
			const float cW  = LOCAL( grid, W  );
			const float cT  = LOCAL( grid, T  );
			const float cB  = LOCAL( grid, B  );
			const float cNE = LOCAL( grid, NE );
			const float cNW = LOCAL( grid, NW );
			const float cSE = LOCAL( grid, SE );
			const float cSW = LOCAL( grid, SW );
			const float cNT = LOCAL( grid, NT );
			const float cNB = LOCAL( grid, NB );
			const float cST = LOCAL( grid, ST );
			const float cSB = LOCAL( grid, SB );
			const float cET = LOCAL( grid, ET );
			const float cEB = LOCAL( grid, EB );
			const float cWT = LOCAL( grid, WT );
			const float cWB = LOCAL( grid, WB );

			rho =  cC  + cN  + cS  + cE  + cW  + cT  + cB  +
			       cNE + cNW + cSE + cSW + cNT + cNB + cST +
			       cSB + cET + cEB + cWT + cWB;

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

				ux =  cE - cW +
				      cNE - cNW +
				      cSE - cSW +
				      cET + cEB -
				      cWT - cWB;
				uy =  cN - cS +
				      cNE + cNW -
				      cSE - cSW +
				      cNT + cNB -
				      cST - cSB;
				uz =  cT - cB +
				      cNT - cNB +
				      cST - cSB +
				      cET - cEB +
				      cWT - cWB;
				u2 = (ux*ux + uy*uy + uz*uz) / (rho*rho);
				if( u2 < minU2 ) minU2 = u2;
				if( u2 > maxU2 ) maxU2 = u2;
			}
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

		fwrite( buffer, sizeof( OUTPUT_PRECISION ), 1, file );
	}
	else {                                                     /* little endian */
		fwrite( v, sizeof( OUTPUT_PRECISION ), 1, file );
	}
}

/*############################################################################*/

static void loadValue( FILE* file, OUTPUT_PRECISION* v ) {
	const int litteBigEndianTest = 1;
	if( (*((const unsigned char*) &litteBigEndianTest)) == 0 ) {         /* big endian */
		char* vPtr = (char*) v;
		char buffer[sizeof( OUTPUT_PRECISION )];
		size_t i;

		fread( buffer, sizeof( OUTPUT_PRECISION ), 1, file );

		for (i = 0; i < sizeof( OUTPUT_PRECISION ); i++)
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
	if (!file) {
		perror("LBM_storeVelocityField: fopen failed");
		exit(1);
	}

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				const float cC  = GRID_ENTRY( grid, x, y, z, C  );
				const float cN  = GRID_ENTRY( grid, x, y, z, N  );
				const float cS  = GRID_ENTRY( grid, x, y, z, S  );
				const float cE  = GRID_ENTRY( grid, x, y, z, E  );
				const float cW  = GRID_ENTRY( grid, x, y, z, W  );
				const float cT  = GRID_ENTRY( grid, x, y, z, T  );
				const float cB  = GRID_ENTRY( grid, x, y, z, B  );
				const float cNE = GRID_ENTRY( grid, x, y, z, NE );
				const float cNW = GRID_ENTRY( grid, x, y, z, NW );
				const float cSE = GRID_ENTRY( grid, x, y, z, SE );
				const float cSW = GRID_ENTRY( grid, x, y, z, SW );
				const float cNT = GRID_ENTRY( grid, x, y, z, NT );
				const float cNB = GRID_ENTRY( grid, x, y, z, NB );
				const float cST = GRID_ENTRY( grid, x, y, z, ST );
				const float cSB = GRID_ENTRY( grid, x, y, z, SB );
				const float cET = GRID_ENTRY( grid, x, y, z, ET );
				const float cEB = GRID_ENTRY( grid, x, y, z, EB );
				const float cWT = GRID_ENTRY( grid, x, y, z, WT );
				const float cWB = GRID_ENTRY( grid, x, y, z, WB );

				rho =  cC  + cN  + cS  + cE  + cW  + cT  + cB  +
				       cNE + cNW + cSE + cSW + cNT + cNB + cST +
				       cSB + cET + cEB + cWT + cWB;
				ux =  cE - cW +
				      cNE - cNW +
				      cSE - cSW +
				      cET + cEB -
				      cWT - cWB;
				uy =  cN - cS +
				      cNE + cNW -
				      cSE - cSW +
				      cNT + cNB -
				      cST - cSB;
				uz =  cT - cB +
				      cNT - cNB +
				      cST - cSB +
				      cET - cEB +
				      cWT - cWB;
				ux /= rho;
				uy /= rho;
				uz /= rho;

				if( binary ) {
					storeValue( file, &ux );
					storeValue( file, &uy );
					storeValue( file, &uz );
				} else
					fprintf( file, "%e %e %e\n", (double)ux, (double)uy, (double)uz );

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
		perror("LBM_compareVelocityField: fopen failed");
		exit(1);
	}

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				const float cC  = GRID_ENTRY( grid, x, y, z, C  );
				const float cN  = GRID_ENTRY( grid, x, y, z, N  );
				const float cS  = GRID_ENTRY( grid, x, y, z, S  );
				const float cE  = GRID_ENTRY( grid, x, y, z, E  );
				const float cW  = GRID_ENTRY( grid, x, y, z, W  );
				const float cT  = GRID_ENTRY( grid, x, y, z, T  );
				const float cB  = GRID_ENTRY( grid, x, y, z, B  );
				const float cNE = GRID_ENTRY( grid, x, y, z, NE );
				const float cNW = GRID_ENTRY( grid, x, y, z, NW );
				const float cSE = GRID_ENTRY( grid, x, y, z, SE );
				const float cSW = GRID_ENTRY( grid, x, y, z, SW );
				const float cNT = GRID_ENTRY( grid, x, y, z, NT );
				const float cNB = GRID_ENTRY( grid, x, y, z, NB );
				const float cST = GRID_ENTRY( grid, x, y, z, ST );
				const float cSB = GRID_ENTRY( grid, x, y, z, SB );
				const float cET = GRID_ENTRY( grid, x, y, z, ET );
				const float cEB = GRID_ENTRY( grid, x, y, z, EB );
				const float cWT = GRID_ENTRY( grid, x, y, z, WT );
				const float cWB = GRID_ENTRY( grid, x, y, z, WB );

				rho =  cC  + cN  + cS  + cE  + cW  + cT  + cB  +
				       cNE + cNW + cSE + cSW + cNT + cNB + cST +
				       cSB + cET + cEB + cWT + cWB;
				ux =  cE - cW +
				      cNE - cNW +
				      cSE - cSW +
				      cET + cEB -
				      cWT - cWB;
				uy =  cN - cS +
				      cNE + cNW -
				      cSE - cSW +
				      cNT + cNB -
				      cST - cSB;
				uz =  cT - cB +
				      cNT - cNB +
				      cST - cSB +
				      cET - cEB +
				      cWT - cWB;
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

				dUx = (OUTPUT_PRECISION)ux - fileUx;
				dUy = (OUTPUT_PRECISION)uy - fileUy;
				dUz = (OUTPUT_PRECISION)uz - fileUz;
				diff2 = dUx*dUx + dUy*dUy + dUz*dUz;
				if( diff2 > maxDiff2 ) maxDiff2 = diff2;
			}
		}
	}

#if defined(SPEC_CPU)
	printf( "LBM_compareVelocityField: maxDiff = %e  \n\n",
	        sqrt( (double)maxDiff2 )  );
#else
	printf( "LBM_compareVelocityField: maxDiff = %e  ==>  %s\n\n",
	        sqrt( (double)maxDiff2 ),
	        sqrt( (double)maxDiff2 ) > 1e-5 ? "##### ERROR #####" : "OK" );
#endif
	fclose( file );
}

