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
	const size_t plane = (size_t)SIZE_X * (size_t)SIZE_Y * (size_t)N_CELL_ENTRIES;
	const size_t margin = 2 * plane;
	const size_t size   = sizeof( LBM_Grid ) + 2 * margin * sizeof( float );

	*ptr = (float *)malloc( size );
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
	const size_t plane = (size_t)SIZE_X * (size_t)SIZE_Y * (size_t)N_CELL_ENTRIES;
	const size_t margin = 2 * plane;

	free( *ptr - margin );
	*ptr = NULL;
}

void LBM_initializeGrid( LBM_Grid grid ) {
	SWEEP_VAR

	const float c_val  = (float)DFL1;
	const float ns_val = (float)DFL2;
	const float ew_val = (float)DFL2;
	const float tb_val = (float)DFL2;
	const float diag_val = (float)DFL3;

	SWEEP_START( 0, 0, -2, 0, 0, SIZE_Z+2 )
		LOCAL( grid, C  ) = c_val;
		LOCAL( grid, N  ) = ns_val;
		LOCAL( grid, S  ) = ns_val;
		LOCAL( grid, E  ) = ew_val;
		LOCAL( grid, W  ) = ew_val;
		LOCAL( grid, T  ) = tb_val;
		LOCAL( grid, B  ) = tb_val;
		LOCAL( grid, NE ) = diag_val;
		LOCAL( grid, NW ) = diag_val;
		LOCAL( grid, SE ) = diag_val;
		LOCAL( grid, SW ) = diag_val;
		LOCAL( grid, NT ) = diag_val;
		LOCAL( grid, NB ) = diag_val;
		LOCAL( grid, ST ) = diag_val;
		LOCAL( grid, SB ) = diag_val;
		LOCAL( grid, ET ) = diag_val;
		LOCAL( grid, EB ) = diag_val;
		LOCAL( grid, WT ) = diag_val;
		LOCAL( grid, WB ) = diag_val;

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
				const int c = fgetc( file );
				if( c != '.' && c != EOF ) {
					SET_FLAG( grid, x, y, z, OBSTACLE );
				}
			}
			fgetc( file );
		}
		fgetc( file );
	}

	fclose( file );
}

void LBM_initializeSpecialCellsForLDC( LBM_Grid grid ) {
	int x,  y,  z;

	const int maxX = SIZE_X - 1;
	const int maxY = SIZE_Y - 1;
	const int maxZ = SIZE_Z - 1;
	const int innerMaxX = SIZE_X - 2;
	const int innerMaxY = SIZE_Y - 2;

	for( z = -2; z < SIZE_Z+2; z++ ) {
		const int isZBorder = (z == 0 || z == maxZ);
		const int isZAccel  = (z == 1 || z == SIZE_Z-2);
		for( y = 0; y < SIZE_Y; y++ ) {
			const int isYBorder = (y == 0 || y == maxY);
			const int yInner = (y > 1 && y < innerMaxY);
			for( x = 0; x < SIZE_X; x++ ) {
				const int isXBorder = (x == 0 || x == maxX);
				if( isXBorder || isYBorder || isZBorder ) {
					SET_FLAG( grid, x, y, z, OBSTACLE );
				}
				else if( isZAccel &&
				         x > 1 && x < innerMaxX &&
				         yInner ) {
					SET_FLAG( grid, x, y, z, ACCEL );
				}
			}
		}
	}
}

void LBM_initializeSpecialCellsForChannel( LBM_Grid grid ) {
	int x,  y,  z;

	const int maxX = SIZE_X - 1;
	const int maxY = SIZE_Y - 1;
	const int maxZ = SIZE_Z - 1;

	for( z = -2; z < SIZE_Z+2; z++ ) {
		const int isZIO = (z == 0 || z == maxZ);
		for( y = 0; y < SIZE_Y; y++ ) {
			const int isYBorder = (y == 0 || y == maxY);
			for( x = 0; x < SIZE_X; x++ ) {
				const int isXBorder = (x == 0 || x == maxX);
				if( isXBorder || isYBorder ) {
					SET_FLAG( grid, x, y, z, OBSTACLE );

					if( isZIO &&
					    ! TEST_FLAG( grid, x, y, z, OBSTACLE ))
						SET_FLAG( grid, x, y, z, IN_OUT_FLOW );
				}
			}
		}
	}
}

void LBM_performStreamCollide( LBM_Grid srcGrid, LBM_Grid dstGrid ) {
	SWEEP_VAR

	const float omega  = OMEGA;
	const float omega_1 = 1.0f - omega;
	const float dfl1_omega = (float)DFL1 * omega;
	const float dfl2_omega = (float)DFL2 * omega;
	const float dfl3_omega = (float)DFL3 * omega;

	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
			const float c  = SRC_C ( srcGrid );
			const float n  = SRC_N ( srcGrid );
			const float s  = SRC_S ( srcGrid );
			const float e  = SRC_E ( srcGrid );
			const float w  = SRC_W ( srcGrid );
			const float t  = SRC_T ( srcGrid );
			const float b  = SRC_B ( srcGrid );
			const float ne = SRC_NE( srcGrid );
			const float nw = SRC_NW( srcGrid );
			const float se = SRC_SE( srcGrid );
			const float sw = SRC_SW( srcGrid );
			const float nt = SRC_NT( srcGrid );
			const float nb = SRC_NB( srcGrid );
			const float st = SRC_ST( srcGrid );
			const float sb = SRC_SB( srcGrid );
			const float et = SRC_ET( srcGrid );
			const float eb = SRC_EB( srcGrid );
			const float wt = SRC_WT( srcGrid );
			const float wb = SRC_WB( srcGrid );

			DST_C ( dstGrid ) = c;
			DST_S ( dstGrid ) = n;
			DST_N ( dstGrid ) = s;
			DST_W ( dstGrid ) = e;
			DST_E ( dstGrid ) = w;
			DST_B ( dstGrid ) = t;
			DST_T ( dstGrid ) = b;
			DST_SW( dstGrid ) = ne;
			DST_SE( dstGrid ) = nw;
			DST_NW( dstGrid ) = se;
			DST_NE( dstGrid ) = sw;
			DST_SB( dstGrid ) = nt;
			DST_ST( dstGrid ) = nb;
			DST_NB( dstGrid ) = st;
			DST_NT( dstGrid ) = sb;
			DST_WB( dstGrid ) = et;
			DST_WT( dstGrid ) = eb;
			DST_EB( dstGrid ) = wt;
			DST_ET( dstGrid ) = wb;
			continue;
		}

		const float c  = SRC_C ( srcGrid );
		const float n  = SRC_N ( srcGrid );
		const float s  = SRC_S ( srcGrid );
		const float e  = SRC_E ( srcGrid );
		const float w  = SRC_W ( srcGrid );
		const float t  = SRC_T ( srcGrid );
		const float b  = SRC_B ( srcGrid );
		const float ne = SRC_NE( srcGrid );
		const float nw = SRC_NW( srcGrid );
		const float se = SRC_SE( srcGrid );
		const float sw = SRC_SW( srcGrid );
		const float nt = SRC_NT( srcGrid );
		const float nb = SRC_NB( srcGrid );
		const float st = SRC_ST( srcGrid );
		const float sb = SRC_SB( srcGrid );
		const float et = SRC_ET( srcGrid );
		const float eb = SRC_EB( srcGrid );
		const float wt = SRC_WT( srcGrid );
		const float wb = SRC_WB( srcGrid );

		const float rho = c + n + s + e + w + t + b +
		                  ne + nw + se + sw +
		                  nt + nb + st + sb +
		                  et + eb + wt + wb;

		float ux =  e - w +
		            ne - nw +
		            se - sw +
		            et + eb -
		            wt - wb;
		float uy =  n - s +
		            ne + nw -
		            se - sw +
		            nt + nb -
		            st - sb;
		float uz =  t - b +
		            nt - nb +
		            st - sb +
		            et - eb +
		            wt - wb;

		const float invRho = 1.0f / rho;
		ux *= invRho;
		uy *= invRho;
		uz *= invRho;

		if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
			ux = 0.005f;
			uy = 0.002f;
			uz = 0.000f;
		}

		const float ux2 = ux * ux;
		const float uy2 = uy * uy;
		const float uz2 = uz * uz;
		const float u2  = 1.5f * (ux2 + uy2 + uz2);

		DST_C ( dstGrid ) = omega_1 * c + dfl1_omega * rho * (1.0f - u2);

		DST_N ( dstGrid ) = omega_1 * n + dfl2_omega * rho * (1.0f + uy * (4.5f * uy + 3.0f) - u2);
		DST_S ( dstGrid ) = omega_1 * s + dfl2_omega * rho * (1.0f + uy * (4.5f * uy - 3.0f) - u2);
		DST_E ( dstGrid ) = omega_1 * e + dfl2_omega * rho * (1.0f + ux * (4.5f * ux + 3.0f) - u2);
		DST_W ( dstGrid ) = omega_1 * w + dfl2_omega * rho * (1.0f + ux * (4.5f * ux - 3.0f) - u2);
		DST_T ( dstGrid ) = omega_1 * t + dfl2_omega * rho * (1.0f + uz * (4.5f * uz + 3.0f) - u2);
		DST_B ( dstGrid ) = omega_1 * b + dfl2_omega * rho * (1.0f + uz * (4.5f * uz - 3.0f) - u2);

		const float uxpuy = ux + uy;
		const float uxmuy = ux - uy;
		const float mxpuy = -ux + uy;
		const float mxmuy = -ux - uy;
		const float uypuz = uy + uz;
		const float uymuz = uy - uz;
		const float mypuz = -uy + uz;
		const float mymuz = -uy - uz;
		const float uxpuz = ux + uz;
		const float uxmuz = ux - uz;
		const float mxpuz = -ux + uz;
		const float mxmuz = -ux - uz;

		DST_NE( dstGrid ) = omega_1 * ne + dfl3_omega * rho * (1.0f + uxpuy * (4.5f * uxpuy + 3.0f) - u2);
		DST_NW( dstGrid ) = omega_1 * nw + dfl3_omega * rho * (1.0f + mxpuy * (4.5f * mxpuy + 3.0f) - u2);
		DST_SE( dstGrid ) = omega_1 * se + dfl3_omega * rho * (1.0f + uxmuy * (4.5f * uxmuy + 3.0f) - u2);
		DST_SW( dstGrid ) = omega_1 * sw + dfl3_omega * rho * (1.0f + mxmuy * (4.5f * mxmuy + 3.0f) - u2);
		DST_NT( dstGrid ) = omega_1 * nt + dfl3_omega * rho * (1.0f + uypuz * (4.5f * uypuz + 3.0f) - u2);
		DST_NB( dstGrid ) = omega_1 * nb + dfl3_omega * rho * (1.0f + uymuz * (4.5f * uymuz + 3.0f) - u2);
		DST_ST( dstGrid ) = omega_1 * st + dfl3_omega * rho * (1.0f + mypuz * (4.5f * mypuz + 3.0f) - u2);
		DST_SB( dstGrid ) = omega_1 * sb + dfl3_omega * rho * (1.0f + mymuz * (4.5f * mymuz + 3.0f) - u2);
		DST_ET( dstGrid ) = omega_1 * et + dfl3_omega * rho * (1.0f + uxpuz * (4.5f * uxpuz + 3.0f) - u2);
		DST_EB( dstGrid ) = omega_1 * eb + dfl3_omega * rho * (1.0f + uxmuz * (4.5f * uxmuz + 3.0f) - u2);
		DST_WT( dstGrid ) = omega_1 * wt + dfl3_omega * rho * (1.0f + mxpuz * (4.5f * mxpuz + 3.0f) - u2);
		DST_WB( dstGrid ) = omega_1 * wb + dfl3_omega * rho * (1.0f + mxmuz * (4.5f * mxmuz + 3.0f) - u2);
	SWEEP_END
}

void LBM_handleInOutFlow( LBM_Grid srcGrid ) {
	float ux , uy , uz , rho ,
	       ux1, uy1, uz1, rho1,
	       ux2, uy2, uz2, rho2,
	       u2, px, py;
	SWEEP_VAR

	const float dfl1 = (float)DFL1;
	const float dfl2 = (float)DFL2;
	const float dfl3 = (float)DFL3;

	/* inflow */
	SWEEP_START( 0, 0, 0, 0, 0, 1 )
		{
			const float c1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, C  );
			const float n1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, N  );
			const float s1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, S  );
			const float e1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, E  );
			const float w1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, W  );
			const float t1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, T  );
			const float b1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, B  );
			const float ne1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NE );
			const float nw1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NW );
			const float se1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SE );
			const float sw1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SW );
			const float nt1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NT );
			const float nb1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NB );
			const float st1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ST );
			const float sb1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SB );
			const float et1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ET );
			const float eb1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, EB );
			const float wt1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WT );
			const float wb1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WB );

			const float c2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, C  );
			const float n2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, N  );
			const float s2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, S  );
			const float e2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, E  );
			const float w2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, W  );
			const float t2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, T  );
			const float b2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, B  );
			const float ne2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NE );
			const float nw2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NW );
			const float se2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SE );
			const float sw2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SW );
			const float nt2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NT );
			const float nb2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NB );
			const float st2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ST );
			const float sb2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SB );
			const float et2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ET );
			const float eb2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, EB );
			const float wt2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WT );
			const float wb2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WB );

			rho1 = c1 + n1 + s1 + e1 + w1 + t1 + b1 +
			       ne1 + nw1 + se1 + sw1 +
			       nt1 + nb1 + st1 + sb1 +
			       et1 + eb1 + wt1 + wb1;
			rho2 = c2 + n2 + s2 + e2 + w2 + t2 + b2 +
			       ne2 + nw2 + se2 + sw2 +
			       nt2 + nb2 + st2 + sb2 +
			       et2 + eb2 + wt2 + wb2;
		}

		rho = 2.0f * rho1 - rho2;

		px = (float)SWEEP_X / (0.5f * (float)(SIZE_X - 1)) - 1.0f;
		py = (float)SWEEP_Y / (0.5f * (float)(SIZE_Y - 1)) - 1.0f;
		ux = 0.00f;
		uy = 0.00f;
		uz = 0.01f * (1.0f - px*px) * (1.0f - py*py);

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = dfl1 * rho * (1.0f - u2);

		LOCAL( srcGrid, N ) = dfl2 * rho * (1.0f + uy * (4.5f * uy + 3.0f) - u2);
		LOCAL( srcGrid, S ) = dfl2 * rho * (1.0f + uy * (4.5f * uy - 3.0f) - u2);
		LOCAL( srcGrid, E ) = dfl2 * rho * (1.0f + ux * (4.5f * ux + 3.0f) - u2);
		LOCAL( srcGrid, W ) = dfl2 * rho * (1.0f + ux * (4.5f * ux - 3.0f) - u2);
		LOCAL( srcGrid, T ) = dfl2 * rho * (1.0f + uz * (4.5f * uz + 3.0f) - u2);
		LOCAL( srcGrid, B ) = dfl2 * rho * (1.0f + uz * (4.5f * uz - 3.0f) - u2);

		{
			const float uxpuy = ux + uy;
			const float uxmuy = ux - uy;
			const float mxpuy = -ux + uy;
			const float mxmuy = -ux - uy;
			const float uypuz = uy + uz;
			const float uymuz = uy - uz;
			const float mypuz = -uy + uz;
			const float mymuz = -uy - uz;
			const float uxpuz = ux + uz;
			const float uxmuz = ux - uz;
			const float mxpuz = -ux + uz;
			const float mxmuz = -ux - uz;

			LOCAL( srcGrid, NE) = dfl3 * rho * (1.0f + uxpuy * (4.5f * uxpuy + 3.0f) - u2);
			LOCAL( srcGrid, NW) = dfl3 * rho * (1.0f + mxpuy * (4.5f * mxpuy + 3.0f) - u2);
			LOCAL( srcGrid, SE) = dfl3 * rho * (1.0f + uxmuy * (4.5f * uxmuy + 3.0f) - u2);
			LOCAL( srcGrid, SW) = dfl3 * rho * (1.0f + mxmuy * (4.5f * mxmuy + 3.0f) - u2);
			LOCAL( srcGrid, NT) = dfl3 * rho * (1.0f + uypuz * (4.5f * uypuz + 3.0f) - u2);
			LOCAL( srcGrid, NB) = dfl3 * rho * (1.0f + uymuz * (4.5f * uymuz + 3.0f) - u2);
			LOCAL( srcGrid, ST) = dfl3 * rho * (1.0f + mypuz * (4.5f * mypuz + 3.0f) - u2);
			LOCAL( srcGrid, SB) = dfl3 * rho * (1.0f + mymuz * (4.5f * mymuz + 3.0f) - u2);
			LOCAL( srcGrid, ET) = dfl3 * rho * (1.0f + uxpuz * (4.5f * uxpuz + 3.0f) - u2);
			LOCAL( srcGrid, EB) = dfl3 * rho * (1.0f + uxmuz * (4.5f * uxmuz + 3.0f) - u2);
			LOCAL( srcGrid, WT) = dfl3 * rho * (1.0f + mxpuz * (4.5f * mxpuz + 3.0f) - u2);
			LOCAL( srcGrid, WB) = dfl3 * rho * (1.0f + mxmuz * (4.5f * mxmuz + 3.0f) - u2);
		}
	SWEEP_END

	/* outflow */
	SWEEP_START( 0, 0, SIZE_Z-1, 0, 0, SIZE_Z )
		{
			const float c1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, C  );
			const float n1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, N  );
			const float s1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, S  );
			const float e1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, E  );
			const float w1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, W  );
			const float t1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, T  );
			const float b1  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, B  );
			const float ne1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE );
			const float nw1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW );
			const float se1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE );
			const float sw1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW );
			const float nt1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT );
			const float nb1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB );
			const float st1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST );
			const float sb1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB );
			const float et1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET );
			const float eb1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB );
			const float wt1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT );
			const float wb1 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );

			rho1 = c1 + n1 + s1 + e1 + w1 + t1 + b1 +
			       ne1 + nw1 + se1 + sw1 +
			       nt1 + nb1 + st1 + sb1 +
			       et1 + eb1 + wt1 + wb1;

			ux1 =  e1 - w1 +
			       ne1 - nw1 +
			       se1 - sw1 +
			       et1 + eb1 -
			       wt1 - wb1;
			uy1 =  n1 - s1 +
			       ne1 + nw1 -
			       se1 - sw1 +
			       nt1 + nb1 -
			       st1 - sb1;
			uz1 =  t1 - b1 +
			       nt1 - nb1 +
			       st1 - sb1 +
			       et1 - eb1 +
			       wt1 - wb1;

			const float invRho1 = 1.0f / rho1;
			ux1 *= invRho1;
			uy1 *= invRho1;
			uz1 *= invRho1;
		}

		{
			const float c2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, C  );
			const float n2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, N  );
			const float s2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, S  );
			const float e2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, E  );
			const float w2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, W  );
			const float t2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, T  );
			const float b2  = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, B  );
			const float ne2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE );
			const float nw2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW );
			const float se2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE );
			const float sw2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW );
			const float nt2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT );
			const float nb2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB );
			const float st2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST );
			const float sb2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB );
			const float et2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET );
			const float eb2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB );
			const float wt2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT );
			const float wb2 = GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );

			rho2 = c2 + n2 + s2 + e2 + w2 + t2 + b2 +
			       ne2 + nw2 + se2 + sw2 +
			       nt2 + nb2 + st2 + sb2 +
			       et2 + eb2 + wt2 + wb2;

			ux2 =  e2 - w2 +
			       ne2 - nw2 +
			       se2 - sw2 +
			       et2 + eb2 -
			       wt2 - wb2;
			uy2 =  n2 - s2 +
			       ne2 + nw2 -
			       se2 - sw2 +
			       nt2 + nb2 -
			       st2 - sb2;
			uz2 =  t2 - b2 +
			       nt2 - nb2 +
			       st2 - sb2 +
			       et2 - eb2 +
			       wt2 - wb2;

			const float invRho2 = 1.0f / rho2;
			ux2 *= invRho2;
			uy2 *= invRho2;
			uz2 *= invRho2;
		}

		rho = 1.0f;

		ux = 2.0f * ux1 - ux2;
		uy = 2.0f * uy1 - uy2;
		uz = 2.0f * uz1 - uz2;

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = dfl1 * rho * (1.0f - u2);

		LOCAL( srcGrid, N ) = dfl2 * rho * (1.0f + uy * (4.5f * uy + 3.0f) - u2);
		LOCAL( srcGrid, S ) = dfl2 * rho * (1.0f + uy * (4.5f * uy - 3.0f) - u2);
		LOCAL( srcGrid, E ) = dfl2 * rho * (1.0f + ux * (4.5f * ux + 3.0f) - u2);
		LOCAL( srcGrid, W ) = dfl2 * rho * (1.0f + ux * (4.5f * ux - 3.0f) - u2);
		LOCAL( srcGrid, T ) = dfl2 * rho * (1.0f + uz * (4.5f * uz + 3.0f) - u2);
		LOCAL( srcGrid, B ) = dfl2 * rho * (1.0f + uz * (4.5f * uz - 3.0f) - u2);

		{
			const float uxpuy = ux + uy;
			const float uxmuy = ux - uy;
			const float mxpuy = -ux + uy;
			const float mxmuy = -ux - uy;
			const float uypuz = uy + uz;
			const float uymuz = uy - uz;
			const float mypuz = -uy + uz;
			const float mymuz = -uy - uz;
			const float uxpuz = ux + uz;
			const float uxmuz = ux - uz;
			const float mxpuz = -ux + uz;
			const float mxmuz = -ux - uz;

			LOCAL( srcGrid, NE) = dfl3 * rho * (1.0f + uxpuy * (4.5f * uxpuy + 3.0f) - u2);
			LOCAL( srcGrid, NW) = dfl3 * rho * (1.0f + mxpuy * (4.5f * mxpuy + 3.0f) - u2);
			LOCAL( srcGrid, SE) = dfl3 * rho * (1.0f + uxmuy * (4.5f * uxmuy + 3.0f) - u2);
			LOCAL( srcGrid, SW) = dfl3 * rho * (1.0f + mxmuy * (4.5f * mxmuy + 3.0f) - u2);
			LOCAL( srcGrid, NT) = dfl3 * rho * (1.0f + uypuz * (4.5f * uypuz + 3.0f) - u2);
			LOCAL( srcGrid, NB) = dfl3 * rho * (1.0f + uymuz * (4.5f * uymuz + 3.0f) - u2);
			LOCAL( srcGrid, ST) = dfl3 * rho * (1.0f + mypuz * (4.5f * mypuz + 3.0f) - u2);
			LOCAL( srcGrid, SB) = dfl3 * rho * (1.0f + mymuz * (4.5f * mymuz + 3.0f) - u2);
			LOCAL( srcGrid, ET) = dfl3 * rho * (1.0f + uxpuz * (4.5f * uxpuz + 3.0f) - u2);
			LOCAL( srcGrid, EB) = dfl3 * rho * (1.0f + uxmuz * (4.5f * uxmuz + 3.0f) - u2);
			LOCAL( srcGrid, WT) = dfl3 * rho * (1.0f + mxpuz * (4.5f * mxpuz + 3.0f) - u2);
			LOCAL( srcGrid, WB) = dfl3 * rho * (1.0f + mxmuz * (4.5f * mxmuz + 3.0f) - u2);
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
		{
			const float c  = LOCAL( grid, C  );
			const float n  = LOCAL( grid, N  );
			const float s  = LOCAL( grid, S  );
			const float e  = LOCAL( grid, E  );
			const float w  = LOCAL( grid, W  );
			const float t  = LOCAL( grid, T  );
			const float b  = LOCAL( grid, B  );
			const float ne = LOCAL( grid, NE );
			const float nw = LOCAL( grid, NW );
			const float se = LOCAL( grid, SE );
			const float sw = LOCAL( grid, SW );
			const float nt = LOCAL( grid, NT );
			const float nb = LOCAL( grid, NB );
			const float st = LOCAL( grid, ST );
			const float sb = LOCAL( grid, SB );
			const float et = LOCAL( grid, ET );
			const float eb = LOCAL( grid, EB );
			const float wt = LOCAL( grid, WT );
			const float wb = LOCAL( grid, WB );

			rho = c + n + s + e + w + t + b +
			      ne + nw + se + sw +
			      nt + nb + st + sb +
			      et + eb + wt + wb;
		}

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

			{
				const float e  = LOCAL( grid, E  );
				const float w  = LOCAL( grid, W  );
				const float n  = LOCAL( grid, N  );
				const float s  = LOCAL( grid, S  );
				const float ne = LOCAL( grid, NE );
				const float nw = LOCAL( grid, NW );
				const float se = LOCAL( grid, SE );
				const float sw = LOCAL( grid, SW );
				const float nt = LOCAL( grid, NT );
				const float nb = LOCAL( grid, NB );
				const float st = LOCAL( grid, ST );
				const float sb = LOCAL( grid, SB );
				const float t  = LOCAL( grid, T  );
				const float b  = LOCAL( grid, B  );
				const float et = LOCAL( grid, ET );
				const float eb = LOCAL( grid, EB );
				const float wt = LOCAL( grid, WT );
				const float wb = LOCAL( grid, WB );

				ux =  e - w +
				      ne - nw +
				      se - sw +
				      et + eb -
				      wt - wb;
				uy =  n - s +
				      ne + nw -
				      se - sw +
				      nt + nb -
				      st - sb;
				uz =  t - b +
				      nt - nb +
				      st - sb +
				      et - eb +
				      wt - wb;
			}

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
	if( (*((unsigned char*) &litteBigEndianTest)) == 0 ) {         /* big endian */
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
	if( (*((unsigned char*) &litteBigEndianTest)) == 0 ) {         /* big endian */
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

				rho =  c + n + s + e + w + t + b +
				       ne + nw + se + sw +
				       nt + nb + st + sb +
				       et + eb + wt + wb;

				ux =  e - w +
				      ne - nw +
				      se - sw +
				      et + eb -
				      wt - wb;
				uy =  n - s +
				      ne + nw -
				      se - sw +
				      nt + nb -
				      st - sb;
				uz =  t - b +
				      nt - nb +
				      st - sb +
				      et - eb +
				      wt - wb;
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

				rho =  c + n + s + e + w + t + b +
				       ne + nw + se + sw +
				       nt + nb + st + sb +
				       et + eb + wt + wb;

				ux =  e - w +
				      ne - nw +
				      se - sw +
				      et + eb -
				      wt - wb;
				uy =  n - s +
				      ne + nw -
				      se - sw +
				      nt + nb -
				      st - sb;
				uz =  t - b +
				      nt - nb +
				      st - sb +
				      et - eb +
				      wt - wb;
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
	        sqrtf( maxDiff2 )  );
#else
	printf( "LBM_compareVelocityField: maxDiff = %e  ==>  %s\n\n",
	        sqrtf( maxDiff2 ),
	        sqrtf( maxDiff2 ) > 1e-5 ? "##### ERROR #####" : "OK" );
#endif
	fclose( file );
}

