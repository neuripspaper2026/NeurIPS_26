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
	const size_t plane   = (size_t)SIZE_X*SIZE_Y*N_CELL_ENTRIES;
	const size_t margin  = 2*plane;
	const size_t size    = sizeof( LBM_Grid ) + 2*margin*sizeof( float );

	float *base = (float*)malloc( size );
	if( !base ) {
		printf( "LBM_allocateGrid: could not allocate %.1f MByte\n",
		        size / (1024.0*1024.0) );
		exit( 1 );
	}
#if !defined(SPEC_CPU)
	printf( "LBM_allocateGrid: allocated %.1f MByte\n",
	        size / (1024.0*1024.0) );
#endif
	*ptr = base + margin;
}

void LBM_freeGrid( float** ptr ) {
	const size_t margin = 2*(size_t)SIZE_X*SIZE_Y*N_CELL_ENTRIES;

	free( *ptr - margin );
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

		{
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

			rho = c + n + s + e + w + t + b +
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

			if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
				ux = 0.005f;
				uy = 0.002f;
				uz = 0.000f;
			}

			u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

			const float om   = OMEGA;
			const float iom  = 1.0f - om;
			const float omDFL1 = om * (float)DFL1 * rho;
			const float omDFL2 = om * (float)DFL2 * rho;
			const float omDFL3 = om * (float)DFL3 * rho;

			DST_C ( dstGrid ) = iom*c + omDFL1*(1.0f                                 - u2);

			const float cuy  = 4.5f*uy;
			const float cux  = 4.5f*ux;
			const float cuz  = 4.5f*uz;

			DST_N ( dstGrid ) = iom*n  + omDFL2*(1.0f +       uy*(cuy + 3.0f) - u2);
			DST_S ( dstGrid ) = iom*s  + omDFL2*(1.0f +       uy*(cuy - 3.0f) - u2);
			DST_E ( dstGrid ) = iom*e  + omDFL2*(1.0f +       ux*(cux + 3.0f) - u2);
			DST_W ( dstGrid ) = iom*w  + omDFL2*(1.0f +       ux*(cux - 3.0f) - u2);
			DST_T ( dstGrid ) = iom*t  + omDFL2*(1.0f +       uz*(cuz + 3.0f) - u2);
			DST_B ( dstGrid ) = iom*b  + omDFL2*(1.0f +       uz*(cuz - 3.0f) - u2);

			const float uxuy_p =  ux + uy;
			const float uxuy_m =  ux - uy;
			const float uyuz_p =  uy + uz;
			const float uyuz_m =  uy - uz;
			const float uxuz_p =  ux + uz;
			const float uxuz_m =  ux - uz;

			DST_NE( dstGrid ) = iom*ne + omDFL3*(1.0f + ( uxuy_p)*(4.5f*( uxuy_p) + 3.0f) - u2);
			DST_NW( dstGrid ) = iom*nw + omDFL3*(1.0f + (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
			DST_SE( dstGrid ) = iom*se + omDFL3*(1.0f + ( uxuy_m)*(4.5f*( uxuy_m) + 3.0f) - u2);
			DST_SW( dstGrid ) = iom*sw + omDFL3*(1.0f + (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
			DST_NT( dstGrid ) = iom*nt + omDFL3*(1.0f + ( uyuz_p)*(4.5f*( uyuz_p) + 3.0f) - u2);
			DST_NB( dstGrid ) = iom*nb + omDFL3*(1.0f + ( uyuz_m)*(4.5f*( uyuz_m) + 3.0f) - u2);
			DST_ST( dstGrid ) = iom*st + omDFL3*(1.0f + (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
			DST_SB( dstGrid ) = iom*sb + omDFL3*(1.0f + (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
			DST_ET( dstGrid ) = iom*et + omDFL3*(1.0f + ( uxuz_p)*(4.5f*( uxuz_p) + 3.0f) - u2);
			DST_EB( dstGrid ) = iom*eb + omDFL3*(1.0f + ( uxuz_m)*(4.5f*( uxuz_m) + 3.0f) - u2);
			DST_WT( dstGrid ) = iom*wt + omDFL3*(1.0f + (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
			DST_WB( dstGrid ) = iom*wb + omDFL3*(1.0f + (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
		}
	SWEEP_END
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

			rho1 = c1 + n1 + s1 + e1 + w1 + t1 + b1 +
			       ne1 + nw1 + se1 + sw1 +
			       nt1 + nb1 + st1 + sb1 +
			       et1 + eb1 + wt1 + wb1;

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

			rho2 = c2 + n2 + s2 + e2 + w2 + t2 + b2 +
			       ne2 + nw2 + se2 + sw2 +
			       nt2 + nb2 + st2 + sb2 +
			       et2 + eb2 + wt2 + wb2;
		}

		rho = 2.0f*rho1 - rho2;

		px = (float)(SWEEP_X) / (0.5f*(SIZE_X-1)) - 1.0f;
		py = (float)(SWEEP_Y) / (0.5f*(SIZE_Y-1)) - 1.0f;
		ux = 0.00f;
		uy = 0.00f;
		uz = 0.01f * (1.0f-px*px) * (1.0f-py*py);

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

		{
			const float om   = OMEGA;
			const float omDFL1 = om * (float)DFL1 * rho;
			const float omDFL2 = om * (float)DFL2 * rho;
			const float omDFL3 = om * (float)DFL3 * rho;

			const float cuy  = 4.5f*uy;
			const float cux  = 4.5f*ux;
			const float cuz  = 4.5f*uz;

			LOCAL( srcGrid, C ) = omDFL1*(1.0f                                 - u2);

			LOCAL( srcGrid, N ) = omDFL2*(1.0f +       uy*(cuy + 3.0f) - u2);
			LOCAL( srcGrid, S ) = omDFL2*(1.0f +       uy*(cuy - 3.0f) - u2);
			LOCAL( srcGrid, E ) = omDFL2*(1.0f +       ux*(cux + 3.0f) - u2);
			LOCAL( srcGrid, W ) = omDFL2*(1.0f +       ux*(cux - 3.0f) - u2);
			LOCAL( srcGrid, T ) = omDFL2*(1.0f +       uz*(cuz + 3.0f) - u2);
			LOCAL( srcGrid, B ) = omDFL2*(1.0f +       uz*(cuz - 3.0f) - u2);

			const float uxuy_p =  ux + uy;
			const float uxuy_m =  ux - uy;
			const float uyuz_p =  uy + uz;
			const float uyuz_m =  uy - uz;
			const float uxuz_p =  ux + uz;
			const float uxuz_m =  ux - uz;

			LOCAL( srcGrid, NE) = omDFL3*(1.0f + ( uxuy_p)*(4.5f*( uxuy_p) + 3.0f) - u2);
			LOCAL( srcGrid, NW) = omDFL3*(1.0f + (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
			LOCAL( srcGrid, SE) = omDFL3*(1.0f + ( uxuy_m)*(4.5f*( uxuy_m) + 3.0f) - u2);
			LOCAL( srcGrid, SW) = omDFL3*(1.0f + (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
			LOCAL( srcGrid, NT) = omDFL3*(1.0f + ( uyuz_p)*(4.5f*( uyuz_p) + 3.0f) - u2);
			LOCAL( srcGrid, NB) = omDFL3*(1.0f + ( uyuz_m)*(4.5f*( uyuz_m) + 3.0f) - u2);
			LOCAL( srcGrid, ST) = omDFL3*(1.0f + (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
			LOCAL( srcGrid, SB) = omDFL3*(1.0f + (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
			LOCAL( srcGrid, ET) = omDFL3*(1.0f + ( uxuz_p)*(4.5f*( uxuz_p) + 3.0f) - u2);
			LOCAL( srcGrid, EB) = omDFL3*(1.0f + ( uxuz_m)*(4.5f*( uxuz_m) + 3.0f) - u2);
			LOCAL( srcGrid, WT) = omDFL3*(1.0f + (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
			LOCAL( srcGrid, WB) = omDFL3*(1.0f + (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
		}
	SWEEP_END

	/* outflow */
	/*voption indep*/
#if defined(_OPENMP) && !defined(SPEC_CPU)
	#pragma omp parallel for private(i,ux,uy,uz,rho,ux1,uy1,uz1,rho1,ux2,uy2,uz2,rho2,u2,px,py) schedule(static)
#endif
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

			ux1 /= rho1;
			uy1 /= rho1;
			uz1 /= rho1;

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

			ux2 /= rho2;
			uy2 /= rho2;
			uz2 /= rho2;
		}

		rho = 1.0f;

		ux = 2.0f*ux1 - ux2;
		uy = 2.0f*uy1 - uy2;
		uz = 2.0f*uz1 - uz2;

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

		{
			const float om   = OMEGA;
			const float omDFL1 = om * (float)DFL1 * rho;
			const float omDFL2 = om * (float)DFL2 * rho;
			const float omDFL3 = om * (float)DFL3 * rho;

			const float cuy  = 4.5f*uy;
			const float cux  = 4.5f*ux;
			const float cuz  = 4.5f*uz;

			LOCAL( srcGrid, C ) = omDFL1*(1.0f                                 - u2);

			LOCAL( srcGrid, N ) = omDFL2*(1.0f +       uy*(cuy + 3.0f) - u2);
			LOCAL( srcGrid, S ) = omDFL2*(1.0f +       uy*(cuy - 3.0f) - u2);
			LOCAL( srcGrid, E ) = omDFL2*(1.0f +       ux*(cux + 3.0f) - u2);
			LOCAL( srcGrid, W ) = omDFL2*(1.0f +       ux*(cux - 3.0f) - u2);
			LOCAL( srcGrid, T ) = omDFL2*(1.0f +       uz*(cuz + 3.0f) - u2);
			LOCAL( srcGrid, B ) = omDFL2*(1.0f +       uz*(cuz - 3.0f) - u2);

			const float uxuy_p =  ux + uy;
			const float uxuy_m =  ux - uy;
			const float uyuz_p =  uy + uz;
			const float uyuz_m =  uy - uz;
			const float uxuz_p =  ux + uz;
			const float uxuz_m =  ux - uz;

			LOCAL( srcGrid, NE) = omDFL3*(1.0f + ( uxuy_p)*(4.5f*( uxuy_p) + 3.0f) - u2);
			LOCAL( srcGrid, NW) = omDFL3*(1.0f + (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
			LOCAL( srcGrid, SE) = omDFL3*(1.0f + ( uxuy_m)*(4.5f*( uxuy_m) + 3.0f) - u2);
			LOCAL( srcGrid, SW) = omDFL3*(1.0f + (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
			LOCAL( srcGrid, NT) = omDFL3*(1.0f + ( uyuz_p)*(4.5f*( uyuz_p) + 3.0f) - u2);
			LOCAL( srcGrid, NB) = omDFL3*(1.0f + ( uyuz_m)*(4.5f*( uyuz_m) + 3.0f) - u2);
			LOCAL( srcGrid, ST) = omDFL3*(1.0f + (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
			LOCAL( srcGrid, SB) = omDFL3*(1.0f + (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
			LOCAL( srcGrid, ET) = omDFL3*(1.0f + ( uxuz_p)*(4.5f*( uxuz_p) + 3.0f) - u2);
			LOCAL( srcGrid, EB) = omDFL3*(1.0f + ( uxuz_m)*(4.5f*( uxuz_m) + 3.0f) - u2);
			LOCAL( srcGrid, WT) = omDFL3*(1.0f + (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
			LOCAL( srcGrid, WB) = omDFL3*(1.0f + (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
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

#if defined(_OPENMP) && !defined(SPEC_CPU)
	#pragma omp parallel private(i,ux,uy,uz,u2,rho) reduction(+:mass,nObstacleCells,nAccelCells,nFluidCells) reduction(min:minU2,minRho) reduction(max:maxU2,maxRho)
#endif
	{
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
	}

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
	        sqrt( maxDiff2 )  );
#else
	printf( "LBM_compareVelocityField: maxDiff = %e  ==>  %s\n\n",
	        sqrt( maxDiff2 ),
	        sqrt( maxDiff2 ) > 1e-5 ? "##### ERROR #####" : "OK" );
#endif
	fclose( file );
}

