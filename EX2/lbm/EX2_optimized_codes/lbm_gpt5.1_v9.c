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

	const float c_val  = (float)DFL1;
	const float nswe_val = (float)DFL2;
	const float diag_val = (float)DFL3;

	/*voption indep*/
	SWEEP_START( 0, 0, -2, 0, 0, SIZE_Z+2 )
		/* use local pointer to reduce repeated address arithmetic */
		float *const __restrict base = &LOCAL(grid, C);

		base[C ]  = c_val;
		base[N ]  = nswe_val;
		base[S ]  = nswe_val;
		base[E ]  = nswe_val;
		base[W ]  = nswe_val;
		base[T ]  = nswe_val;
		base[B ]  = nswe_val;
		base[NE]  = diag_val;
		base[NW]  = diag_val;
		base[SE]  = diag_val;
		base[SW]  = diag_val;
		base[NT]  = diag_val;
		base[NB]  = diag_val;
		base[ST]  = diag_val;
		base[SB]  = diag_val;
		base[ET]  = diag_val;
		base[EB]  = diag_val;
		base[WT]  = diag_val;
		base[WB]  = diag_val;

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
			(void)fgetc( file );
		}
		(void)fgetc( file );
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
				else if( (z == 1 || z == SIZE_Z-2) &&
				         x > 1 && x < SIZE_X-2 &&
				         y > 1 && y < SIZE_Y-2 ) {
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
#pragma omp parallel for private(i,ux,uy,uz,u2,rho)
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
		}

		ux /= rho;
		uy /= rho;
		uz /= rho;

		if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
			ux = 0.005f;
			uy = 0.002f;
			uz = 0.000f;
		}

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);
		const float om = OMEGA;
		const float one_minus_om = 1.0f - om;

		DST_C ( dstGrid ) = one_minus_om*SRC_C ( srcGrid ) + (float)DFL1*om*rho*(1.0f                                 - u2);

		DST_N ( dstGrid ) = one_minus_om*SRC_N ( srcGrid ) + (float)DFL2*om*rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2);
		DST_S ( dstGrid ) = one_minus_om*SRC_S ( srcGrid ) + (float)DFL2*om*rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2);
		DST_E ( dstGrid ) = one_minus_om*SRC_E ( srcGrid ) + (float)DFL2*om*rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2);
		DST_W ( dstGrid ) = one_minus_om*SRC_W ( srcGrid ) + (float)DFL2*om*rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2);
		DST_T ( dstGrid ) = one_minus_om*SRC_T ( srcGrid ) + (float)DFL2*om*rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2);
		DST_B ( dstGrid ) = one_minus_om*SRC_B ( srcGrid ) + (float)DFL2*om*rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2);

		DST_NE( dstGrid ) = one_minus_om*SRC_NE( srcGrid ) + (float)DFL3*om*rho*(1.0f + (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
		DST_NW( dstGrid ) = one_minus_om*SRC_NW( srcGrid ) + (float)DFL3*om*rho*(1.0f + (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
		DST_SE( dstGrid ) = one_minus_om*SRC_SE( srcGrid ) + (float)DFL3*om*rho*(1.0f + (+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
		DST_SW( dstGrid ) = one_minus_om*SRC_SW( srcGrid ) + (float)DFL3*om*rho*(1.0f + (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
		DST_NT( dstGrid ) = one_minus_om*SRC_NT( srcGrid ) + (float)DFL3*om*rho*(1.0f + (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
		DST_NB( dstGrid ) = one_minus_om*SRC_NB( srcGrid ) + (float)DFL3*om*rho*(1.0f + (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
		DST_ST( dstGrid ) = one_minus_om*SRC_ST( srcGrid ) + (float)DFL3*om*rho*(1.0f + (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
		DST_SB( dstGrid ) = one_minus_om*SRC_SB( srcGrid ) + (float)DFL3*om*rho*(1.0f + (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
		DST_ET( dstGrid ) = one_minus_om*SRC_ET( srcGrid ) + (float)DFL3*om*rho*(1.0f + (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
		DST_EB( dstGrid ) = one_minus_om*SRC_EB( srcGrid ) + (float)DFL3*om*rho*(1.0f + (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
		DST_WT( dstGrid ) = one_minus_om*SRC_WT( srcGrid ) + (float)DFL3*om*rho*(1.0f + (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
		DST_WB( dstGrid ) = one_minus_om*SRC_WB( srcGrid ) + (float)DFL3*om*rho*(1.0f + (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
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
#pragma omp parallel for private(i,ux,uy,uz,rho,ux1,uy1,uz1,rho1,ux2,uy2,uz2,rho2,u2,px,py)
#endif
	SWEEP_START( 0, 0, 0, 0, 0, 1 )
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
	SWEEP_END

	/* outflow */
	/*voption indep*/
#if defined(_OPENMP) && !defined(SPEC_CPU)
#pragma omp parallel for private(i,ux,uy,uz,rho,ux1,uy1,uz1,rho1,ux2,uy2,uz2,rho2,u2,px,py)
#endif
	SWEEP_START( 0, 0, SIZE_Z-1, 0, 0, SIZE_Z )
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
	float mass = 0;

	SWEEP_VAR

	/* reduction over whole grid; parallelize outer sweep */
#if defined(_OPENMP) && !defined(SPEC_CPU)
#pragma omp parallel private(i,ux,uy,uz,u2,rho) reduction(+:nObstacleCells,nAccelCells,nFluidCells,mass) \
	reduction(min:minRho,minU2) reduction(max:maxRho,maxU2)
#endif
	{
#if defined(_OPENMP) && !defined(SPEC_CPU)
#pragma omp for
#endif
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

			const float e  = LOCAL( grid, E  );
			const float w  = LOCAL( grid, W  );
			const float ne = LOCAL( grid, NE );
			const float nw = LOCAL( grid, NW );
			const float se = LOCAL( grid, SE );
			const float sw = LOCAL( grid, SW );
			const float et = LOCAL( grid, ET );
			const float eb = LOCAL( grid, EB );
			const float wt = LOCAL( grid, WT );
			const float wb = LOCAL( grid, WB );
			const float n  = LOCAL( grid, N  );
			const float s  = LOCAL( grid, S  );
			const float nt = LOCAL( grid, NT );
			const float nb = LOCAL( grid, NB );
			const float st = LOCAL( grid, ST );
			const float sb = LOCAL( grid, SB );
			const float t  = LOCAL( grid, T  );
			const float b  = LOCAL( grid, B  );

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
			u2 = (ux*ux + uy*uy + uz*uz) / (rho*rho);
			if( u2 < minU2 ) minU2 = u2;
			if( u2 > maxU2 ) maxU2 = u2;
		}
	SWEEP_END
	} /* end parallel region */

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
		perror("LBM_storeVelocityField: fopen failed");
		return;
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
		perror("LBM_compareVelocityField: fopen failed");
		return;
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
	        sqrt( maxDiff2 )  );
#else
	printf( "LBM_compareVelocityField: maxDiff = %e  ==>  %s\n\n",
	        sqrt( maxDiff2 ),
	        sqrt( maxDiff2 ) > 1e-5 ? "##### ERROR #####" : "OK" );
#endif
	fclose( file );
}

