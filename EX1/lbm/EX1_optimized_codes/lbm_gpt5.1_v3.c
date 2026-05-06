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
	const size_t margin = 2u*SIZE_X*SIZE_Y*N_CELL_ENTRIES;
	const size_t size   = sizeof( LBM_Grid ) + 2u*margin*sizeof( float );

	*ptr = (float*)malloc( size );
	if( ! *ptr ) {
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
	const size_t margin = 2u*SIZE_X*SIZE_Y*N_CELL_ENTRIES;

	free( *ptr-margin );
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
		perror("LBM_loadObstacleFile");
		exit(1);
	}

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				int c = fgetc( file );
				if( c == EOF ) {
					fclose(file);
					fprintf(stderr, "LBM_loadObstacleFile: unexpected EOF\n");
					exit(1);
				}
				if( c != '.' ) SET_FLAG( grid, x, y, z, OBSTACLE );
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
		const int z_is_inner = (z != 0 && z != SIZE_Z-1);
		const int z_is_accel = (z == 1 || z == SIZE_Z-2);
		for( y = 0; y < SIZE_Y; y++ ) {
			const int y_is_border = (y == 0 || y == SIZE_Y-1);
			const int y_is_inner = !y_is_border && y > 1 && y < SIZE_Y-2;
			for( x = 0; x < SIZE_X; x++ ) {
				const int x_is_border = (x == 0 || x == SIZE_X-1);
				if( x_is_border || y_is_border || !z_is_inner ) {
					SET_FLAG( grid, x, y, z, OBSTACLE );
				}
				else if( z_is_accel &&
				         x > 1 && x < SIZE_X-2 &&
				         y_is_inner ) {
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
		const int z_is_inout = (z == 0 || z == SIZE_Z-1);
		for( y = 0; y < SIZE_Y; y++ ) {
			const int y_is_border = (y == 0 || y == SIZE_Y-1);
			for( x = 0; x < SIZE_X; x++ ) {
				const int x_is_border = (x == 0 || x == SIZE_X-1);
				if( x_is_border || y_is_border ) {
					SET_FLAG( grid, x, y, z, OBSTACLE );

					if( z_is_inout &&
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

	const float one_minus_omega = 1.0f - OMEGA;
	const float omega_dfl1 = DFL1 * OMEGA;
	const float omega_dfl2 = DFL2 * OMEGA;
	const float omega_dfl3 = DFL3 * OMEGA;

	/*voption indep*/
	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
			const float src_c  = SRC_C ( srcGrid );
			const float src_n  = SRC_N ( srcGrid );
			const float src_s  = SRC_S ( srcGrid );
			const float src_e  = SRC_E ( srcGrid );
			const float src_w  = SRC_W ( srcGrid );
			const float src_t  = SRC_T ( srcGrid );
			const float src_b  = SRC_B ( srcGrid );
			const float src_ne = SRC_NE( srcGrid );
			const float src_nw = SRC_NW( srcGrid );
			const float src_se = SRC_SE( srcGrid );
			const float src_sw = SRC_SW( srcGrid );
			const float src_nt = SRC_NT( srcGrid );
			const float src_nb = SRC_NB( srcGrid );
			const float src_st = SRC_ST( srcGrid );
			const float src_sb = SRC_SB( srcGrid );
			const float src_et = SRC_ET( srcGrid );
			const float src_eb = SRC_EB( srcGrid );
			const float src_wt = SRC_WT( srcGrid );
			const float src_wb = SRC_WB( srcGrid );

			DST_C ( dstGrid ) = src_c;
			DST_S ( dstGrid ) = src_n;
			DST_N ( dstGrid ) = src_s;
			DST_W ( dstGrid ) = src_e;
			DST_E ( dstGrid ) = src_w;
			DST_B ( dstGrid ) = src_t;
			DST_T ( dstGrid ) = src_b;
			DST_SW( dstGrid ) = src_ne;
			DST_SE( dstGrid ) = src_nw;
			DST_NW( dstGrid ) = src_se;
			DST_NE( dstGrid ) = src_sw;
			DST_SB( dstGrid ) = src_nt;
			DST_ST( dstGrid ) = src_nb;
			DST_NB( dstGrid ) = src_st;
			DST_NT( dstGrid ) = src_sb;
			DST_WB( dstGrid ) = src_et;
			DST_WT( dstGrid ) = src_eb;
			DST_EB( dstGrid ) = src_wt;
			DST_ET( dstGrid ) = src_wb;
			continue;
		}

		{
		const float src_c  = SRC_C ( srcGrid );
		const float src_n  = SRC_N ( srcGrid );
		const float src_s  = SRC_S ( srcGrid );
		const float src_e  = SRC_E ( srcGrid );
		const float src_w  = SRC_W ( srcGrid );
		const float src_t  = SRC_T ( srcGrid );
		const float src_b  = SRC_B ( srcGrid );
		const float src_ne = SRC_NE( srcGrid );
		const float src_nw = SRC_NW( srcGrid );
		const float src_se = SRC_SE( srcGrid );
		const float src_sw = SRC_SW( srcGrid );
		const float src_nt = SRC_NT( srcGrid );
		const float src_nb = SRC_NB( srcGrid );
		const float src_st = SRC_ST( srcGrid );
		const float src_sb = SRC_SB( srcGrid );
		const float src_et = SRC_ET( srcGrid );
		const float src_eb = SRC_EB( srcGrid );
		const float src_wt = SRC_WT( srcGrid );
		const float src_wb = SRC_WB( srcGrid );

		rho = src_c + src_n + src_s + src_e + src_w + src_t + src_b +
		      src_ne + src_nw + src_se + src_sw +
		      src_nt + src_nb + src_st + src_sb +
		      src_et + src_eb + src_wt + src_wb;

		ux =  src_e - src_w +
		      src_ne - src_nw +
		      src_se - src_sw +
		      src_et + src_eb -
		      src_wt - src_wb;
		uy =  src_n - src_s +
		      src_ne + src_nw -
		      src_se - src_sw +
		      src_nt + src_nb -
		      src_st - src_sb;
		uz =  src_t - src_b +
		      src_nt - src_nb +
		      src_st - src_sb +
		      src_et - src_eb +
		      src_wt - src_wb;

		const float inv_rho = 1.0f / rho;
		ux *= inv_rho;
		uy *= inv_rho;
		uz *= inv_rho;

		if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
			ux = 0.005f;
			uy = 0.002f;
			uz = 0.000f;
		}

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

		DST_C ( dstGrid ) = one_minus_omega*src_c + omega_dfl1*rho*(1.0f                                 - u2);

		DST_N ( dstGrid ) = one_minus_omega*src_n + omega_dfl2*rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2);
		DST_S ( dstGrid ) = one_minus_omega*src_s + omega_dfl2*rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2);
		DST_E ( dstGrid ) = one_minus_omega*src_e + omega_dfl2*rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2);
		DST_W ( dstGrid ) = one_minus_omega*src_w + omega_dfl2*rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2);
		DST_T ( dstGrid ) = one_minus_omega*src_t + omega_dfl2*rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2);
		DST_B ( dstGrid ) = one_minus_omega*src_b + omega_dfl2*rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2);

		const float ux_uy_p = ux + uy;
		const float ux_uy_m = ux - uy;
		const float uy_uz_p = uy + uz;
		const float uy_uz_m = uy - uz;
		const float ux_uz_p = ux + uz;
		const float ux_uz_m = ux - uz;
		const float nux_uy_p = -ux + uy;
		const float nux_uy_m = -ux - uy;
		const float nuy_uz_p = -uy + uz;
		const float nuy_uz_m = -uy - uz;
		const float nux_uz_p = -ux + uz;
		const float nux_uz_m = -ux - uz;

		DST_NE( dstGrid ) = one_minus_omega*src_ne + omega_dfl3*rho*(1.0f + ( ux_uy_p)*(4.5f*( ux_uy_p) + 3.0f) - u2);
		DST_NW( dstGrid ) = one_minus_omega*src_nw + omega_dfl3*rho*(1.0f + ( nux_uy_p)*(4.5f*( nux_uy_p) + 3.0f) - u2);
		DST_SE( dstGrid ) = one_minus_omega*src_se + omega_dfl3*rho*(1.0f + ( ux_uy_m)*(4.5f*( ux_uy_m) + 3.0f) - u2);
		DST_SW( dstGrid ) = one_minus_omega*src_sw + omega_dfl3*rho*(1.0f + ( nux_uy_m)*(4.5f*( nux_uy_m) + 3.0f) - u2);
		DST_NT( dstGrid ) = one_minus_omega*src_nt + omega_dfl3*rho*(1.0f + ( uy_uz_p)*(4.5f*( uy_uz_p) + 3.0f) - u2);
		DST_NB( dstGrid ) = one_minus_omega*src_nb + omega_dfl3*rho*(1.0f + ( uy_uz_m)*(4.5f*( uy_uz_m) + 3.0f) - u2);
		DST_ST( dstGrid ) = one_minus_omega*src_st + omega_dfl3*rho*(1.0f + ( nuy_uz_p)*(4.5f*( nuy_uz_p) + 3.0f) - u2);
		DST_SB( dstGrid ) = one_minus_omega*src_sb + omega_dfl3*rho*(1.0f + ( nuy_uz_m)*(4.5f*( nuy_uz_m) + 3.0f) - u2);
		DST_ET( dstGrid ) = one_minus_omega*src_et + omega_dfl3*rho*(1.0f + ( ux_uz_p)*(4.5f*( ux_uz_p) + 3.0f) - u2);
		DST_EB( dstGrid ) = one_minus_omega*src_eb + omega_dfl3*rho*(1.0f + ( ux_uz_m)*(4.5f*( ux_uz_m) + 3.0f) - u2);
		DST_WT( dstGrid ) = one_minus_omega*src_wt + omega_dfl3*rho*(1.0f + ( nux_uz_p)*(4.5f*( nux_uz_p) + 3.0f) - u2);
		DST_WB( dstGrid ) = one_minus_omega*src_wb + omega_dfl3*rho*(1.0f + ( nux_uz_m)*(4.5f*( nux_uz_m) + 3.0f) - u2);
		}
	SWEEP_END
}

void LBM_handleInOutFlow( LBM_Grid srcGrid ) {
	float ux , uy , uz , rho ,
	       ux1, uy1, uz1, rho1,
	       ux2, uy2, uz2, rho2,
	       u2, px, py;
	SWEEP_VAR

	const float coeff = 1.0f / (0.5f*(SIZE_X-1));
	const float coeff_y = 1.0f / (0.5f*(SIZE_Y-1));

	const float omega_dfl1 = DFL1;
	const float omega_dfl2 = DFL2;
	const float omega_dfl3 = DFL3;

	/* inflow */
	/*voption indep*/
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

		rho = 2.0f*rho1 - rho2;

		px = (SWEEP_X * coeff) - 1.0f;
		py = (SWEEP_Y * coeff_y) - 1.0f;
		ux = 0.00f;
		uy = 0.00f;
		uz = 0.01f * (1.0f-px*px) * (1.0f-py*py);

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = omega_dfl1*rho*(1.0f                                 - u2);

		LOCAL( srcGrid, N ) = omega_dfl2*rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2);
		LOCAL( srcGrid, S ) = omega_dfl2*rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2);
		LOCAL( srcGrid, E ) = omega_dfl2*rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2);
		LOCAL( srcGrid, W ) = omega_dfl2*rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2);
		LOCAL( srcGrid, T ) = omega_dfl2*rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2);
		LOCAL( srcGrid, B ) = omega_dfl2*rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2);

		{
		const float ux_uy_p = ux + uy;
		const float ux_uy_m = ux - uy;
		const float uy_uz_p = uy + uz;
		const float uy_uz_m = uy - uz;
		const float ux_uz_p = ux + uz;
		const float ux_uz_m = ux - uz;
		const float nux_uy_p = -ux + uy;
		const float nux_uy_m = -ux - uy;
		const float nuy_uz_p = -uy + uz;
		const float nuy_uz_m = -uy - uz;

		LOCAL( srcGrid, NE) = omega_dfl3*rho*(1.0f + ( ux_uy_p)*(4.5f*( ux_uy_p) + 3.0f) - u2);
		LOCAL( srcGrid, NW) = omega_dfl3*rho*(1.0f + ( nux_uy_p)*(4.5f*( nux_uy_p) + 3.0f) - u2);
		LOCAL( srcGrid, SE) = omega_dfl3*rho*(1.0f + ( ux_uy_m)*(4.5f*( ux_uy_m) + 3.0f) - u2);
		LOCAL( srcGrid, SW) = omega_dfl3*rho*(1.0f + ( nux_uy_m)*(4.5f*( nux_uy_m) + 3.0f) - u2);
		LOCAL( srcGrid, NT) = omega_dfl3*rho*(1.0f + ( uy_uz_p)*(4.5f*( uy_uz_p) + 3.0f) - u2);
		LOCAL( srcGrid, NB) = omega_dfl3*rho*(1.0f + ( uy_uz_m)*(4.5f*( uy_uz_m) + 3.0f) - u2);
		LOCAL( srcGrid, ST) = omega_dfl3*rho*(1.0f + ( nuy_uz_p)*(4.5f*( nuy_uz_p) + 3.0f) - u2);
		LOCAL( srcGrid, SB) = omega_dfl3*rho*(1.0f + ( nuy_uz_m)*(4.5f*( nuy_uz_m) + 3.0f) - u2);
		LOCAL( srcGrid, ET) = omega_dfl3*rho*(1.0f + ( ux_uz_p)*(4.5f*( ux_uz_p) + 3.0f) - u2);
		LOCAL( srcGrid, EB) = omega_dfl3*rho*(1.0f + ( ux_uz_m)*(4.5f*( ux_uz_m) + 3.0f) - u2);
		LOCAL( srcGrid, WT) = omega_dfl3*rho*(1.0f + ( -ux+uz)*(4.5f*( -ux+uz) + 3.0f) - u2);
		LOCAL( srcGrid, WB) = omega_dfl3*rho*(1.0f + ( -ux-uz)*(4.5f*( -ux-uz) + 3.0f) - u2);
		}
	SWEEP_END

	/* outflow */
	/*voption indep*/
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
		}

		{
		const float inv_rho1 = 1.0f / rho1;
		const float inv_rho2 = 1.0f / rho2;
		ux1 *= inv_rho1;
		uy1 *= inv_rho1;
		uz1 *= inv_rho1;

		ux2 *= inv_rho2;
		uy2 *= inv_rho2;
		uz2 *= inv_rho2;
		}

		rho = 1.0f;

		ux = 2.0f*ux1 - ux2;
		uy = 2.0f*uy1 - uy2;
		uz = 2.0f*uz1 - uz2;

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = omega_dfl1*rho*(1.0f                                 - u2);

		LOCAL( srcGrid, N ) = omega_dfl2*rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2);
		LOCAL( srcGrid, S ) = omega_dfl2*rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2);
		LOCAL( srcGrid, E ) = omega_dfl2*rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2);
		LOCAL( srcGrid, W ) = omega_dfl2*rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2);
		LOCAL( srcGrid, T ) = omega_dfl2*rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2);
		LOCAL( srcGrid, B ) = omega_dfl2*rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2);

		{
		const float ux_uy_p = ux + uy;
		const float ux_uy_m = ux - uy;
		const float uy_uz_p = uy + uz;
		const float uy_uz_m = uy - uz;
		const float ux_uz_p = ux + uz;
		const float ux_uz_m = ux - uz;
		const float nux_uy_p = -ux + uy;
		const float nux_uy_m = -ux - uy;
		const float nuy_uz_p = -uy + uz;
		const float nuy_uz_m = -uy - uz;

		LOCAL( srcGrid, NE) = omega_dfl3*rho*(1.0f + ( ux_uy_p)*(4.5f*( ux_uy_p) + 3.0f) - u2);
		LOCAL( srcGrid, NW) = omega_dfl3*rho*(1.0f + ( nux_uy_p)*(4.5f*( nux_uy_p) + 3.0f) - u2);
		LOCAL( srcGrid, SE) = omega_dfl3*rho*(1.0f + ( ux_uy_m)*(4.5f*( ux_uy_m) + 3.0f) - u2);
		LOCAL( srcGrid, SW) = omega_dfl3*rho*(1.0f + ( nux_uy_m)*(4.5f*( nux_uy_m) + 3.0f) - u2);
		LOCAL( srcGrid, NT) = omega_dfl3*rho*(1.0f + ( uy_uz_p)*(4.5f*( uy_uz_p) + 3.0f) - u2);
		LOCAL( srcGrid, NB) = omega_dfl3*rho*(1.0f + ( uy_uz_m)*(4.5f*( uy_uz_m) + 3.0f) - u2);
		LOCAL( srcGrid, ST) = omega_dfl3*rho*(1.0f + ( nuy_uz_p)*(4.5f*( nuy_uz_p) + 3.0f) - u2);
		LOCAL( srcGrid, SB) = omega_dfl3*rho*(1.0f + ( nuy_uz_m)*(4.5f*( nuy_uz_m) + 3.0f) - u2);
		LOCAL( srcGrid, ET) = omega_dfl3*rho*(1.0f + ( ux_uz_p)*(4.5f*( ux_uz_p) + 3.0f) - u2);
		LOCAL( srcGrid, EB) = omega_dfl3*rho*(1.0f + ( ux_uz_m)*(4.5f*( ux_uz_m) + 3.0f) - u2);
		LOCAL( srcGrid, WT) = omega_dfl3*rho*(1.0f + ( -ux+uz)*(4.5f*( -ux+uz) + 3.0f) - u2);
		LOCAL( srcGrid, WB) = omega_dfl3*rho*(1.0f + ( -ux-uz)*(4.5f*( -ux-uz) + 3.0f) - u2);
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

		(void)fwrite( buffer, sizeof( OUTPUT_PRECISION ), 1, file );
	}
	else {                                                     /* little endian */
		(void)fwrite( v, sizeof( OUTPUT_PRECISION ), 1, file );
	}
}

/*############################################################################*/

static void loadValue( FILE* file, OUTPUT_PRECISION* v ) {
	const int litteBigEndianTest = 1;
	if( (*((unsigned char*) &litteBigEndianTest)) == 0 ) {         /* big endian */
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

				{
				const float inv_rho = 1.0f / rho;
				ux *= inv_rho;
				uy *= inv_rho;
				uz *= inv_rho;
				}

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

				{
				const float inv_rho = 1.0f / rho;
				ux *= inv_rho;
				uy *= inv_rho;
				uz *= inv_rho;
				}

				if( binary ) {
					loadValue( file, &fileUx );
					loadValue( file, &fileUy );
					loadValue( file, &fileUz );
				}
				else {
					if( sizeof( OUTPUT_PRECISION ) == sizeof( double )) {
						if (fscanf( file, "%lf %lf %lf\n", &fileUx, &fileUy, &fileUz ) != 3) {
							fclose(file);
							fprintf(stderr, "LBM_compareVelocityField: read error\n");
							exit(1);
						}
					}
					else {
						if (fscanf( file, "%f %f %f\n", &fileUx, &fileUy, &fileUz ) != 3) {
							fclose(file);
							fprintf(stderr, "LBM_compareVelocityField: read error\n");
							exit(1);
						}
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
	        sqrtf( maxDiff2 ) > 1e-5f ? "##### ERROR #####" : "OK" );
#endif
	fclose( file );
}

