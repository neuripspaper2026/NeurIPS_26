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

	const float c_init  = (float)DFL1;
	const float ns_init = (float)DFL2;
	const float ew_init = (float)DFL2;
	const float tb_init = (float)DFL2;
	const float diag_init = (float)DFL3;

	/*voption indep*/
	SWEEP_START( 0, 0, -2, 0, 0, SIZE_Z+2 )
		LOCAL( grid, C  ) = c_init;
		LOCAL( grid, N  ) = ns_init;
		LOCAL( grid, S  ) = ns_init;
		LOCAL( grid, E  ) = ew_init;
		LOCAL( grid, W  ) = ew_init;
		LOCAL( grid, T  ) = tb_init;
		LOCAL( grid, B  ) = tb_init;
		LOCAL( grid, NE ) = diag_init;
		LOCAL( grid, NW ) = diag_init;
		LOCAL( grid, SE ) = diag_init;
		LOCAL( grid, SW ) = diag_init;
		LOCAL( grid, NT ) = diag_init;
		LOCAL( grid, NB ) = diag_init;
		LOCAL( grid, ST ) = diag_init;
		LOCAL( grid, SB ) = diag_init;
		LOCAL( grid, ET ) = diag_init;
		LOCAL( grid, EB ) = diag_init;
		LOCAL( grid, WT ) = diag_init;
		LOCAL( grid, WB ) = diag_init;

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
		fprintf(stderr, "LBM_loadObstacleFile: could not open %s\n", filename);
		exit(1);
	}

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				int c = fgetc( file );
				if( c != '.' && c != EOF ) SET_FLAG( grid, x, y, z, OBSTACLE );
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

	const float omega      = OMEGA;
	const float one_m_omega = 1.0f - omega;
	const float omega_dfl1 = (float)(DFL1) * omega;
	const float omega_dfl2 = (float)(DFL2) * omega;
	const float omega_dfl3 = (float)(DFL3) * omega;
	const float accel_ux = 0.005f;
	const float accel_uy = 0.002f;
	const float accel_uz = 0.0f;
	const float c1_5 = 1.5f;
	const float c4_5 = 4.5f;
	const float c3_0 = 3.0f;

	/*voption indep*/
#if defined(_OPENMP) && !defined(SPEC_CPU)
#pragma omp parallel for private(i,ux,uy,uz,u2,rho) schedule(static)
#endif
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
				ux = accel_ux;
				uy = accel_uy;
				uz = accel_uz;
			}

			u2 = c1_5 * (ux*ux + uy*uy + uz*uz);

			const float cuN  = uy;
			const float cuS  = uy;
			const float cuE  = ux;
			const float cuW  = ux;
			const float cuT  = uz;
			const float cuB  = uz;

			const float uxuy_p = ux + uy;
			const float uxuy_m = ux - uy;
			const float uyuz_p = uy + uz;
			const float uyuz_m = uy - uz;
			const float uxuz_p = ux + uz;
			const float uxuz_m = ux - uz;

			const float one_minus_u2 = 1.0f - u2;

			DST_C ( dstGrid ) = one_m_omega*src_c  + omega_dfl1*rho*(one_minus_u2);

			DST_N ( dstGrid ) = one_m_omega*src_n  + omega_dfl2*rho*(1.0f + cuN*(c4_5*cuN + c3_0) - u2);
			DST_S ( dstGrid ) = one_m_omega*src_s  + omega_dfl2*rho*(1.0f + cuS*(c4_5*cuS - c3_0) - u2);
			DST_E ( dstGrid ) = one_m_omega*src_e  + omega_dfl2*rho*(1.0f + cuE*(c4_5*cuE + c3_0) - u2);
			DST_W ( dstGrid ) = one_m_omega*src_w  + omega_dfl2*rho*(1.0f + cuW*(c4_5*cuW - c3_0) - u2);
			DST_T ( dstGrid ) = one_m_omega*src_t  + omega_dfl2*rho*(1.0f + cuT*(c4_5*cuT + c3_0) - u2);
			DST_B ( dstGrid ) = one_m_omega*src_b  + omega_dfl2*rho*(1.0f + cuB*(c4_5*cuB - c3_0) - u2);

			const float cuNE = uxuy_p;
			const float cuNW = -ux + uy;
			const float cuSE = ux - uy;
			const float cuSW = -uxuy_p;

			const float cuNT = uyuz_p;
			const float cuNB = uyuz_m;
			const float cuST = -uy + uz;
			const float cuSB = -uyuz_p;

			const float cuET = uxuz_p;
			const float cuEB = uxuz_m;
			const float cuWT = -ux + uz;
			const float cuWB = -uxuz_p;

			DST_NE( dstGrid ) = one_m_omega*src_ne + omega_dfl3*rho*(1.0f + cuNE*(c4_5*cuNE + c3_0) - u2);
			DST_NW( dstGrid ) = one_m_omega*src_nw + omega_dfl3*rho*(1.0f + cuNW*(c4_5*cuNW + c3_0) - u2);
			DST_SE( dstGrid ) = one_m_omega*src_se + omega_dfl3*rho*(1.0f + cuSE*(c4_5*cuSE + c3_0) - u2);
			DST_SW( dstGrid ) = one_m_omega*src_sw + omega_dfl3*rho*(1.0f + cuSW*(c4_5*cuSW + c3_0) - u2);
			DST_NT( dstGrid ) = one_m_omega*src_nt + omega_dfl3*rho*(1.0f + cuNT*(c4_5*cuNT + c3_0) - u2);
			DST_NB( dstGrid ) = one_m_omega*src_nb + omega_dfl3*rho*(1.0f + cuNB*(c4_5*cuNB + c3_0) - u2);
			DST_ST( dstGrid ) = one_m_omega*src_st + omega_dfl3*rho*(1.0f + cuST*(c4_5*cuST + c3_0) - u2);
			DST_SB( dstGrid ) = one_m_omega*src_sb + omega_dfl3*rho*(1.0f + cuSB*(c4_5*cuSB + c3_0) - u2);
			DST_ET( dstGrid ) = one_m_omega*src_et + omega_dfl3*rho*(1.0f + cuET*(c4_5*cuET + c3_0) - u2);
			DST_EB( dstGrid ) = one_m_omega*src_eb + omega_dfl3*rho*(1.0f + cuEB*(c4_5*cuEB + c3_0) - u2);
			DST_WT( dstGrid ) = one_m_omega*src_wt + omega_dfl3*rho*(1.0f + cuWT*(c4_5*cuWT + c3_0) - u2);
			DST_WB( dstGrid ) = one_m_omega*src_wb + omega_dfl3*rho*(1.0f + cuWB*(c4_5*cuWB + c3_0) - u2);
		}
	SWEEP_END
}

void LBM_handleInOutFlow( LBM_Grid srcGrid ) {
	float ux , uy , uz , rho ,
	       ux1, uy1, uz1, rho1,
	       ux2, uy2, uz2, rho2,
	       u2, px, py;
	SWEEP_VAR

	const float c1_5 = 1.5f;
	const float c4_5 = 4.5f;
	const float c3_0 = 3.0f;
	const float inflow_uz_scale = 0.01f;

	/* inflow */
	/*voption indep*/
#if defined(_OPENMP) && !defined(SPEC_CPU)
#pragma omp parallel for private(i,ux,uy,uz,rho,ux1,uy1,uz1,rho1,ux2,uy2,uz2,rho2,u2,px,py) schedule(static)
#endif
	SWEEP_START( 0, 0, 0, 0, 0, 1 )
		rho1 =
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, N  )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, E  )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, T  )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NE )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SE )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NT )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ST )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ET )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WT )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WB );

		rho2 =
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, N  )
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
		ux = 0.0f;
		uy = 0.0f;
		uz = inflow_uz_scale * (1.0f-px*px) * (1.0f-py*py);

		u2 = c1_5 * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = (float)DFL1*rho*(1.0f - u2);

		LOCAL( srcGrid, N ) = (float)DFL2*rho*(1.0f +       uy*(c4_5*uy       + c3_0) - u2);
		LOCAL( srcGrid, S ) = (float)DFL2*rho*(1.0f +       uy*(c4_5*uy       - c3_0) - u2);
		LOCAL( srcGrid, E ) = (float)DFL2*rho*(1.0f +       ux*(c4_5*ux       + c3_0) - u2);
		LOCAL( srcGrid, W ) = (float)DFL2*rho*(1.0f +       ux*(c4_5*ux       - c3_0) - u2);
		LOCAL( srcGrid, T ) = (float)DFL2*rho*(1.0f +       uz*(c4_5*uz       + c3_0) - u2);
		LOCAL( srcGrid, B ) = (float)DFL2*rho*(1.0f +       uz*(c4_5*uz       - c3_0) - u2);

		{
			const float uxuy_p = ux + uy;
			const float uxuy_m = ux - uy;
			const float uyuz_p = uy + uz;
			const float uyuz_m = uy - uz;
			const float uxuz_p = ux + uz;
			const float uxuz_m = ux - uz;

			LOCAL( srcGrid, NE) = (float)DFL3*rho*(1.0f + (uxuy_p)*(c4_5*(uxuy_p) + c3_0) - u2);
			LOCAL( srcGrid, NW) = (float)DFL3*rho*(1.0f + (-ux+uy)*(c4_5*(-ux+uy) + c3_0) - u2);
			LOCAL( srcGrid, SE) = (float)DFL3*rho*(1.0f + (uxuy_m)*(c4_5*(uxuy_m) + c3_0) - u2);
			LOCAL( srcGrid, SW) = (float)DFL3*rho*(1.0f + (-ux-uy)*(c4_5*(-ux-uy) + c3_0) - u2);
			LOCAL( srcGrid, NT) = (float)DFL3*rho*(1.0f + (uyuz_p)*(c4_5*(uyuz_p) + c3_0) - u2);
			LOCAL( srcGrid, NB) = (float)DFL3*rho*(1.0f + (uyuz_m)*(c4_5*(uyuz_m) + c3_0) - u2);
			LOCAL( srcGrid, ST) = (float)DFL3*rho*(1.0f + (-uy+uz)*(c4_5*(-uy+uz) + c3_0) - u2);
			LOCAL( srcGrid, SB) = (float)DFL3*rho*(1.0f + (-uy-uz)*(c4_5*(-uy-uz) + c3_0) - u2);
			LOCAL( srcGrid, ET) = (float)DFL3*rho*(1.0f + (uxuz_p)*(c4_5*(uxuz_p) + c3_0) - u2);
			LOCAL( srcGrid, EB) = (float)DFL3*rho*(1.0f + (uxuz_m)*(c4_5*(uxuz_m) + c3_0) - u2);
			LOCAL( srcGrid, WT) = (float)DFL3*rho*(1.0f + (-ux+uz)*(c4_5*(-ux+uz) + c3_0) - u2);
			LOCAL( srcGrid, WB) = (float)DFL3*rho*(1.0f + (-ux-uz)*(c4_5*(-ux-uz) + c3_0) - u2);
		}
	SWEEP_END

	/* outflow */
	/*voption indep*/
#if defined(_OPENMP) && !defined(SPEC_CPU)
#pragma omp parallel for private(i,ux,uy,uz,rho,ux1,uy1,uz1,rho1,ux2,uy2,uz2,rho2,u2) schedule(static)
#endif
	SWEEP_START( 0, 0, SIZE_Z-1, 0, 0, SIZE_Z )
		rho1 =
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, N  )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, E  )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, T  )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );

		ux1 =
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, E  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, W  )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB )
		    - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );

		uy1 =
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, N  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, S  )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW )
		    - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB )
		    - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB );

		uz1 =
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, T  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, B  )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );

		ux1 /= rho1;
		uy1 /= rho1;
		uz1 /= rho1;

		rho2 =
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, N  )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, E  )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, T  )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );

		ux2 =
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, E  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, W  )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB )
		    - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );

		uy2 =
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, N  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, S  )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW )
		    - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW )
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB )
		    - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB );

		uz2 =
		    + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, T  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, B  )
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

		u2 = c1_5 * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = (float)DFL1*rho*(1.0f - u2);

		LOCAL( srcGrid, N ) = (float)DFL2*rho*(1.0f +       uy*(c4_5*uy       + c3_0) - u2);
		LOCAL( srcGrid, S ) = (float)DFL2*rho*(1.0f +       uy*(c4_5*uy       - c3_0) - u2);
		LOCAL( srcGrid, E ) = (float)DFL2*rho*(1.0f +       ux*(c4_5*ux       + c3_0) - u2);
		LOCAL( srcGrid, W ) = (float)DFL2*rho*(1.0f +       ux*(c4_5*ux       - c3_0) - u2);
		LOCAL( srcGrid, T ) = (float)DFL2*rho*(1.0f +       uz*(c4_5*uz       + c3_0) - u2);
		LOCAL( srcGrid, B ) = (float)DFL2*rho*(1.0f +       uz*(c4_5*uz       - c3_0) - u2);

		{
			const float uxuy_p = ux + uy;
			const float uxuy_m = ux - uy;
			const float uyuz_p = uy + uz;
			const float uyuz_m = uy - uz;
			const float uxuz_p = ux + uz;
			const float uxuz_m = ux - uz;

			LOCAL( srcGrid, NE) = (float)DFL3*rho*(1.0f + (uxuy_p)*(c4_5*(uxuy_p) + c3_0) - u2);
			LOCAL( srcGrid, NW) = (float)DFL3*rho*(1.0f + (-ux+uy)*(c4_5*(-ux+uy) + c3_0) - u2);
			LOCAL( srcGrid, SE) = (float)DFL3*rho*(1.0f + (uxuy_m)*(c4_5*(uxuy_m) + c3_0) - u2);
			LOCAL( srcGrid, SW) = (float)DFL3*rho*(1.0f + (-ux-uy)*(c4_5*(-ux-uy) + c3_0) - u2);
			LOCAL( srcGrid, NT) = (float)DFL3*rho*(1.0f + (uyuz_p)*(c4_5*(uyuz_p) + c3_0) - u2);
			LOCAL( srcGrid, NB) = (float)DFL3*rho*(1.0f + (uyuz_m)*(c4_5*(uyuz_m) + c3_0) - u2);
			LOCAL( srcGrid, ST) = (float)DFL3*rho*(1.0f + (-uy+uz)*(c4_5*(-uy+uz) + c3_0) - u2);
			LOCAL( srcGrid, SB) = (float)DFL3*rho*(1.0f + (-uy-uz)*(c4_5*(-uy-uz) + c3_0) - u2);
			LOCAL( srcGrid, ET) = (float)DFL3*rho*(1.0f + (uxuz_p)*(c4_5*(uxuz_p) + c3_0) - u2);
			LOCAL( srcGrid, EB) = (float)DFL3*rho*(1.0f + (uxuz_m)*(c4_5*(uxuz_m) + c3_0) - u2);
			LOCAL( srcGrid, WT) = (float)DFL3*rho*(1.0f + (-ux+uz)*(c4_5*(-ux+uz) + c3_0) - u2);
			LOCAL( srcGrid, WB) = (float)DFL3*rho*(1.0f + (-ux-uz)*(c4_5*(-ux-uz) + c3_0) - u2);
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

	/* Sequential accumulation kept for determinism */
	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		rho =
		      + LOCAL( grid, C  ) + LOCAL( grid, N  )
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

			ux =
			      + LOCAL( grid, E  ) - LOCAL( grid, W  )
			      + LOCAL( grid, NE ) - LOCAL( grid, NW )
			      + LOCAL( grid, SE ) - LOCAL( grid, SW )
			      + LOCAL( grid, ET ) + LOCAL( grid, EB )
			      - LOCAL( grid, WT ) - LOCAL( grid, WB );
			uy =
			      + LOCAL( grid, N  ) - LOCAL( grid, S  )
			      + LOCAL( grid, NE ) + LOCAL( grid, NW )
			      - LOCAL( grid, SE ) - LOCAL( grid, SW )
			      + LOCAL( grid, NT ) + LOCAL( grid, NB )
			      - LOCAL( grid, ST ) - LOCAL( grid, SB );
			uz =
			      + LOCAL( grid, T  ) - LOCAL( grid, B  )
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
		fprintf(stderr, "LBM_storeVelocityField: could not open %s\n", filename);
		exit(1);
	}

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				const OUTPUT_PRECISION c  = GRID_ENTRY( grid, x, y, z, C  );
				const OUTPUT_PRECISION n  = GRID_ENTRY( grid, x, y, z, N  );
				const OUTPUT_PRECISION s  = GRID_ENTRY( grid, x, y, z, S  );
				const OUTPUT_PRECISION e  = GRID_ENTRY( grid, x, y, z, E  );
				const OUTPUT_PRECISION w  = GRID_ENTRY( grid, x, y, z, W  );
				const OUTPUT_PRECISION t  = GRID_ENTRY( grid, x, y, z, T  );
				const OUTPUT_PRECISION b  = GRID_ENTRY( grid, x, y, z, B  );
				const OUTPUT_PRECISION ne = GRID_ENTRY( grid, x, y, z, NE );
				const OUTPUT_PRECISION nw = GRID_ENTRY( grid, x, y, z, NW );
				const OUTPUT_PRECISION se = GRID_ENTRY( grid, x, y, z, SE );
				const OUTPUT_PRECISION sw = GRID_ENTRY( grid, x, y, z, SW );
				const OUTPUT_PRECISION nt = GRID_ENTRY( grid, x, y, z, NT );
				const OUTPUT_PRECISION nb = GRID_ENTRY( grid, x, y, z, NB );
				const OUTPUT_PRECISION st = GRID_ENTRY( grid, x, y, z, ST );
				const OUTPUT_PRECISION sb = GRID_ENTRY( grid, x, y, z, SB );
				const OUTPUT_PRECISION et = GRID_ENTRY( grid, x, y, z, ET );
				const OUTPUT_PRECISION eb = GRID_ENTRY( grid, x, y, z, EB );
				const OUTPUT_PRECISION wt = GRID_ENTRY( grid, x, y, z, WT );
				const OUTPUT_PRECISION wb = GRID_ENTRY( grid, x, y, z, WB );

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
		fprintf(stderr, "LBM_compareVelocityField: could not open %s\n", filename);
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

