#include <time.h>
#include "../gemm.h"

static double gemm_ncubed_kernel_time_acc = 0.0;

void reset_gemm_ncubed_kernel_time(void) { gemm_ncubed_kernel_time_acc = 0.0; }
double get_gemm_ncubed_kernel_time(void) { return gemm_ncubed_kernel_time_acc; }

void gemm(TYPE m1[N], TYPE m2[N], TYPE prod[N]) {
  int i, j, k;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  for (i = 0; i < row_size; i++) {
    int i_col = i * col_size;
    for (j = 0; j < col_size; j++) {
      TYPE sum = 0;
      TYPE m2_j0  = m2[0  * col_size + j];
      TYPE m2_j1  = m2[1  * col_size + j];
      TYPE m2_j2  = m2[2  * col_size + j];
      TYPE m2_j3  = m2[3  * col_size + j];
      TYPE m2_j4  = m2[4  * col_size + j];
      TYPE m2_j5  = m2[5  * col_size + j];
      TYPE m2_j6  = m2[6  * col_size + j];
      TYPE m2_j7  = m2[7  * col_size + j];
      TYPE m2_j8  = m2[8  * col_size + j];
      TYPE m2_j9  = m2[9  * col_size + j];
      TYPE m2_j10 = m2[10 * col_size + j];
      TYPE m2_j11 = m2[11 * col_size + j];
      TYPE m2_j12 = m2[12 * col_size + j];
      TYPE m2_j13 = m2[13 * col_size + j];
      TYPE m2_j14 = m2[14 * col_size + j];
      TYPE m2_j15 = m2[15 * col_size + j];
      TYPE m2_j16 = m2[16 * col_size + j];
      TYPE m2_j17 = m2[17 * col_size + j];
      TYPE m2_j18 = m2[18 * col_size + j];
      TYPE m2_j19 = m2[19 * col_size + j];
      TYPE m2_j20 = m2[20 * col_size + j];
      TYPE m2_j21 = m2[21 * col_size + j];
      TYPE m2_j22 = m2[22 * col_size + j];
      TYPE m2_j23 = m2[23 * col_size + j];
      TYPE m2_j24 = m2[24 * col_size + j];
      TYPE m2_j25 = m2[25 * col_size + j];
      TYPE m2_j26 = m2[26 * col_size + j];
      TYPE m2_j27 = m2[27 * col_size + j];
      TYPE m2_j28 = m2[28 * col_size + j];
      TYPE m2_j29 = m2[29 * col_size + j];
      TYPE m2_j30 = m2[30 * col_size + j];
      TYPE m2_j31 = m2[31 * col_size + j];
      TYPE m2_j32 = m2[32 * col_size + j];
      TYPE m2_j33 = m2[33 * col_size + j];
      TYPE m2_j34 = m2[34 * col_size + j];
      TYPE m2_j35 = m2[35 * col_size + j];
      TYPE m2_j36 = m2[36 * col_size + j];
      TYPE m2_j37 = m2[37 * col_size + j];
      TYPE m2_j38 = m2[38 * col_size + j];
      TYPE m2_j39 = m2[39 * col_size + j];
      TYPE m2_j40 = m2[40 * col_size + j];
      TYPE m2_j41 = m2[41 * col_size + j];
      TYPE m2_j42 = m2[42 * col_size + j];
      TYPE m2_j43 = m2[43 * col_size + j];
      TYPE m2_j44 = m2[44 * col_size + j];
      TYPE m2_j45 = m2[45 * col_size + j];
      TYPE m2_j46 = m2[46 * col_size + j];
      TYPE m2_j47 = m2[47 * col_size + j];
      TYPE m2_j48 = m2[48 * col_size + j];
      TYPE m2_j49 = m2[49 * col_size + j];
      TYPE m2_j50 = m2[50 * col_size + j];
      TYPE m2_j51 = m2[51 * col_size + j];
      TYPE m2_j52 = m2[52 * col_size + j];
      TYPE m2_j53 = m2[53 * col_size + j];
      TYPE m2_j54 = m2[54 * col_size + j];
      TYPE m2_j55 = m2[55 * col_size + j];
      TYPE m2_j56 = m2[56 * col_size + j];
      TYPE m2_j57 = m2[57 * col_size + j];
      TYPE m2_j58 = m2[58 * col_size + j];
      TYPE m2_j59 = m2[59 * col_size + j];
      TYPE m2_j60 = m2[60 * col_size + j];
      TYPE m2_j61 = m2[61 * col_size + j];
      TYPE m2_j62 = m2[62 * col_size + j];
      TYPE m2_j63 = m2[63 * col_size + j];

      sum += m1[i_col +  0] * m2_j0;
      sum += m1[i_col +  1] * m2_j1;
      sum += m1[i_col +  2] * m2_j2;
      sum += m1[i_col +  3] * m2_j3;
      sum += m1[i_col +  4] * m2_j4;
      sum += m1[i_col +  5] * m2_j5;
      sum += m1[i_col +  6] * m2_j6;
      sum += m1[i_col +  7] * m2_j7;
      sum += m1[i_col +  8] * m2_j8;
      sum += m1[i_col +  9] * m2_j9;
      sum += m1[i_col + 10] * m2_j10;
      sum += m1[i_col + 11] * m2_j11;
      sum += m1[i_col + 12] * m2_j12;
      sum += m1[i_col + 13] * m2_j13;
      sum += m1[i_col + 14] * m2_j14;
      sum += m1[i_col + 15] * m2_j15;
      sum += m1[i_col + 16] * m2_j16;
      sum += m1[i_col + 17] * m2_j17;
      sum += m1[i_col + 18] * m2_j18;
      sum += m1[i_col + 19] * m2_j19;
      sum += m1[i_col + 20] * m2_j20;
      sum += m1[i_col + 21] * m2_j21;
      sum += m1[i_col + 22] * m2_j22;
      sum += m1[i_col + 23] * m2_j23;
      sum += m1[i_col + 24] * m2_j24;
      sum += m1[i_col + 25] * m2_j25;
      sum += m1[i_col + 26] * m2_j26;
      sum += m1[i_col + 27] * m2_j27;
      sum += m1[i_col + 28] * m2_j28;
      sum += m1[i_col + 29] * m2_j29;
      sum += m1[i_col + 30] * m2_j30;
      sum += m1[i_col + 31] * m2_j31;
      sum += m1[i_col + 32] * m2_j32;
      sum += m1[i_col + 33] * m2_j33;
      sum += m1[i_col + 34] * m2_j34;
      sum += m1[i_col + 35] * m2_j35;
      sum += m1[i_col + 36] * m2_j36;
      sum += m1[i_col + 37] * m2_j37;
      sum += m1[i_col + 38] * m2_j38;
      sum += m1[i_col + 39] * m2_j39;
      sum += m1[i_col + 40] * m2_j40;
      sum += m1[i_col + 41] * m2_j41;
      sum += m1[i_col + 42] * m2_j42;
      sum += m1[i_col + 43] * m2_j43;
      sum += m1[i_col + 44] * m2_j44;
      sum += m1[i_col + 45] * m2_j45;
      sum += m1[i_col + 46] * m2_j46;
      sum += m1[i_col + 47] * m2_j47;
      sum += m1[i_col + 48] * m2_j48;
      sum += m1[i_col + 49] * m2_j49;
      sum += m1[i_col + 50] * m2_j50;
      sum += m1[i_col + 51] * m2_j51;
      sum += m1[i_col + 52] * m2_j52;
      sum += m1[i_col + 53] * m2_j53;
      sum += m1[i_col + 54] * m2_j54;
      sum += m1[i_col + 55] * m2_j55;
      sum += m1[i_col + 56] * m2_j56;
      sum += m1[i_col + 57] * m2_j57;
      sum += m1[i_col + 58] * m2_j58;
      sum += m1[i_col + 59] * m2_j59;
      sum += m1[i_col + 60] * m2_j60;
      sum += m1[i_col + 61] * m2_j61;
      sum += m1[i_col + 62] * m2_j62;
      sum += m1[i_col + 63] * m2_j63;

      prod[i_col + j] = sum;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  gemm_ncubed_kernel_time_acc +=
      (kernel_end.tv_sec - kernel_start.tv_sec) +
      (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
