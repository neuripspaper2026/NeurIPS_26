#include <math.h>
#include <time.h>
#include "../snipmath.h"

static double solve_cubic_kernel_time_acc = 0.0;

void reset_solve_cubic_kernel_time(void) { solve_cubic_kernel_time_acc = 0.0; }
double get_solve_cubic_kernel_time(void) { return solve_cubic_kernel_time_acc; }

void SolveCubic(double  a,
                double  b,
                double  c,
                double  d,
                int    *solutions,
                double *x)
{
      struct timespec kernel_start, kernel_end;
      clock_gettime(CLOCK_MONOTONIC, &kernel_start);

      const long double inv_a = 1.0L / a;
      const long double a1 = b * inv_a;
      const long double a2 = c * inv_a;
      const long double a3 = d * inv_a;

      const long double a1_sq = a1 * a1;
      const long double Q = (a1_sq - 3.0L * a2) / 9.0L;
      const long double R = (2.0L * a1_sq * a1 - 9.0L * a1 * a2 + 27.0L * a3) / 54.0L;
      const long double Q_sq = Q * Q;
      const long double Q_cu = Q_sq * Q;
      const long double R2_Q3_ld = R * R - Q_cu;
      const double R2_Q3 = (double)R2_Q3_ld;

      if (R2_Q3 <= 0.0)
      {
            *solutions = 3;
            const long double Q_sqrt = sqrtl(Q);
            const long double Q_sqrt_c2 = -2.0L * Q_sqrt;
            const long double a1_div3 = a1 / 3.0L;
            const long double Q_cu_sqrt = Q_sqrt * Q;
            const long double R_div_Qc = R / Q_cu_sqrt;
            const double theta = acosl((long double)R_div_Qc);

            x[0] = (double)(Q_sqrt_c2 * cosl(theta / 3.0L) - a1_div3);
            x[1] = (double)(Q_sqrt_c2 * cosl((theta + 2.0L * PI) / 3.0L) - a1_div3);
            x[2] = (double)(Q_sqrt_c2 * cosl((theta + 4.0L * PI) / 3.0L) - a1_div3);
      }
      else
      {
            *solutions = 1;
            const long double R_abs = fabsl(R);
            const long double R2_Q3_sqrt = sqrtl(R2_Q3_ld);
            long double t = powl(R2_Q3_sqrt + R_abs, 1.0L / 3.0L);
            t += Q / t;
            if (R >= 0.0L)
                  t = -t;
            const long double a1_div3 = a1 / 3.0L;
            x[0] = (double)(t - a1_div3);
      }

      clock_gettime(CLOCK_MONOTONIC, &kernel_end);
      solve_cubic_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                     (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}

#ifdef TEST

int main(void)
{
      double  a1 = 1.0, b1 = -10.5, c1 = 32.0, d1 = -30.0;
      double  a2 = 1.0, b2 = -4.5, c2 = 17.0, d2 = -30.0;
      double  x[3];
      int     solutions;

      SolveCubic(a1, b1, c1, d1, &solutions, x);

      /* should get 3 solutions: 2, 6 & 2.5   */

      SolveCubic(a2, b2, c2, d2, &solutions, x);

      /* should get 1 solution: 2.5           */

      return 0;
}

#endif /* TEST */
