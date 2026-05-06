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
      const long double a1_cu = a1_sq * a1;
      const long double R = (2.0L * a1_cu - 9.0L * a1 * a2 + 27.0L * a3) / 54.0L;
      const long double Q_sq = Q * Q;
      const long double Q_cu = Q_sq * Q;
      const long double R2_Q3 = R * R - Q_cu;

      const double a1_div_3 = (double)(a1 / 3.0L);

      if (R2_Q3 <= 0.0L)
      {
            *solutions = 3;

            const double Qd = (double)Q;
            const double sqrtQ = sqrt(Qd);
            const double sqrtQ2 = -2.0 * sqrtQ;
            const double R_over_sqrtQ3 = (double)(R / sqrtQ / Q_sq);
            const double theta = acos(R_over_sqrtQ3);
            const double theta_over_3 = theta / 3.0;
            const double two_pi_over_3 = (2.0 * PI) / 3.0;
            const double four_pi_over_3 = (4.0 * PI) / 3.0;

            x[0] = sqrtQ2 * cos(theta_over_3) - a1_div_3;
            x[1] = sqrtQ2 * cos(theta_over_3 + two_pi_over_3) - a1_div_3;
            x[2] = sqrtQ2 * cos(theta_over_3 + four_pi_over_3) - a1_div_3;
      }
      else
      {
            *solutions = 1;

            const long double sqrtR2_Q3 = sqrtl(R2_Q3);
            long double x0 = sqrtR2_Q3 + fabsl(R);
            long double cbrt_part = cbrt(x0);
            const long double Q_over_cbrt = Q / cbrt_part;

            if (R < 0.0L)
                  x0 = cbrt_part + Q_over_cbrt;
            else
                  x0 = -(cbrt_part + Q_over_cbrt);

            x[0] = (double)(x0 - a1 / 3.0L);
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
