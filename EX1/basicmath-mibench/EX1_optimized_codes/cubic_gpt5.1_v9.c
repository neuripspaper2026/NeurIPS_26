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
      const long double Q = (a1_sq - 3.0L * a2) * (1.0L / 9.0L);
      const long double a1_cu = a1_sq * a1;
      const long double R = (2.0L * a1_cu - 9.0L * a1 * a2 + 27.0L * a3) * (1.0L / 54.0L);
      const long double Q_sq = Q * Q;
      const long double Q_cu = Q_sq * Q;
      const long double R_sq = R * R;
      const double R2_Q3 = (double)(R_sq - Q_cu);

      if (R2_Q3 <= 0.0)
      {
            *solutions = 3;

            const long double sqrt_Q = sqrtl(Q);
            const long double Q_sqrt_cu = sqrt_Q * Q;
            const long double ratio = R / Q_sqrt_cu;
            const long double theta = acosl(ratio);
            const long double two_sqrt_Q = 2.0L * sqrt_Q;
            const long double a1_div_3 = a1 * (1.0L / 3.0L);

            const long double theta_over_3 = theta * (1.0L / 3.0L);
            const long double two_pi_over_3 = (2.0L * PI) * (1.0L / 3.0L);
            const long double four_pi_over_3 = (4.0L * PI) * (1.0L / 3.0L);

            const long double cos_t1 = cosl(theta_over_3);
            const long double cos_t2 = cosl(theta_over_3 + two_pi_over_3);
            const long double cos_t3 = cosl(theta_over_3 + four_pi_over_3);

            x[0] = (double)(-two_sqrt_Q * cos_t1 - a1_div_3);
            x[1] = (double)(-two_sqrt_Q * cos_t2 - a1_div_3);
            x[2] = (double)(-two_sqrt_Q * cos_t3 - a1_div_3);
      }
      else
      {
            *solutions = 1;

            const long double sqrt_R2_Q3 = sqrtl((long double)R2_Q3);
            const long double abs_R = fabsl(R);
            long double x0 = powl(sqrt_R2_Q3 + abs_R, 1.0L / 3.0L);
            x0 += Q / x0;
            x0 *= (R < 0.0L) ? 1.0L : -1.0L;
            x0 -= a1 * (1.0L / 3.0L);

            x[0] = (double)x0;
      }

      clock_gettime(CLOCK_MONOTONIC, &kernel_end);
      solve_cubic_kernel_time_acc +=
          (kernel_end.tv_sec - kernel_start.tv_sec) +
          (kernel_end.tv_nsec - kernel_start.tv_nsec) * 1e-9;
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
