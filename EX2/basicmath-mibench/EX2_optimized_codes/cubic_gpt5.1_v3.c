#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
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

      /* Use long double for improved precision in intermediate steps */
      const long double inv_a = 1.0L / a;
      const long double a1 = (long double)b * inv_a;
      const long double a2 = (long double)c * inv_a;
      const long double a3 = (long double)d * inv_a;

      const long double a1_sq = a1 * a1;
      const long double Q = (a1_sq - 3.0L * a2) / 9.0L;
      const long double a1_cu = a1_sq * a1;
      const long double R = (2.0L * a1_cu - 9.0L * a1 * a2 + 27.0L * a3) / 54.0L;

      const long double Q_sq = Q * Q;
      const long double Q_cu = Q_sq * Q;
      const long double R_sq = R * R;
      const long double R2_Q3_ld = R_sq - Q_cu;

      const double R2_Q3 = (double)R2_Q3_ld;
      const double a1_div_3 = (double)(a1 / 3.0L);

      double theta = 0.0;

      if (R2_Q3 <= 0.0)
      {
            *solutions = 3;

            const long double sqrt_Q_ld = sqrtl(Q);
            const long double Q_cu_sqrt_ld = sqrtl(Q_cu);
            const long double R_over_Q_cu_sqrt_ld = R / Q_cu_sqrt_ld;

            /* Clamp argument of acos to [-1,1] to avoid NaN due to rounding */
            long double acos_arg = R_over_Q_cu_sqrt_ld;
            if (acos_arg > 1.0L)  acos_arg = 1.0L;
            else if (acos_arg < -1.0L) acos_arg = -1.0L;

            theta = (double)acosl(acos_arg);

            const double sqrt_Q = (double)sqrt_Q_ld;
            const double two_sqrt_Q = -2.0 * sqrt_Q;
            const double theta_div_3 = theta / 3.0;
            const double two_pi_over_3 = 2.0 * PI / 3.0;

            const double base_angle0 = theta_div_3;
            const double base_angle1 = theta_div_3 + two_pi_over_3;
            const double base_angle2 = theta_div_3 + 2.0 * two_pi_over_3;

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(x, two_sqrt_Q, a1_div_3, base_angle0, base_angle1, base_angle2)
#endif
            for (int i = 0; i < 3; ++i)
            {
                  double angle;
                  if (i == 0)
                        angle = base_angle0;
                  else if (i == 1)
                        angle = base_angle1;
                  else
                        angle = base_angle2;

                  x[i] = two_sqrt_Q * cos(angle) - a1_div_3;
            }
      }
      else
      {
            *solutions = 1;

            const double sqrt_R2_Q3 = sqrt(R2_Q3);
            const long double abs_R_ld = fabsl(R);
            const double abs_R = (double)abs_R_ld;

            double tmp = sqrt_R2_Q3 + abs_R;
            double cbrt_val = cbrt(tmp);

            const long double Q_over_cbrt = Q / (long double)cbrt_val;
            double Q_over_cbrt_d = (double)Q_over_cbrt;

            double root = cbrt_val + Q_over_cbrt_d;

            root *= (R < 0.0L) ? 1.0 : -1.0;
            root -= a1_div_3;

            x[0] = root;
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
