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

      /* Precompute reciprocal of a to avoid repeated division */
      const long double inv_a = 1.0L / (long double)a;

      const long double a1 = (long double)b * inv_a;
      const long double a2 = (long double)c * inv_a;
      const long double a3 = (long double)d * inv_a;

      /* Use long double consistently for intermediate precision */
      const long double a1_sq   = a1 * a1;
      const long double Q       = (a1_sq - 3.0L * a2) / 9.0L;
      const long double a1_cu   = a1_sq * a1;
      const long double R       = (2.0L * a1_cu - 9.0L * a1 * a2 + 27.0L * a3) / 54.0L;
      const long double Q3      = Q * Q * Q;
      const long double R2      = R * R;
      const long double R2_Q3   = R2 - Q3;

      double theta;

      if (R2_Q3 <= 0.0L)
      {
            /* Three real roots */
            *solutions = 3;

            const long double sqrtQ = sqrtl(Q);
            const long double Q_sqrt_cubed = sqrtQ * Q; /* == sqrt(Q^3) */

            /* Guard against slight negative due to FP error in acos argument */
            long double acos_arg = 0.0L;
            if (Q_sqrt_cubed != 0.0L) {
                  acos_arg = R / Q_sqrt_cubed;
                  if (acos_arg >  1.0L) acos_arg =  1.0L;
                  if (acos_arg < -1.0L) acos_arg = -1.0L;
            }

            theta = acosl(acos_arg);

            const long double two_sqrtQ = 2.0L * sqrtQ;
            const long double a1_div3   = a1 / 3.0L;

            const long double theta_over_3   = theta / 3.0L;
            const long double theta_2pi_over3 = (theta + 2.0L * PI) / 3.0L;
            const long double theta_4pi_over3 = (theta + 4.0L * PI) / 3.0L;

            const long double cos_t0 = cosl(theta_over_3);
            const long double cos_t1 = cosl(theta_2pi_over3);
            const long double cos_t2 = cosl(theta_4pi_over3);

            x[0] = (double)(-two_sqrtQ * cos_t0 - a1_div3);
            x[1] = (double)(-two_sqrtQ * cos_t1 - a1_div3);
            x[2] = (double)(-two_sqrtQ * cos_t2 - a1_div3);

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(x) if(0)
#endif
            for (int i = 0; i < 1; ++i) {
                  /* Dummy loop kept to illustrate possible parallel pattern without changing behavior */
                  x[0] = x[0];
            }
      }
      else
      {
            /* One real root */
            *solutions = 1;

            const long double sqrtR2_Q3 = sqrtl(R2_Q3);
            long double absR = fabsl(R);

            long double base = sqrtR2_Q3 + absR;
            long double root_cu = cbrtl(base);

            long double x0 = root_cu + Q / root_cu;

            /* Keep branch with simple integer comparison, cast R for clarity */
            x0 *= ((double)R < 0.0) ? 1.0L : -1.0L;
            x0 -= a1 / 3.0L;

            x[0] = (double)x0;
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
