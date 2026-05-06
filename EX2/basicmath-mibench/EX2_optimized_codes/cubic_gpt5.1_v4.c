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

      /* Precompute reciprocal to avoid repeated division */
      const long double inv_a = 1.0L / (long double)a;

      const long double a1 = (long double)b * inv_a;
      const long double a2 = (long double)c * inv_a;
      const long double a3 = (long double)d * inv_a;

      const long double a1_sq = a1 * a1;
      const long double Q = (a1_sq - 3.0L * a2) / 9.0L;
      const long double a1_cu = a1_sq * a1;
      const long double R = (2.0L * a1_cu - 9.0L * a1 * a2 + 27.0L * a3) / 54.0L;

      const long double Q_sq = Q * Q;
      const long double Q_cu = Q_sq * Q;
      const long double R2_Q3_ld = R * R - Q_cu;
      const double R2_Q3 = (double)R2_Q3_ld;

      const double a1_div_3 = (double)a1 / 3.0;

      double theta;

      if (R2_Q3 <= 0.0)
      {
            *solutions = 3;

            const double Q_sqrt = sqrt((double)Q);
            const double Q_sqrt2 = -2.0 * Q_sqrt;

            theta = acos((double)(R / sqrtl(Q_cu)));

#if defined(_OPENMP)
            /* Parallelize independent cosine evaluations */
            #pragma omp parallel
            {
                  #pragma omp single nowait
                  {
                        x[0] = Q_sqrt2 * cos(theta / 3.0) - a1_div_3;
                  }
                  #pragma omp single nowait
                  {
                        x[1] = Q_sqrt2 * cos((theta + 2.0 * PI) / 3.0) - a1_div_3;
                  }
                  #pragma omp single nowait
                  {
                        x[2] = Q_sqrt2 * cos((theta + 4.0 * PI) / 3.0) - a1_div_3;
                  }
            }
#else
            x[0] = Q_sqrt2 * cos(theta / 3.0) - a1_div_3;
            x[1] = Q_sqrt2 * cos((theta + 2.0 * PI) / 3.0) - a1_div_3;
            x[2] = Q_sqrt2 * cos((theta + 4.0 * PI) / 3.0) - a1_div_3;
#endif
      }
      else
      {
            *solutions = 1;

            const long double R_abs = fabsl(R);
            const long double base = sqrtl(R2_Q3_ld) + R_abs;
            double x0 = pow((double)base, 1.0 / 3.0);

            x0 += (double)(Q / (long double)x0);
            x0 *= (R < 0.0L) ? 1.0 : -1.0;
            x0 -= a1_div_3;
            x[0] = x0;
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
