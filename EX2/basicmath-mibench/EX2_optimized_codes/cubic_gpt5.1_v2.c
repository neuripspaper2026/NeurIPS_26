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

      /* Precompute reciprocal of 'a' to avoid repeated divisions */
      const long double inv_a = 1.0L / (long double)a;
      const long double a1 = (long double)b * inv_a;
      const long double a2 = (long double)c * inv_a;
      const long double a3 = (long double)d * inv_a;

      const long double a1_sq = a1 * a1;
      const long double Q = (a1_sq - 3.0L * a2) / 9.0L;

      const long double two = 2.0L;
      const long double nine = 9.0L;
      const long double twenty_seven = 27.0L;

      const long double R =
            (two * a1 * a1_sq - nine * a1 * a2 + twenty_seven * a3) / 54.0L;

      const long double Q3 = Q * Q * Q;
      const long double R2 = R * R;
      const long double R2_Q3_ld = R2 - Q3;
      const double R2_Q3 = (double)R2_Q3_ld;

      double theta;

      if (R2_Q3 <= 0.0)
      {
            *solutions = 3;

            const long double sqrtQ = sqrtl(Q);
            const long double QQQ = Q3; /* already Q^3 */
            const long double R_over_sqrtQQQ = R / sqrtl(QQQ);
            theta = acosl(R_over_sqrtQQQ);

            const long double minus_a1_over_3 = -a1 / 3.0L;
            const long double two_sqrtQ = 2.0L * sqrtQ;

            const double base = (double)two_sqrtQ;
            const double shift = (double)minus_a1_over_3;

            /* Parallelize independent root evaluations */
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for (int i = 0; i < 3; ++i) {
                  const double angle = (theta + (double)(2 * i) * PI) / 3.0;
                  x[i] = -base * cos(angle) + shift;
            }
      }
      else
      {
            *solutions = 1;

            const long double sqrt_R2_Q3 = sqrtl(R2_Q3_ld);
            const long double absR = fabsl(R);
            const long double inner = sqrt_R2_Q3 + absR;

            long double root = powl(inner, 1.0L / 3.0L);
            const long double Q_over_root = Q / root;
            root += Q_over_root;

            const long double sign = (R < 0.0L) ? 1.0L : -1.0L;
            root *= sign;
            root -= a1 / 3.0L;

            x[0] = (double)root;
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
