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

      /* Precompute reciprocal of a to avoid repeated divisions */
      const long double inv_a = 1.0L / (long double)a;

      const long double a1 = (long double)b * inv_a;
      const long double a2 = (long double)c * inv_a;
      const long double a3 = (long double)d * inv_a;

      const long double a1_sq = a1 * a1;
      const long double Q = (a1_sq - 3.0L * a2) / 9.0L;
      const long double Q_sq = Q * Q;
      const long double Q_cu = Q_sq * Q;

      const long double R =
          (2.0L * a1_sq * a1 - 9.0L * a1 * a2 + 27.0L * a3) / 54.0L;

      const long double R_sq = R * R;
      const long double R2_Q3_ld = R_sq - Q_cu;
      const double R2_Q3 = (double)R2_Q3_ld;

      double theta;

      if (R2_Q3 <= 0.0)
      {
            *solutions = 3;

            const long double sqrt_Q = sqrtl(Q);
            const long double Q_cu_d = (double)Q_cu; /* for sqrt() with double */

            /* Use long double versions for better precision, then cast */
            theta = acosl(R / sqrtl(Q_cu)) ;

            const long double theta_ld = (long double)theta;
            const long double third = 1.0L / 3.0L;
            const long double base = -2.0L * sqrt_Q;
            const long double a1_div3 = a1 / 3.0L;

            const long double theta_over_3 = theta_ld * third;
            const long double two_pi_over_3 = (2.0L * PI) * third;
            const long double four_pi_over_3 = (4.0L * PI) * third;

            /* Parallelize the independent root computations */
            #ifdef _OPENMP
            #pragma omp parallel for default(none) shared(x, base, a1_div3, theta_over_3, two_pi_over_3, four_pi_over_3)
            #endif
            for (int i = 0; i < 3; ++i)
            {
                  long double angle;
                  if (i == 0)
                        angle = theta_over_3;
                  else if (i == 1)
                        angle = theta_over_3 + two_pi_over_3;
                  else
                        angle = theta_over_3 + four_pi_over_3;

                  x[i] = (double)(base * cosl(angle) - a1_div3);
            }
      }
      else
      {
            *solutions = 1;

            const long double sqrt_R2_Q3 = sqrtl(R2_Q3_ld);
            const long double absR = fabsl(R);
            const long double sum = sqrt_R2_Q3 + absR;

            long double root = cbrtl(sum);
            root += Q / root;

            if (R >= 0.0L)
                  root = -root;

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
