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

      /* Use long double for improved precision and fewer casts */
      const long double inv_a = 1.0L / (long double)a;
      const long double a1 = (long double)b * inv_a;
      const long double a2 = (long double)c * inv_a;
      const long double a3 = (long double)d * inv_a;

      const long double a1_sq = a1 * a1;
      const long double Q = (a1_sq - 3.0L * a2) / 9.0L;
      const long double numR = 2.0L * a1_sq * a1 - 9.0L * a1 * a2 + 27.0L * a3;
      const long double R = numR / 54.0L;
      const long double Q2 = Q * Q;
      const long double Q3 = Q2 * Q;
      const long double R2 = R * R;
      const long double R2_Q3_ld = R2 - Q3;
      const double R2_Q3 = (double)R2_Q3_ld;

      double theta;

      if (R2_Q3 <= 0.0)
      {
            *solutions = 3;

            const long double sqrtQ = sqrtl(Q);
            const long double denom_a1 = a1 / 3.0L;

            theta = acosl(R / sqrtl(Q3));
            const long double theta_ld = (long double)theta;
            const long double twoPI   = 2.0L * (long double)PI;
            const long double fourPI  = 4.0L * (long double)PI;

            const long double common = -2.0L * sqrtQ;

            /* Compute cos arguments in parallel (independent evaluations) */
            long double c0, c1, c2;
            #ifdef _OPENMP
            #pragma omp parallel sections
            {
                  #pragma omp section
                  { c0 = cosl(theta_ld / 3.0L); }
                  #pragma omp section
                  { c1 = cosl((theta_ld + twoPI) / 3.0L); }
                  #pragma omp section
                  { c2 = cosl((theta_ld + fourPI) / 3.0L); }
            }
            #else
            c0 = cosl(theta_ld / 3.0L);
            c1 = cosl((theta_ld + twoPI) / 3.0L);
            c2 = cosl((theta_ld + fourPI) / 3.0L);
            #endif

            const long double shift = denom_a1;

            x[0] = (double)(common * c0 - shift);
            x[1] = (double)(common * c1 - shift);
            x[2] = (double)(common * c2 - shift);
      }
      else
      {
            *solutions = 1;

            const long double absR = fabsl(R);
            const long double sqrtR2_Q3 = sqrtl(R2_Q3_ld);
            long double x0 = powl(sqrtR2_Q3 + absR, 1.0L / 3.0L);
            x0 += Q / x0;

            if (R < 0.0L)
                  x0 = +x0;
            else
                  x0 = -x0;

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
