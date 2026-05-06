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
      const long double inv_a = 1.0L / a;
      const long double a1 = b * inv_a;
      const long double a2 = c * inv_a;
      const long double a3 = d * inv_a;

      /* Use long double consistently for intermediate precision */
      const long double a1_sq = a1 * a1;
      const long double Q  = (a1_sq - 3.0L * a2) / 9.0L;
      const long double a1_cu = a1_sq * a1;
      const long double R  = (2.0L * a1_cu - 9.0L * a1 * a2 + 27.0L * a3) / 54.0L;
      const long double Q3 = Q * Q * Q;
      const long double R2 = R * R;
      const long double R2_Q3 = R2 - Q3;

      double theta = 0.0;

      if (R2_Q3 <= 0.0L)
      {
            *solutions = 3;

            /* All three real roots: compute shared terms once */
            const long double sqrtQ   = sqrtl(Q);
            const long double two_sQ  = 2.0L * sqrtQ;
            const long double Q3_sqrt = sqrtl(Q3);
            const long double ratio   = (Q3_sqrt != 0.0L) ? (R / Q3_sqrt) : 0.0L;

            /* Clamp the argument of acos to [-1,1] for numerical robustness */
            long double acos_arg = ratio;
            if (acos_arg > 1.0L)
                  acos_arg = 1.0L;
            else if (acos_arg < -1.0L)
                  acos_arg = -1.0L;

            theta = acosl(acos_arg);

            const long double a1_div3 = a1 / 3.0L;
            const long double theta_div3 = theta / 3.0L;
            const long double twoPI = 2.0L * (long double)PI;

            const long double t0 = theta_div3;
            const long double t1 = (theta + twoPI) / 3.0L;
            const long double t2 = (theta + 2.0L * twoPI) / 3.0L;

            /* Parallelize the three independent root computations */
            #ifdef _OPENMP
            #pragma omp parallel for default(none) shared(x, two_sQ, a1_div3, t0, t1, t2)
            #endif
            for (int i = 0; i < 3; ++i)
            {
                  long double angle;
                  if (i == 0)
                        angle = t0;
                  else if (i == 1)
                        angle = t1;
                  else
                        angle = t2;

                  const long double root = -two_sQ * cosl(angle) - a1_div3;
                  x[i] = (double)root;
            }
      }
      else
      {
            *solutions = 1;

            const long double sqrtR2_Q3 = sqrtl(R2_Q3);
            const long double absR      = fabsl(R);

            /* Single real root: use cbrt for better stability and performance */
            long double temp = sqrtR2_Q3 + absR;
            long double u    = cbrtl(temp);

            /* Avoid division by zero */
            if (u != 0.0L)
            {
                  long double x0 = u + Q / u;
                  if (R >= 0.0L)
                        x0 = -x0;
                  x0 -= a1 / 3.0L;
                  x[0] = (double)x0;
            }
            else
            {
                  /* Fallback: when u == 0, the cubic is near-degenerate */
                  long double x0 = -a1 / 3.0L;
                  x[0] = (double)x0;
            }
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
