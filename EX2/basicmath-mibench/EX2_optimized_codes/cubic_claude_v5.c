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

      long double    a1 = b/a, a2 = c/a, a3 = d/a;
      long double    Q = (a1*a1 - 3.0*a2)/9.0;
      long double    a1_sq = a1*a1;
      long double    R = (2.0*a1_sq*a1 - 9.0*a1*a2 + 27.0*a3)/54.0;
      long double    Q_cubed = Q*Q*Q;
      double         R2_Q3 = R*R - Q_cubed;

      int            sol;
      double         x0, x1, x2;

      if (R2_Q3 <= 0)
      {
            sol = 3;
            long double sqrt_Q = sqrtl(Q);
            long double sqrt_Q3 = sqrt_Q * Q;
            double theta = acos(R/sqrt_Q3);
            double theta_div3 = theta/3.0;
            double neg_2sqrt_Q = -2.0*sqrt_Q;
            double a1_div3 = a1/3.0;
            
            x0 = neg_2sqrt_Q*cos(theta_div3) - a1_div3;
            x1 = neg_2sqrt_Q*cos((theta+2.0*PI)/3.0) - a1_div3;
            x2 = neg_2sqrt_Q*cos((theta+4.0*PI)/3.0) - a1_div3;
      }
      else
      {
            sol = 1;
            long double abs_R = fabsl(R);
            long double sqrt_R2_Q3 = sqrtl(R2_Q3);
            x0 = cbrt(sqrt_R2_Q3 + abs_R);
            x0 += Q/x0;
            x0 *= (R < 0.0) ? 1.0 : -1.0;
            x0 -= a1/3.0;
            x1 = 0.0;
            x2 = 0.0;
      }

      *solutions = sol;
      x[0] = x0;
      x[1] = x1;
      x[2] = x2;

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
