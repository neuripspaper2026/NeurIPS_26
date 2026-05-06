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

      long double    a1 = b/a, a2 = c/a, a3 = d/a;
      long double    Q = (a1*a1 - 3.0L*a2)/9.0L;
      long double    R = (2.0L*a1*a1*a1 - 9.0L*a1*a2 + 27.0L*a3)/54.0L;
      long double    R2_Q3 = R*R - Q*Q*Q;

      long double    sqrt_Q, theta;

      if (R2_Q3 <= 0.0L)
      {
            *solutions = 3;
            sqrt_Q = sqrtl(Q);
            theta = acosl(R/(sqrt_Q*Q));
            x[0] = -2.0L*sqrt_Q*cosl(theta/3.0L) - a1/3.0L;
            x[1] = -2.0L*sqrt_Q*cosl((theta+2.0L*PI)/3.0L) - a1/3.0L;
            x[2] = -2.0L*sqrt_Q*cosl((theta+4.0L*PI)/3.0L) - a1/3.0L;
      }
      else
      {
            *solutions = 1;
            long double A = powl(fabsl(sqrtl(R2_Q3)+fabsl(R)), 1.0L/3.0L);
            long double B = Q/A;
            x[0] = (R < 0.0L) ? A + B : -A - B;
            x[0] -= a1/3.0L;
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
