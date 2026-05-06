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
      long double R = (2.0*a1*a1*a1 - 9.0*a1*a2 + 27.0*a3)/54.0;
      double    R2_Q3 = R*R - Q*Q*Q;

      double    theta;
      int local_solutions;
      double local_x[3];

      if (R2_Q3 <= 0)
      {
            local_solutions = 3;
            theta = acos(R/sqrt(Q*Q*Q));
            local_x[0] = -2.0*sqrt(Q)*cos(theta/3.0) - a1/3.0;
            local_x[1] = -2.0*sqrt(Q)*cos((theta+2.0*PI)/3.0) - a1/3.0;
            local_x[2] = -2.0*sqrt(Q)*cos((theta+4.0*PI)/3.0) - a1/3.0;
      }
      else
      {
            local_solutions = 1;
            local_x[0] = pow(sqrt(R2_Q3)+fabsl(R), 1/3.0);
            local_x[0] += Q/local_x[0];
            local_x[0] *= (R < 0.0) ? 1 : -1;
            local_x[0] -= a1/3.0;
      }

      *solutions = local_solutions;
      x[0] = local_x[0];
      if (local_solutions > 1) {
          x[1] = local_x[1];
          x[2] = local_x[2];
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
