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

      double    a1 = b/a, a2 = c/a, a3 = d/a;
      double    a1_sq = a1*a1;
      double    Q = (a1_sq - 3.0*a2)/9.0;
      double    R = (2.0*a1_sq*a1 - 9.0*a1*a2 + 27.0*a3)/54.0;
      double    R2_Q3 = R*R - Q*Q*Q;

      double    theta;

      if (R2_Q3 <= 0)
      {
            *solutions = 3;
            double sqrt_Q = sqrt(Q);
            double Q_cubed_sqrt = sqrt_Q * Q;
            theta = acos(R/Q_cubed_sqrt);
            double two_sqrt_Q = -2.0*sqrt_Q;
            double a1_third = a1/3.0;
            double theta_third = theta/3.0;
            x[0] = two_sqrt_Q*cos(theta_third) - a1_third;
            x[1] = two_sqrt_Q*cos((theta+2.0*PI)/3.0) - a1_third;
            x[2] = two_sqrt_Q*cos((theta+4.0*PI)/3.0) - a1_third;
      }
      else
      {
            *solutions = 1;
            double sqrt_R2_Q3 = sqrt(R2_Q3);
            double abs_R = fabs(R);
            x[0] = pow(sqrt_R2_Q3+abs_R, 1.0/3.0);
            x[0] += Q/x[0];
            x[0] *= (R < 0.0) ? 1.0 : -1.0;
            x[0] -= a1/3.0;
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
