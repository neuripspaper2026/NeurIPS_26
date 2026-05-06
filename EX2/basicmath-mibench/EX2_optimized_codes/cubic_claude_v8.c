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
      long double    a1_sq = a1*a1;
      long double    Q = (a1_sq - 3.0*a2)/9.0;
      long double    R = (2.0*a1_sq*a1 - 9.0*a1*a2 + 27.0*a3)/54.0;
      long double    R_sq = R*R;
      long double    Q_cubed = Q*Q*Q;
      double         R2_Q3 = R_sq - Q_cubed;

      double    theta;
      double    a1_div3 = a1/3.0;

      if (R2_Q3 <= 0)
      {
            *solutions = 3;
            long double sqrt_Q = sqrtl(Q);
            long double sqrt_Q_cubed = sqrt_Q*Q;
            theta = acosl(R/sqrt_Q_cubed);
            double neg_2sqrt_Q = -2.0*sqrt_Q;
            double theta_div3 = theta/3.0;
            x[0] = neg_2sqrt_Q*cosl(theta_div3) - a1_div3;
            x[1] = neg_2sqrt_Q*cosl((theta+2.0*PI)/3.0) - a1_div3;
            x[2] = neg_2sqrt_Q*cosl((theta+4.0*PI)/3.0) - a1_div3;
      }
      else
      {
            *solutions = 1;
            long double sqrt_R2_Q3 = sqrtl(R2_Q3);
            long double abs_R = fabsl(R);
            x[0] = powl(sqrt_R2_Q3 + abs_R, 1.0L/3.0L);
            x[0] += Q/x[0];
            x[0] *= (R < 0.0) ? 1 : -1;
            x[0] -= a1_div3;
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
