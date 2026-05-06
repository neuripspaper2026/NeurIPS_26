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
      double    sqrt_Q, cos_theta_3, cos_theta_2pi_3, cos_theta_4pi_3;
      double    sqrt_R2_Q3, fabs_R, temp1, temp2;
      int       sol_count;
      double    x0, x1, x2;

      if (R2_Q3 <= 0)
      {
            sol_count = 3;
            sqrt_Q = sqrt(Q);
            theta = acos(R/(sqrt_Q*Q));
            cos_theta_3 = cos(theta/3.0);
            cos_theta_2pi_3 = cos((theta+2.0*PI)/3.0);
            cos_theta_4pi_3 = cos((theta+4.0*PI)/3.0);
            
            x0 = -2.0*sqrt_Q*cos_theta_3 - a1/3.0;
            x1 = -2.0*sqrt_Q*cos_theta_2pi_3 - a1/3.0;
            x2 = -2.0*sqrt_Q*cos_theta_4pi_3 - a1/3.0;
      }
      else
      {
            sol_count = 1;
            sqrt_R2_Q3 = sqrt(R2_Q3);
            fabs_R = fabsl(R);
            temp1 = pow(sqrt_R2_Q3+fabs_R, 1/3.0);
            temp2 = Q/temp1;
            x0 = temp1 + temp2;
            x0 *= (R < 0.0) ? 1 : -1;
            x0 -= a1/3.0;
      }
      
      *solutions = sol_count;
      if(sol_count == 3) {
            x[0] = x0;
            x[1] = x1;
            x[2] = x2;
      } else {
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
