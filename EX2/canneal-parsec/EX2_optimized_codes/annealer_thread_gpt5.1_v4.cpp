#include <cassert>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../annealer_thread.h"
#include "../location_t.h"
#include "../annealer_types.h"
#include "../netlist_elem.h"
#include <math.h>
#include "../rng.h"

static double annealer_run_kernel_time_acc = 0.0;

void reset_annealer_run_kernel_time(void) { annealer_run_kernel_time_acc = 0.0; }
double get_annealer_run_kernel_time(void) { return annealer_run_kernel_time_acc; }


//*****************************************************************************************
//
//*****************************************************************************************
void annealer_thread::Run()
{
	int accepted_good_moves = 0;
	int accepted_bad_moves = -1;
	double T = static_cast<double>(_start_temp);
	Rng rng; // store of randomness

	long a_id;
	long b_id;

	netlist_elem* a = _netlist->get_random_element(&a_id, NO_MATCHING_ELEMENT, &rng);
	netlist_elem* b = _netlist->get_random_element(&b_id, NO_MATCHING_ELEMENT, &rng);

	int temp_steps_completed = 0;
	struct timespec kernel_start, kernel_end;
	clock_gettime(CLOCK_MONOTONIC, &kernel_start);

	while (keep_going(temp_steps_completed, accepted_good_moves, accepted_bad_moves)) {
		T /= 1.5;
		accepted_good_moves = 0;
		accepted_bad_moves = 0;

		const int moves_per_temp = _moves_per_thread_temp;

		// Parallelize the swap attempts across iterations using OpenMP if available.
		// Each iteration is independent with respect to RNG and decision making,
		// but swaps operate on shared netlist; we therefore serialize swaps while
		// keeping the rest of the computation parallel.
		#pragma omp parallel if (moves_per_temp > 1)
		{
			Rng local_rng;
			int local_accepted_good = 0;
			int local_accepted_bad = 0;

			long local_a_id = a_id;
			long local_b_id = b_id;
			netlist_elem* local_a = a;
			netlist_elem* local_b = b;

			#pragma omp for schedule(static)
			for (int i = 0; i < moves_per_temp; i++) {
				// Reuse previous 'b' as new 'a'
				local_a = local_b;
				local_a_id = local_b_id;

				local_b = _netlist->get_random_element(&local_b_id, local_a_id, &local_rng);

				const routing_cost_t delta_cost = calculate_delta_routing_cost(local_a, local_b);
				const move_decision_t is_good_move = accept_move(delta_cost, T, &local_rng);

				if (is_good_move == move_decision_accepted_good) {
					// Good moves: update stats and swap
					local_accepted_good++;
					#pragma omp critical
					{
						_netlist->swap_locations(local_a, local_b);
					}
				} else if (is_good_move == move_decision_accepted_bad) {
					// Bad but accepted moves: update stats and swap
					local_accepted_bad++;
					#pragma omp critical
					{
						_netlist->swap_locations(local_a, local_b);
					}
				}
				// Rejected moves require no action
			}

			#pragma omp atomic
			accepted_good_moves += local_accepted_good;
			#pragma omp atomic
			accepted_bad_moves += local_accepted_bad;
		}

		temp_steps_completed++;
	}

	clock_gettime(CLOCK_MONOTONIC, &kernel_end);
	annealer_run_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
	                                (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}

//*****************************************************************************************
//
//*****************************************************************************************
annealer_thread::move_decision_t annealer_thread::accept_move(routing_cost_t delta_cost, double T, Rng* rng)
{
	//always accept moves that lower the cost function
	if (delta_cost < 0){
		return move_decision_accepted_good;
	} else {
		double random_value = rng->drand();
		double boltzman = exp(- delta_cost/T);
		if (boltzman > random_value){
			return move_decision_accepted_bad;
		} else {
			return move_decision_rejected;
		}
	}
}


//*****************************************************************************************
//  If get turns out to be expensive, I can reduce the # by passing it into the swap cost fcn
//*****************************************************************************************
routing_cost_t annealer_thread::calculate_delta_routing_cost(netlist_elem* a, netlist_elem* b)
{
	location_t* a_loc = a->present_loc.Get();
	location_t* b_loc = b->present_loc.Get();
	
	routing_cost_t delta_cost = a->swap_cost(a_loc, b_loc);
	delta_cost += b->swap_cost(b_loc, a_loc);

	return delta_cost;
}

//*****************************************************************************************
//  Check whether design has converged or maximum number of steps has reached
//*****************************************************************************************
bool annealer_thread::keep_going(int temp_steps_completed, int accepted_good_moves, int accepted_bad_moves)
{
	bool rv;

	if(_number_temp_steps == -1) {
		//run until design converges
		rv = _keep_going_global_flag && (accepted_good_moves > accepted_bad_moves);
		if(!rv) _keep_going_global_flag = false; // signal we have converged
	} else {
		//run a fixed amount of steps
		rv = temp_steps_completed < _number_temp_steps;
	}

	return rv;
}

