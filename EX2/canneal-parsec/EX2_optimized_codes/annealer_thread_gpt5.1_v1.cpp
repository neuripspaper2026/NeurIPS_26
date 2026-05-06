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
	int accepted_bad_moves  = -1;
	double T                = static_cast<double>(_start_temp);
	Rng rng; // store of randomness

	long a_id;
	long b_id;

	netlist_elem* a = _netlist->get_random_element(&a_id, NO_MATCHING_ELEMENT, &rng);
	netlist_elem* b = _netlist->get_random_element(&b_id, NO_MATCHING_ELEMENT, &rng);

	int temp_steps_completed = 0;

	struct timespec kernel_start, kernel_end;
	clock_gettime(CLOCK_MONOTONIC, &kernel_start);

	while (keep_going(temp_steps_completed, accepted_good_moves, accepted_bad_moves)) {
		// cooling schedule
		T *= (1.0 / 1.5);

		accepted_good_moves = 0;
		accepted_bad_moves  = 0;

		// Cache values used in loop to help optimizer
		const int moves_per_temp = _moves_per_thread_temp;
		double    localT         = T;

#if defined(_OPENMP)
		// Parallelize independent move attempts
		#pragma omp parallel
		{
			Rng thread_rng; // private RNG per thread
			int thread_good_moves = 0;
			int thread_bad_moves  = 0;

			// Each thread keeps its own working copy of (a, b, ids)
			long           ta_id;
			long           tb_id;
			netlist_elem*  ta = _netlist->get_random_element(&ta_id, NO_MATCHING_ELEMENT, &thread_rng);
			netlist_elem*  tb = _netlist->get_random_element(&tb_id, NO_MATCHING_ELEMENT, &thread_rng);

			#pragma omp for schedule(static)
			for (int i = 0; i < moves_per_temp; ++i) {
				// get a new element; reuse one to increase locality
				ta   = tb;
				ta_id = tb_id;
				tb   = _netlist->get_random_element(&tb_id, ta_id, &thread_rng);

				const routing_cost_t delta_cost = calculate_delta_routing_cost(ta, tb);
				const move_decision_t decision  = accept_move(delta_cost, localT, &thread_rng);

				if (decision == move_decision_accepted_good) {
					++thread_good_moves;
					_netlist->swap_locations(ta, tb);
				} else if (decision == move_decision_accepted_bad) {
					++thread_bad_moves;
					_netlist->swap_locations(ta, tb);
				}
				// rejected move: nothing to do
			}

			#pragma omp atomic
			accepted_good_moves += thread_good_moves;
			#pragma omp atomic
			accepted_bad_moves  += thread_bad_moves;
		}
#else
		for (int i = 0; i < moves_per_temp; ++i) {
			// get a new element. Only get one new element, so that reuse should help the cache
			a    = b;
			a_id = b_id;
			b    = _netlist->get_random_element(&b_id, a_id, &rng);

			const routing_cost_t  delta_cost = calculate_delta_routing_cost(a, b);
			const move_decision_t decision   = accept_move(delta_cost, localT, &rng);

			// make the move, and update stats:
			if (decision == move_decision_accepted_good) {
				++accepted_good_moves;
				_netlist->swap_locations(a, b);
			} else if (decision == move_decision_accepted_bad) {
				++accepted_bad_moves;
				_netlist->swap_locations(a, b);
			}
			// rejected move: nothing to do
		}
#endif
		++temp_steps_completed;
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

