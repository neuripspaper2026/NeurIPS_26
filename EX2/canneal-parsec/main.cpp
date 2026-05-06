// main.cpp
//
// Created by Daniel Schwartz-Narbonne on 13/04/07.
// Modified by Christian Bienia
//
// Copyright 2007-2008 Princeton University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.


#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <cstdio>
#include <time.h>

#include "annealer_types.h"
#include "annealer_thread.h"
#include "netlist.h"
#include "rng.h"

using namespace std;

int main (int argc, char * const argv[]) {
        cout << "PARSEC Benchmark Suite" << endl << flush;

        srandom(3);

        int status = 0;
        FILE *timing_file = stderr;
        const char *timing_path = getenv("TIMING_LOG_FILE");
        struct timespec main_start, main_end;
        clock_gettime(CLOCK_MONOTONIC, &main_start);
        if (timing_path && timing_path[0] != '\0') {
                FILE *tmp = fopen(timing_path, "w");
                if (tmp) {
                        timing_file = tmp;
                }
        }
        reset_annealer_run_kernel_time();

        if(argc != 5 && argc != 6) {
                cout << "Usage: " << argv[0] << " NTHREADS NSWAPS TEMP NETLIST [NSTEPS]" << endl;
                status = 1;
        } else {
        //argument 1 is numthreads
        int num_threads = atoi(argv[1]);
        cout << "Threadcount: " << num_threads << endl;
        
        // Note: For OpenMP parallelization, remove this check and add:
        // 1. Add -fopenmp to CXXFLAGS in Makefile
        // 2. Include <omp.h> at the top of this file
        // 3. Add #pragma omp parallel directives in annealer_thread.cpp
        // Currently only single-threaded mode is fully tested
        if (num_threads != 1){
                cout << "Warning: Multi-threading not yet enabled. Running with 1 thread." << endl;
                num_threads = 1;
        }

        //argument 2 is the num moves / temp
        int swaps_per_temp = atoi(argv[2]);
        cout << swaps_per_temp << " swaps per temperature step" << endl;

        //argument 3 is the start temp
        int start_temp =  atoi(argv[3]);
        cout << "start temperature: " << start_temp << endl;

        //argument 4 is the netlist filename
        string filename(argv[4]);
        cout << "netlist filename: " << filename << endl;

        //argument 5 (optional) is the number of temperature steps before termination
        int number_temp_steps = -1;
        if(argc == 6) {
                number_temp_steps = atoi(argv[5]);
                cout << "number of temperature steps: " << number_temp_steps << endl;
        }

        //now that we've read in the commandline, run the program
        netlist my_netlist(filename);

        annealer_thread a_thread(&my_netlist,num_threads,swaps_per_temp,start_temp,number_temp_steps);

        // Get initial routing cost
        routing_cost_t initial_cost = my_netlist.total_routing_cost();
        cout << "Initial routing cost: " << initial_cost << endl << endl;

        a_thread.Run();

        // Get final routing cost
        routing_cost_t final_cost = my_netlist.total_routing_cost();
        routing_cost_t improvement = initial_cost - final_cost;
        double improvement_pct = (improvement * 100.0) / initial_cost;

        cout << endl;
        cout << "========================================" << endl;
        cout << "Routing Optimization Results:" << endl;
        cout << "========================================" << endl;
        cout << "Initial cost: " << initial_cost << endl;
        cout << "Final cost:   " << final_cost << endl;
        cout << "Improvement:  " << improvement << " (" << improvement_pct << "%)" << endl;
        cout << "========================================" << endl;

        status = 0;
        }

timing_cleanup:
        clock_gettime(CLOCK_MONOTONIC, &main_end);
        double kernel_time = get_annealer_run_kernel_time();
        double main_time = (main_end.tv_sec - main_start.tv_sec) +
                           (main_end.tv_nsec - main_start.tv_nsec) / 1e9;
        fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
        fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);
        fflush(timing_file);
        if (timing_file != stderr) {
                fclose(timing_file);
        }
        return status;
}
