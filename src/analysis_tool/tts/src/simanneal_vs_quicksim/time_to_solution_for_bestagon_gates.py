import numpy as np
import os
import time
from datetime import datetime
from mnt.pyfiction import *  # Ensure this import is correct

from pysimanneal import simanneal


# Function to generate simulation parameters
def initialize_simulation(mu=-0.32, lambda_tf=5.0, epsilon_r=5.6):
    physical_parameters = sidb_simulation_parameters()
    physical_parameters.base = 2
    physical_parameters.mu_minus = mu
    physical_parameters.lambda_tf = lambda_tf
    physical_parameters.epsilon_r = epsilon_r

    sp = simanneal.SimParams()
    sp.num_instances = 1
    sp.mu = mu
    sp.debye_length = lambda_tf
    sp.eps_r = epsilon_r
    return physical_parameters, sp


def read_layout(gate_name):
    folder_path = os.path.join(os.getcwd(), "../bestagon_gates")
    folder_file = folder_path + "/" + gate_name
    print(folder_file)
    return read_sqd_layout_100(folder_file)


def run_quickexact_simulation(layout, physical_params):
    quickexact_params_inst = quickexact_params()
    quickexact_params_inst.simulation_parameters = physical_params
    quickexact_params_inst.base_number_detection = automatic_base_number_detection.OFF
    result_quickexact = quickexact(layout, quickexact_params_inst)
    return result_quickexact


def run_simanneal_simulation(sp, layout, physical_parameters, layout_coordinates_angstrom, num_simulations=100):
    sp.set_db_locs(layout_coordinates_angstrom)
    all_sa_solution = []

    for _ in range(num_simulations):
        sa = simanneal.SimAnneal(sp)
        start_time = time.time()
        sa.invokeSimAnneal()
        simulation_runtime_sa = time.time() - start_time

        results = sa.suggested_gs_results()
        pyfiction_simanneal_results = sidb_simulation_result_100()
        pyfiction_simanneal_results.algorithm_name = "simanneal"
        pyfiction_simanneal_results.simulation_runtime = simulation_runtime_sa

        all_cds_solutions = []
        for res in results:
            cds_solution = charge_distribution_surface_100(layout, physical_parameters)
            for c, bit in enumerate(res.config):
                cds_solution.assign_charge_state_by_cell_index(c, sign_to_charge_state(bit))
                cds_solution.update_after_charge_change()
            all_cds_solutions.append(cds_solution)

        pyfiction_simanneal_results.charge_distributions = all_cds_solutions
        all_sa_solution.append(pyfiction_simanneal_results)

    return all_sa_solution


def calculate_tts_simanneal(result_quickexact, all_sa_solution):
    st = time_to_solution_stats()
    time_to_solution_for_given_simulation_results(result_quickexact, all_sa_solution, 0.999, st)
    return st.time_to_solution


def run_grid_search_simanneal(sp, physical_parameters, layout, layout_coordinates_angstrom, param_grid,
                              num_simulations=100):
    best_tts = float('inf')
    best_params = None

    result_quickexact = run_quickexact_simulation(layout, physical_parameters)

    for T_init in param_grid['T_init']:
        for T_min in param_grid['T_min']:
            for alpha in param_grid['alpha']:
                for anneal_cycles in param_grid['anneal_cycles']:
                    sp.T_init = T_init
                    sp.T_min = T_min
                    sp.alpha = alpha
                    sp.anneal_cycles = anneal_cycles

                    all_sa_solution = run_simanneal_simulation(sp, layout, physical_parameters,
                                                               layout_coordinates_angstrom, num_simulations)
                    tts_value = calculate_tts_simanneal(result_quickexact, all_sa_solution)

                    # print(
                    #     f"Params: T_init={T_init}, T_min={T_min}, alpha={alpha}, anneal_cycles={anneal_cycles} => TTS={tts_value:.4f}")

                    if tts_value < best_tts:
                        best_tts = tts_value
                        best_params = {
                            'T_init': T_init,
                            'T_min': T_min,
                            'alpha': alpha,
                            'anneal_cycles': anneal_cycles
                        }

    return best_params, best_tts


def run_grid_search_quicksim(layout, quicksim_params, param_grid):

    best_tts = float('inf')
    best_params = None

    for alpha in param_grid['alpha']:
        for iteration_steps in param_grid['iteration_steps']:
            quicksim_params.alpha = alpha
            quicksim_params.iteration_steps = iteration_steps

            tts_stats_quicksim = time_to_solution_stats()
            time_to_solution(layout, quicksim_params, time_to_solution_params(), tts_stats_quicksim)
            tts_value_quicksim = tts_stats_quicksim.time_to_solution
            #print(f"TTS QuickSim={tts_value_quicksim:.4f}")
            #print(f"Params: alpha={alpha}, iteration_steps={iteration_steps} => TTS={tts_value_quicksim:.4f}")

            if tts_value_quicksim < best_tts:
                best_tts = tts_value_quicksim
                best_params = {
                    'alpha': alpha,
                    'iteration_steps': iteration_steps
                }

    return best_params, best_tts


def enumerate_inputs_and_run_simulation_simanneal(physical_parameters, sp, layout, layout_coordinates_angstrom,
                                                  param_grid, num_simulations=100):
    total_tts = 0
    """
    Enumerates through input patterns and performs simulations.
    """
    cds = charge_distribution_surface_100(layout)
    all_positions_nm = cds.get_all_sidb_locations_in_nm()
    layout_coordinates_angstrom = [[pos[0] * 10, pos[1] * 10] for pos in all_positions_nm]

    best_params, best_tts = run_grid_search_simanneal(sp, physical_parameters, layout, layout_coordinates_angstrom,
                                                      param_grid, num_simulations)

    print(f"Best parameters (no input): {best_params}")
    print(f"Best TTS (no input): {best_tts:.4f} seconds")

    total_tts += best_tts

    bii = bdl_input_iterator_100(layout)
    number_input_patterns = bii.num_input_pairs() ** 2

    for i in range(number_input_patterns):
        layout = bii.get_layout()
        cds = charge_distribution_surface_100(layout)
        all_positions_nm = cds.get_all_sidb_locations_in_nm()
        layout_coordinates_angstrom = [[pos[0] * 10, pos[1] * 10] for pos in all_positions_nm]

        best_params, best_tts = run_grid_search_simanneal(sp, physical_parameters, layout, layout_coordinates_angstrom,
                                                          param_grid, num_simulations)

        print(f"Input pattern {i + 1}/{number_input_patterns}:")
        print(f"Best parameters: {best_params}")
        print(f"Best TTS: {best_tts:.4f} seconds")

        total_tts += best_tts

        if (i == number_input_patterns - 1):
            break
        bii.__next__()

    return total_tts


def enumerate_inputs_and_run_simulation_quicksim(layout, physical_parameters, param_grid_quicksim):
    """
    Enumerates through input patterns and performs simulations.
    """
    quicksim_param = quicksim_params()
    quicksim_param.number_threads = 1
    quicksim_param.simulation_parameters = physical_parameters

    total_tts_quicksim = 0
    best_params, best_tts = run_grid_search_quicksim(layout, quicksim_param, param_grid_quicksim)
    total_tts_quicksim += best_tts

    print(f"Best parameters (no input): {best_params}")
    print(f"Best TTS (no input): {best_tts:.4f} seconds")

    bii = bdl_input_iterator_100(layout)
    number_input_patterns = bii.num_input_pairs() ** 2

    for i in range(number_input_patterns):
        layout = bii.get_layout()

        best_params, best_tts = run_grid_search_quicksim(layout, quicksim_param, param_grid_quicksim)

        print(f"Input pattern {i + 1}/{number_input_patterns}:")
        print(f"Best parameters: {best_params}")
        print(f"Best TTS: {best_tts:.4f} seconds")

        total_tts_quicksim += best_tts

        if (i == number_input_patterns - 1):
            break
        bii.__next__()

    return total_tts_quicksim


def main():
    physical_parameters, sp = initialize_simulation()

    gates = [
        ("and", create_and_tt()),
        ("or", create_or_tt()),
        ("nand", create_nand_tt()),
        ("nor", create_nor_tt()),
        ("xor", create_xor_tt()),
        ("xnor", create_xnor_tt()),
        ("hourglass", create_double_wire_tt()),
        ("cx", create_crossing_wire_tt()),
        ("ha", create_half_adder_tt())
    ]

    param_grid = {
        'T_init': np.linspace(1000, 5000, 5).tolist(),
        'T_min': np.linspace(1, 50, 5).tolist(),
        'alpha': np.linspace(0.6, 0.99, 5).tolist(),
        'anneal_cycles': np.linspace(100, 3000, 5).astype(int).tolist()
    }

    param_grid_quicksim = {
        'alpha': np.linspace(0.6, 0.99, 10).tolist(),
        'iteration_steps': np.linspace(10, 200, 10).astype(int).tolist()
    }

    final_tts_simanneal = 0
    final_tts_quicksim = 0


    for gate, _ in gates:
        print(f"Processing gate: {gate}")
        layout = read_layout(gate + ".sqd")
        cds = charge_distribution_surface_100(layout)
        all_positions_nm = cds.get_all_sidb_locations_in_nm()
        layout_coordinates_angstrom = [[pos[0] * 10, pos[1] * 10] for pos in all_positions_nm]

        total_tts = enumerate_inputs_and_run_simulation_simanneal(physical_parameters, sp, layout,
                                                                  layout_coordinates_angstrom, param_grid,
                                                                  num_simulations=100)
        total_tts_quicksim = enumerate_inputs_and_run_simulation_quicksim(layout, physical_parameters,
                                                                          param_grid_quicksim)

        final_tts_simanneal += total_tts
        final_tts_quicksim += total_tts_quicksim

        print(f"Total Best TTS for gate '{gate}': {total_tts:.4f} seconds")
        print(f"Total TTS QuickSim for gate '{gate}': {total_tts_quicksim:.4f} seconds")

    print(f"FINAL TTS SimAnneal: {final_tts_simanneal:.4f} seconds")
    print(f"FINAL TTS QuickSim: {final_tts_quicksim:.4f} seconds")


if __name__ == "__main__":
    main()
