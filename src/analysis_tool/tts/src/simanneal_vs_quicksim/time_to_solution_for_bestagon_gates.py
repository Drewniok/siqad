import os
import time

import numpy as np
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
    folder_path = os.path.join(os.getcwd(), "../../resources/bestagon_gates")
    folder_file = os.path.join(folder_path, f"{gate_name}.sqd")
    print(f"Reading layout file: {folder_file}")
    return read_sqd_layout_100(folder_file)


def run_quickexact_simulation(layout, physical_params):
    quickexact_params_inst = quickexact_params()
    quickexact_params_inst.simulation_parameters = physical_params
    quickexact_params_inst.base_number_detection = automatic_base_number_detection.OFF
    result_quickexact = quickexact(layout, quickexact_params_inst)
    return result_quickexact


def run_simanneal_simulation(sp, layout, physical_parameters, layout_coordinates_angstrom, tts_params):
    sp.set_db_locs(layout_coordinates_angstrom)
    all_sa_solution = []

    for _ in range(tts_params.repetitions):
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
                cds_solution.assign_charge_state_by_index(c, sign_to_charge_state(bit))
                cds_solution.update_after_charge_change()
            all_cds_solutions.append(cds_solution)

        pyfiction_simanneal_results.charge_distributions = all_cds_solutions
        all_sa_solution.append(pyfiction_simanneal_results)

    return all_sa_solution


def calculate_tts_simanneal(result_quickexact, all_sa_solution, tts_params):
    st = time_to_solution_stats()
    time_to_solution_for_given_simulation_results(result_quickexact, all_sa_solution, tts_params.confidence_level, st)
    return st.time_to_solution, st.acc


def calculate_tts_quicksim(result_quickexact, quicksim_solution, tts_params):
    st = time_to_solution_stats()
    time_to_solution_for_given_simulation_results(result_quickexact, quicksim_solution, tts_params.confidence_level, st)
    return st.time_to_solution, st.acc


def run_grid_search_simanneal(sp, physical_parameters, quickexact_results, layout, layout_coordinates_angstrom,
                              param_grid, tts_params):
    results = []

    for anneal_cycles in param_grid['anneal_cycles']:
        sp.anneal_cycles = anneal_cycles
        all_sa_solution = run_simanneal_simulation(sp, layout, physical_parameters,
                                                   layout_coordinates_angstrom, tts_params)
        tts_value, acc_value = calculate_tts_simanneal(quickexact_results, all_sa_solution, tts_params)
        results.append({
            'anneal_cycles': anneal_cycles,
            'tts': tts_value,
            'acc': acc_value
        })

    return results


def enumerate_inputs_and_run_simulation(physical_parameters, sp, layout, layout_coordinates_angstrom, param_grid,
                                        tts_params):
    simanneal_results = {ac: {'tts': 0, 'acc': []} for ac in param_grid['anneal_cycles']}
    quicksim_tts = 0
    quicksim_acc = []

    cds = charge_distribution_surface_100(layout)
    all_positions_nm = cds.get_all_sidb_locations_in_nm()
    layout_coordinates_angstrom = [[pos[0] * 10, pos[1] * 10] for pos in all_positions_nm]

    result_quickexact = run_quickexact_simulation(layout, physical_parameters)

    # Run for no-input case
    grid_results = run_grid_search_simanneal(sp, physical_parameters, result_quickexact, layout,
                                             layout_coordinates_angstrom, param_grid, tts_params)

    quicksim_params_inst = quicksim_params()
    quicksim_params_inst.number_threads = sp.num_instances
    quicksim_params_inst.simulation_parameters = physical_parameters
    quicksim_params_inst.alpha = 0.7
    quicksim_params_inst.iteration_steps = 80

    quicksim_solution = []
    for _ in range(tts_params.repetitions):
        result = quicksim(layout, quicksim_params_inst)
        if result is None:
            quicksim_solution.append(sidb_simulation_result_100())
        else:
            quicksim_solution.append(result)

    quicksim_tts_value, quicksim_acc_value = calculate_tts_quicksim(result_quickexact, quicksim_solution, tts_params)

    for result in grid_results:
        simanneal_results[result['anneal_cycles']]['tts'] += result['tts']
        simanneal_results[result['anneal_cycles']]['acc'].append(result['acc'])
    quicksim_tts += quicksim_tts_value
    quicksim_acc.append(quicksim_acc_value)

    # Iterate over input patterns
    bii = bdl_input_iterator_100(layout)
    number_input_patterns = bii.num_input_pairs() ** 2

    for i in range(number_input_patterns):
        layout = bii.get_layout()
        cds = charge_distribution_surface_100(layout)
        all_positions_nm = cds.get_all_sidb_locations_in_nm()
        layout_coordinates_angstrom = [[pos[0] * 10, pos[1] * 10] for pos in all_positions_nm]

        result_quickexact = run_quickexact_simulation(layout, physical_parameters)

        grid_results = run_grid_search_simanneal(sp, physical_parameters, result_quickexact, layout,
                                                 layout_coordinates_angstrom, param_grid, tts_params)

        quicksim_solution = []
        for num_inter in range(tts_params.repetitions):
            quicksim_solution.append(quicksim(layout, quicksim_params_inst))

        quicksim_tts_value, quicksim_acc_value = calculate_tts_quicksim(result_quickexact, quicksim_solution,
                                                                        tts_params)

        for result in grid_results:
            simanneal_results[result['anneal_cycles']]['tts'] += result['tts']
            simanneal_results[result['anneal_cycles']]['acc'].append(result['acc'])
        quicksim_tts += quicksim_tts_value
        quicksim_acc.append(quicksim_acc_value)

        if i == number_input_patterns - 1:
            break
        bii.__next__()

    # Compute average accuracies
    simanneal_summary = {ac: {'total_tts': data['tts'], 'avg_acc': np.mean(data['acc'])} for ac, data in
                         simanneal_results.items()}
    quicksim_summary = {'total_tts': quicksim_tts, 'avg_acc': np.mean(quicksim_acc)}

    return simanneal_summary, quicksim_summary


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
        'anneal_cycles': [1000, 10000]  # Updated to include two annealing cycles
    }

    tts_params = time_to_solution_params()
    tts_params.repetitions = 10000

    final_simanneal_tts = {ac: 0 for ac in param_grid['anneal_cycles']}
    final_simanneal_acc = {ac: [] for ac in param_grid['anneal_cycles']}
    final_quicksim_tts = 0
    final_quicksim_acc = []

    for gate, _ in gates:
        print(f"\nProcessing gate: {gate}")
        layout = read_layout(gate)

        simanneal_summary, quicksim_summary = enumerate_inputs_and_run_simulation(
            physical_parameters, sp, layout, None, param_grid, tts_params
        )

        # Aggregate results
        for ac in param_grid['anneal_cycles']:
            final_simanneal_tts[ac] += simanneal_summary[ac]['total_tts']
            final_simanneal_acc[ac].append(simanneal_summary[ac]['avg_acc'])
            print(
                f"Gate '{gate}' SimAnneal (anneal_cycles={ac}): Total TTS = {simanneal_summary[ac]['total_tts']:.4f} seconds, "
                f"Average Accuracy = {simanneal_summary[ac]['avg_acc']:.4f}")
        final_quicksim_tts += quicksim_summary['total_tts']
        final_quicksim_acc.append(quicksim_summary['avg_acc'])
        print(f"Gate '{gate}' QuickSim: Total TTS = {quicksim_summary['total_tts']:.4f} seconds, "
              f"Average Accuracy = {quicksim_summary['avg_acc']:.4f}")

    # Final summary
    print("\nFinal Summary Across All Gates:")
    for ac in param_grid['anneal_cycles']:
        final_avg_acc = np.mean(final_simanneal_acc[ac])
        print(f"SimAnneal (anneal_cycles={ac}): Total TTS = {final_simanneal_tts[ac]:.4f} seconds, "
              f"Average Accuracy = {final_avg_acc:.4f}")
    final_avg_quicksim_acc = np.mean(final_quicksim_acc)
    print(f"QuickSim: Total TTS = {final_quicksim_tts:.4f} seconds, Average Accuracy = {final_avg_quicksim_acc:.4f}")


if __name__ == "__main__":
    main()
