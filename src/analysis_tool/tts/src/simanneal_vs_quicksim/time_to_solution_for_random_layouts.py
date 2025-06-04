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

def read_layout(file_path):
    print(f"Reading layout file: {file_path}")
    return read_sqd_layout_100(file_path)

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

def main():
    # Initialize simulation parameters
    physical_parameters, sp = initialize_simulation()

    # Define parameter grid and TTS parameters
    param_grid = {
        'anneal_cycles': [1000, 10000]
    }
    tts_params = time_to_solution_params()
    tts_params.repetitions = 100

    # Directory containing gate layout files
    gate_dir = "../../resources/2_in_1_out/sqd"
    layout_files = [f for f in os.listdir(gate_dir) if f.endswith('.sqd')]

    final_simanneal_tts = {ac: 0 for ac in param_grid['anneal_cycles']}
    final_simanneal_acc = {ac: [] for ac in param_grid['anneal_cycles']}
    final_quicksim_tts = 0
    final_quicksim_acc = []

    for layout_file in layout_files:
        gate_name = os.path.splitext(layout_file)[0]
        print(f"\nProcessing gate: {gate_name}")

        # Load layout
        layout_path = os.path.join(gate_dir, layout_file)
        layout = read_layout(layout_path)

        cds = charge_distribution_surface_100(layout)
        all_positions_nm = cds.get_all_sidb_locations_in_nm()
        layout_coordinates_angstrom = [[pos[0] * 10, pos[1] * 10] for pos in all_positions_nm]

        # Run QuickExact simulation
        result_quickexact = run_quickexact_simulation(layout, physical_parameters)

        # Run SimAnneal grid search
        grid_results = run_grid_search_simanneal(sp, physical_parameters, result_quickexact, layout,
                                                 layout_coordinates_angstrom, param_grid, tts_params)

        # Run QuickSim
        quicksim_params_inst = quicksim_params()
        quicksim_params_inst.number_threads = sp.num_instances
        quicksim_params_inst.simulation_parameters = physical_parameters
        quicksim_params_inst.alpha = 0.7
        quicksim_params_inst.iteration_steps = 80

        quicksim_solution = []
        for _ in range(tts_params.repetitions):
            result = quicksim(layout, quicksim_params_inst)
            if result is not None:
                quicksim_solution.append(result)

        #print(len(quicksim_solution))

        quicksim_tts_value, quicksim_acc_value = calculate_tts_quicksim(result_quickexact, quicksim_solution, tts_params)

        # Aggregate results
        for result in grid_results:
            ac = result['anneal_cycles']
            final_simanneal_tts[ac] += result['tts']
            final_simanneal_acc[ac].append(result['acc'])
            print(f"Gate '{gate_name}' SimAnneal (anneal_cycles={ac}): Total TTS = {result['tts']:.4f} seconds, "
                  f"Average Accuracy = {result['acc']:.4f}")

        final_quicksim_tts += quicksim_tts_value
        final_quicksim_acc.append(quicksim_acc_value)
        print(f"Gate '{gate_name}' QuickSim: Total TTS = {quicksim_tts_value:.4f} seconds, "
              f"Average Accuracy = {quicksim_acc_value:.4f}")

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
