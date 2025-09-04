import os
import time
import numpy as np
from mnt.pyfiction import *  # Ensure this import is correct
from pysimanneal import simanneal


# --- Simulation Utilities ---
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


def read_layout_file(filepath):
    print(f"Reading layout file: {filepath}")
    return read_sqd_layout_100(filepath)


# --- Simulation Functions ---
def run_quickexact_simulation(layout, physical_params):
    quickexact_params_inst = quickexact_params()
    quickexact_params_inst.simulation_parameters = physical_params
    quickexact_params_inst.base_number_detection = automatic_base_number_detection.OFF
    return quickexact(layout, quickexact_params_inst)


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
        all_sa_solution = run_simanneal_simulation(sp, layout, physical_parameters, layout_coordinates_angstrom, tts_params)
        tts_value, acc_value = calculate_tts_simanneal(quickexact_results, all_sa_solution, tts_params)
        results.append({
            'anneal_cycles': anneal_cycles,
            'tts': tts_value,
            'acc': acc_value
        })
    return results


# --- Main Function for ML_generated ---
def main():
    physical_parameters, sp = initialize_simulation()

    base_folder = os.path.join(os.getcwd(), "../../resources/ML_generated")  # Sibling of bestagon_gates
    gate_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

    param_grid = {'anneal_cycles': [1000, 10000]}  # Annealing cycles
    tts_params = time_to_solution_params()
    tts_params.repetitions = 10000

    overall_sa_tts = {ac: 0 for ac in param_grid['anneal_cycles']}
    overall_sa_acc = {ac: [] for ac in param_grid['anneal_cycles']}
    overall_quicksim_tts = 0
    overall_quicksim_acc = []

    for gate_folder in gate_folders:
        folder_path = os.path.join(base_folder, gate_folder)
        layout_files = [f for f in os.listdir(folder_path) if f.endswith(".sqd")]

        folder_sa_tts = {ac: 0 for ac in param_grid['anneal_cycles']}
        folder_sa_acc = {ac: [] for ac in param_grid['anneal_cycles']}
        folder_quicksim_tts = 0
        folder_quicksim_acc = []

        print(f"\nProcessing gate folder: {gate_folder} ({len(layout_files)} layouts)")

        for layout_file in layout_files:
            layout_path = os.path.join(folder_path, layout_file)
            layout = read_layout_file(layout_path)

            # Prepare layout coordinates
            cds = charge_distribution_surface_100(layout)
            all_positions_nm = cds.get_all_sidb_locations_in_nm()
            layout_coordinates_angstrom = [[pos[0]*10, pos[1]*10] for pos in all_positions_nm]

            # QuickExact
            result_quickexact = run_quickexact_simulation(layout, physical_parameters)

            # SimAnneal
            grid_results = run_grid_search_simanneal(sp, physical_parameters, result_quickexact, layout,
                                                     layout_coordinates_angstrom, param_grid, tts_params)
            for res in grid_results:
                folder_sa_tts[res['anneal_cycles']] += res['tts']
                folder_sa_acc[res['anneal_cycles']].append(res['acc'])

            # QuickSim
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
            tts_val, acc_val = calculate_tts_quicksim(result_quickexact, quicksim_solution, tts_params)
            folder_quicksim_tts += tts_val
            folder_quicksim_acc.append(acc_val)

        # Folder summary
        print(f"Gate folder '{gate_folder}' summary:")
        for ac in param_grid['anneal_cycles']:
            avg_acc = np.mean(folder_sa_acc[ac])
            print(f"  SimAnneal (anneal_cycles={ac}): Total TTS = {folder_sa_tts[ac]:.4f}, Average Accuracy = {avg_acc:.4f}")
            overall_sa_tts[ac] += folder_sa_tts[ac]
            overall_sa_acc[ac].extend(folder_sa_acc[ac])
        avg_quicksim_acc = np.mean(folder_quicksim_acc)
        print(f"  QuickSim: Total TTS = {folder_quicksim_tts:.4f}, Average Accuracy = {avg_quicksim_acc:.4f}")
        overall_quicksim_tts += folder_quicksim_tts
        overall_quicksim_acc.extend(folder_quicksim_acc)

    # Overall summary
    print("\nOverall Summary Across All Gate Folders:")
    for ac in param_grid['anneal_cycles']:
        print(f"SimAnneal (anneal_cycles={ac}): Total TTS = {overall_sa_tts[ac]:.4f}, Overall Average Accuracy = {np.mean(overall_sa_acc[ac]):.4f}")
    print(f"QuickSim: Total TTS = {overall_quicksim_tts:.4f}, Overall Average Accuracy = {np.mean(overall_quicksim_acc):.4f}")


if __name__ == "__main__":
    main()
