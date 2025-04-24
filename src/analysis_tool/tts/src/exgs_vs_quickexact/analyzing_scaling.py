import numpy as np
import time
from mnt.pyfiction import *  # Ensure this import is correct
from datetime import datetime
import os
import glob
import csv

import pyegs.exhaustive_gs as egs

# Function to generate simulation parameters
def initialize_simulation(mu=-0.32, lambda_tf=5.0, epsilon_r=5.6, num_threads=1):
    """
    Initializes physical and SimAnneal simulation parameters.

    Parameters:
    - mu: Chemical potential
    - lambda_tf: Thermal length
    - epsilon_r: Relative permittivity
    - num_instances: Number of instances for SimAnneal

    Returns:
    - physical_parameters: Configured physical parameters for the simulation
    - sp: Configured SimAnneal parameters
    """
    physical_parameters = sidb_simulation_parameters()
    physical_parameters.base = 2
    physical_parameters.mu_minus = mu
    physical_parameters.lambda_tf = lambda_tf
    physical_parameters.epsilon_r = epsilon_r

    sp = egs.SimParams()
    sp.base = 2
    sp.mu = mu
    sp.num_threads = num_threads

    return physical_parameters, sp

def time_to_solution_quicksim(layout, physical_parameters):

    quicksim_param = quicksim_params()
    quicksim_param.number_threads = 1
    quicksim_param.simulation_parameters = physical_parameters
    quicksim_param.alpha = 0.4
    quicksim_param.iteration_steps = 3000

    tts_stats_quicksim = time_to_solution_stats()
    tts_params = time_to_solution_params()
    tts_params.repetitions = 1000

    time_to_solution(layout, quicksim_param, time_to_solution_params(), tts_stats_quicksim)
    return tts_stats_quicksim.time_to_solution

# Function to run QuickExact simulation
def run_quickexact_simulation(layout, physical_params):
    """
    Runs QuickExact simulation and returns the results.

    Parameters:
    - layout: SiDB layout
    - physical_params: Physical parameters for the simulation

    Returns:
    - result_quickexact: Result from QuickExact simulation
    """
    quickexact_params_inst = quickexact_params()
    quickexact_params_inst.simulation_parameters = physical_params
    quickexact_params_inst.base_number_detection = automatic_base_number_detection.OFF
    result_quickexact = quickexact(layout, quickexact_params_inst)
    return result_quickexact

# Function to run ExGS simulations for a given set of parameters
def run_exgs_simulation(sp, layout_coordinates_angstrom):
    """
    Runs ExGS simulations and collects results.

    Parameters:
    - sp: ExGS parameters
    - layout: SiDB layout
    - physical_parameters: Physical parameters for the simulation
    - layout_coordinates_angstrom: Coordinates of the layout in Angstroms
    - num_simulations: Number of ExGS simulations to run

    Returns:
    - all_sa_solution: List of simulation results
    """
    sp.set_db_locs(layout_coordinates_angstrom)

    egs_eng = egs.EGS(sp)

    start_time = time.time()
    egs_eng.invoke()
    simulation_runtime = time.time() - start_time

    pyfiction_exgs_results = sidb_simulation_result_100()
    pyfiction_exgs_results.algorithm_name = "simanneal"
    pyfiction_exgs_results.simulation_runtime = simulation_runtime

    return pyfiction_exgs_results


def generate_layouts(final_number_of_sidbs=20, x_distance=4, y_distance=4):
    """
    Generates layouts for the simulation with SiDBs arranged on a grid.

    Each layout has an increasing number of SiDBs, from 1 up to final_number_of_sidbs.

    Parameters:
    - final_number_of_sidbs: Maximum number of SiDBs in the final layout.
    - x_distance: Horizontal distance between adjacent SiDBs.
    - y_distance: Vertical distance between adjacent SiDBs.

    Returns:
    - layouts: A list of layouts with SiDBs positioned in a grid pattern.
    """
    layouts = []

    for num_of_sidbs in range(1, final_number_of_sidbs + 1):  # Start with 1 SiDB, go up to final_number_of_sidbs
        layout = sidb_100_lattice()
        sidb_count = 0

        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(num_of_sidbs)))  # Create a square grid or slightly larger

        for row in range(grid_size):
            for col in range(grid_size):
                if sidb_count >= num_of_sidbs:
                    break  # Stop once we've placed the required number of SiDBs
                x = col * x_distance
                y = row * y_distance
                layout.assign_cell_type((x, y), sidb_technology.cell_type.NORMAL)
                sidb_count += 1

            if sidb_count >= num_of_sidbs:
                break

        layouts.append(layout)

    return layouts


def main():
    """
    Main function to set up parameters, generate layout, and perform hyperparameter tuning.
    Writes simulation runtimes to a CSV file.
    """
    # Set up layout parameters and physical properties
    physical_parameters, sp = initialize_simulation()

    # Generate layouts
    layouts = generate_layouts(25, 5, 5)

    # Prepare CSV file
    with open("simulation_runtimes.csv", mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Layout ID", "EXGS Runtime (seconds)", "QuickExact Runtime (seconds)", "QuickSim TTS"])  # Header row

        for layout_id, lyt in enumerate(layouts, start=1):
            print(lyt.num_cells())
            cds = charge_distribution_surface_100(lyt)

            # Convert coordinates from nm to angstroms
            all_positions_nm = cds.get_all_sidb_locations_in_nm()
            layout_coordinates_angstrom = [[pos[0] * 10, pos[1] * 10] for pos in all_positions_nm]

            # Run simulations
            exgs_result = run_exgs_simulation(sp, layout_coordinates_angstrom)
            quickexact_result = run_quickexact_simulation(lyt, physical_parameters)

            # Extract ground state and runtimes
            gs_results = groundstate_from_simulation_result(quickexact_result)
            print(gs_results[0])

            exgs_runtime = exgs_result.simulation_runtime.total_seconds()
            quickexact_runtime = quickexact_result.simulation_runtime.total_seconds()
            tts_quicksim = time_to_solution_quicksim(lyt, physical_parameters)

            print(exgs_runtime)
            print(quickexact_runtime)
            print(tts_quicksim)

            # Write runtimes to CSV
            csv_writer.writerow([layout_id, exgs_runtime, quickexact_runtime, tts_quicksim])



if __name__ == "__main__":
    main()
