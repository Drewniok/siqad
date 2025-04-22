import numpy as np
import time
from mnt.pyfiction import *  # Ensure this import is correct
from datetime import datetime
import os
import glob

import pyegs.exhaustive_gs as egs

# Function to generate simulation parameters
def initialize_simulation(mu=-0.32, lambda_tf=5.0, epsilon_r=5.6, num_instances=1):
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
    sp.num_instances = 1
    sp.mu = mu
    sp.num_instances = num_instances

    return physical_parameters, sp



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


# Main function for hyperparameter tuning
# Function for hyperparameter tuning with TTS threshold
def run_tts_measurement(physical_parameters, sp, layout, layout_coordinates_angstrom, num_simulations):
    """
    Performs hyperparameter tuning by running simulations with different parameter values.
    Aborts if TTS exceeds a given threshold.

    Parameters:
    - physical_parameters: Physical parameters for the simulation
    - sp: SimAnneal parameters
    - param1: Name of the first parameter to tune
    - param1_values: List of values for the first parameter
    - param2: Name of the second parameter to tune
    - param2_values: List of values for the second parameter
    - layout: SiDB layout
    - layout_coordinates_angstrom: Coordinates of the layout in Angstroms
    - num_simulations: Number of SimAnneal simulations to determine TTS
    - tts_threshold: Threshold value for TTS to abort the simulation
    - plot_heatmap: Boolean flag to plot heatmap
    - plot_3d: Boolean flag to plot 3D bar plot
    """
    result_quickexact = run_quickexact_simulation(layout, physical_parameters)

    gs_results = groundstate_from_simulation_result(result_quickexact)

    gs = gs_results[0]
    print(gs)

    # Run the SimAnneal simulation
    all_sa_solution = run_simanneal_simulation(sp, layout, physical_parameters, layout_coordinates_angstrom, num_simulations)

    # Calculate TTS for the current (param1, param2) combination
    tts_value = calculate_tts(result_quickexact, all_sa_solution)

    return tts_value


def main():
    """
    Main function to set up parameters, generate layout, and perform hyperparameter tuning.
    """
    # Set up layout parameters and physical properties

    physical_parameters, sp = initialize_simulation()


    # Generate layout parameters
    generate_params = generate_random_sidb_layout_params()
    generate_params.number_of_sidbs = 10
    generate_params.positive_sidbs = positive_charges.FORBIDDEN
    generate_params.coordinate_pair = ((0, 0), (20, 20))
    generate_params.number_of_unique_generated_layouts = 15

    layouts = generate_multiple_random_sidb_layouts(sidb_100_lattice(), generate_params)

    for lyt in layouts:
        cds = charge_distribution_surface_100(lyt)

        # Convert coordinates from nm to angstroms
        all_positions_nm = cds.get_all_sidb_locations_in_nm()

        layout_coordinates_angstrom = [[pos[0] * 10, pos[1] * 10] for pos in all_positions_nm]

        exgs_result = run_exgs_simulation(sp, layout_coordinates_angstrom)
        quickexact_result = run_quickexact_simulation(lyt, physical_parameters)

        print(exgs_result.simulation_runtime.total_seconds())
        print(quickexact_result.simulation_runtime.total_seconds())


if __name__ == "__main__":
    main()
