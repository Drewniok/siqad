import numpy as np
import os
import time
from datetime import datetime
import csv
from mnt.pyfiction import *  # Ensure this import is correct
import pyegs.exhaustive_gs as egs

# Initialize simulation parameters
def initialize_simulation(mu=-0.32, lambda_tf=5.0, epsilon_r=5.6):
    """Sets up physical and simulation parameters."""
    physical_params = sidb_simulation_parameters()
    physical_params.base = 2
    physical_params.mu_minus = mu
    physical_params.lambda_tf = lambda_tf
    physical_params.epsilon_r = epsilon_r

    sim_params = egs.SimParams()
    sim_params.base = 2
    sim_params.num_instances = 1
    sim_params.mu = mu
    sim_params.debye_length = lambda_tf
    sim_params.eps_r = epsilon_r

    return physical_params, sim_params

# Load gate layout from file
def read_layout(gate_name):
    """Reads a .sqd layout file for the specified gate."""
    folder_path = os.path.join(os.getcwd(), "../../resources/bestagon_gates")
    file_path = os.path.join(folder_path, gate_name)
    return read_sqd_layout_100(file_path)

# Run QuickExact simulation
def run_quickexact_simulation(layout, physical_params):
    """Runs QuickExact simulation on a given layout and parameters."""
    quickexact_params_inst = quickexact_params()
    quickexact_params_inst.simulation_parameters = physical_params
    quickexact_params_inst.base_number_detection = automatic_base_number_detection.OFF
    return quickexact(layout, quickexact_params_inst)

# Run exhaustive GS simulation
def run_exgs_simulation(sim_params, layout_coordinates_angstrom):
    """Runs Exhaustive GS (ExGS) simulation."""
    sim_params.set_db_locs(layout_coordinates_angstrom)

    sa = egs.EGS(sim_params)
    start_time = time.time()
    sa.invoke()
    runtime = time.time() - start_time

    results = sidb_simulation_result_100()
    results.algorithm_name = "exgs"
    results.simulation_runtime = runtime

    return results

# Simulate all input patterns with ExGS
def simulate_exgs_inputs(sim_params, layout):
    """Enumerates input patterns and simulates using ExGS."""
    total_runtime = 0.0
    cds = charge_distribution_surface_100(layout)
    layout_coordinates_angstrom = [
        [pos[0] * 10, pos[1] * 10] for pos in cds.get_all_sidb_locations_in_nm()
    ]

    bii = bdl_input_iterator_100(layout)
    num_patterns = bii.num_input_pairs() ** 2

    for i in range(num_patterns + 1):
        layout = bii.get_layout()
        cds = charge_distribution_surface_100(layout)
        layout_coordinates_angstrom = [
            [pos[0] * 10, pos[1] * 10] for pos in cds.get_all_sidb_locations_in_nm()
        ]

        egs_results = run_exgs_simulation(sim_params, layout_coordinates_angstrom)
        total_runtime += egs_results.simulation_runtime.total_seconds()

        if num_patterns == 1 and i == num_patterns:
            break
        elif i == num_patterns - 1:
            break

        bii.__next__()

    return total_runtime

# Simulate all input patterns with QuickExact
def simulate_quickexact_inputs(layout, physical_params):
    """Enumerates input patterns and simulates using QuickExact."""
    total_runtime = 0.0

    bii = bdl_input_iterator_100(layout)
    num_patterns = bii.num_input_pairs() ** 2

    for i in range(num_patterns + 1):
        quickexact_runtime = run_quickexact_simulation(bii.get_layout(), physical_params)
        total_runtime += quickexact_runtime.simulation_runtime.total_seconds()

        if num_patterns == 1 and i == num_patterns:
            break
        elif i == num_patterns - 1:
            break

        bii.__next__()

    return total_runtime

# Main function
def main():
    """Main driver for running simulations on predefined gates."""
    physical_params, sim_params = initialize_simulation()

    gates = [
        ("wire", create_id_tt()),
        ("wire_diag", create_id_tt()),
        ("inv_diag", create_not_tt()),
        ("inv", create_not_tt()),
        ("fo2", create_fan_out_tt()),
        ("nand", create_nand_tt()),
        ("nor", create_nor_tt()),
        ("and", create_and_tt()),
        ("or", create_or_tt()),
        ("xor", create_xor_tt()),
        ("xnor", create_xnor_tt()),
        ("ha", create_half_adder_tt()),
        ("cx", create_crossing_wire_tt()),
        ("hourglass", create_double_wire_tt()),
    ]

    total_exgs_time = 0
    total_quickexact_time = 0

    statistics = []

    for gate_name, _ in gates:
        print(f"Processing gate: {gate_name}")
        layout = read_layout(f"{gate_name}.sqd")

        exgs_runtime = simulate_exgs_inputs(sim_params, layout)
        quickexact_runtime = simulate_quickexact_inputs(layout, physical_params)

        total_exgs_time += exgs_runtime
        total_quickexact_time += quickexact_runtime
        num_cells = (bdl_input_iterator_100(layout)).get_layout().num_cells()
        print(f"Number of SiDBs: {num_cells}")
        print(f"ExGS Runtime for '{gate_name}': {exgs_runtime:.4f} seconds")
        print(f"QuickExact Runtime for '{gate_name}': {quickexact_runtime:.4f} seconds")
        print(f"Ratio ExGS/QuickExact for '{gate_name}': {exgs_runtime / quickexact_runtime:.4f}")
        print("------------------------------")

        statistics.append({
            "gate": gate_name,
            "num_sidbs": num_cells,
            "exgs_runtime": exgs_runtime,
            "quickexact_runtime": quickexact_runtime,
            "ratio": exgs_runtime / quickexact_runtime,
        })

    # Save results to CSV
    current_date = datetime.now().strftime("%Y-%m-%d")
    file_name = f"simulation_results_{current_date}.csv"

    with open(file_name, "w", newline="") as csvfile:
        fieldnames = ["Gate", "Number of SiDBs", "ExGS Runtime (s)", "QuickExact Runtime (s)", "Ratio (ExGS/QuickExact)"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for stat in statistics:
            writer.writerow({
                "Gate": stat["gate"],
                "Number of SiDBs": stat["num_sidbs"],
                "ExGS Runtime (s)": stat["exgs_runtime"],
                "QuickExact Runtime (s)": stat["quickexact_runtime"],
                "Ratio (ExGS/QuickExact)": stat["ratio"],
            })

    # Print detailed statistics
    print(f"\n=== Detailed Runtime Statistics ===")
    for stat in statistics:
        print(f"Gate: {stat['gate']}")
        print(f"  ExGS Runtime: {stat['exgs_runtime']:.4f} seconds")
        print(f"  QuickExact Runtime: {stat['quickexact_runtime']:.4f} seconds")
        print(f"  Ratio ExGS/QuickExact: {stat['ratio']:.4f}\n")

    # Overall totals
    print(f"\nOverall Total ExGS Runtime: {total_exgs_time:.4f} seconds")
    print(f"Overall Total QuickExact Runtime: {total_quickexact_time:.4f} seconds")

    # --- NEW: Overall average ratio ---
    if statistics:
        average_ratio = sum(stat["ratio"] for stat in statistics) / len(statistics)
        print(f"Overall Average Ratio (ExGS/QuickExact): {average_ratio:.4f}")

if __name__ == "__main__":
    main()
