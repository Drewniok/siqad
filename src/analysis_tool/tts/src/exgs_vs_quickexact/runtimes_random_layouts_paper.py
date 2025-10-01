import numpy as np
import time
from mnt.pyfiction import *  # Ensure this import is correct
from datetime import datetime
import os
import glob
import csv

import pyegs.exhaustive_gs as egs

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

def run_exgs_simulation(sp, layout_coordinates_angstrom):
    """
    Runs ExGS simulations and collects results.

    Parameters:
    - sp: ExGS parameters
    - layout_coordinates_angstrom: Coordinates of the layout in Angstroms

    Returns:
    - pyfiction_exgs_results: Simulation results
    """
    sp.set_db_locs(layout_coordinates_angstrom)

    egs_eng = egs.EGS(sp)

    start_time = time.time()
    egs_eng.invoke()
    simulation_runtime = time.time() - start_time

    pyfiction_exgs_results = sidb_simulation_result_100()
    pyfiction_exgs_results.algorithm_name = "ExGS"
    pyfiction_exgs_results.simulation_runtime = simulation_runtime

    return pyfiction_exgs_results

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

def summarize_runtimes(base_folder="../../resources/random_layouts_quickexact_paper/",
                       output_csv="simulation_runtimes_random_layouts_quickexact_paper.csv"):
    physical_parameters, sp = initialize_simulation()

    folder_pattern = os.path.join(base_folder, "number_sidbs_*")
    folders = sorted(glob.glob(folder_pattern))

    summary_results = {}
    total_quickexact_runtime = 0.0
    total_exgs_runtime = 0.0
    ratio_list = []

    # Prepare CSV file
    with open(output_csv, mode="w", newline="") as csvfile:
        fieldnames = ["layout_id", "num_sidbs", "runtime_exgs_sec", "runtime_quickexact_sec"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        num_sidb_folder = 0

        for folder in folders:
            sqd_files = glob.glob(os.path.join(folder, "*.sqd"))
            print(f"Processing folder {folder} with {len(sqd_files)} layouts")

            quickexact_total_runtime = 0.0
            exgs_total_runtime = 0.0
            file_count = 0

            for sqd_file in sqd_files:
                layout = read_sqd_layout_100(sqd_file)
                num_sidbs = layout.num_cells()
                num_sidb_folder = num_sidbs

                # Run QuickExact
                quickexact_result = run_quickexact_simulation(layout, physical_parameters)
                quickexact_runtime = quickexact_result.simulation_runtime.total_seconds()
                quickexact_total_runtime += quickexact_runtime

                # Prepare coordinates for ExGS
                cds = charge_distribution_surface_100(layout)
                all_positions_nm = cds.get_all_sidb_locations_in_nm()
                layout_coordinates_angstrom = [[pos[0] * 10, pos[1] * 10] for pos in all_positions_nm]

                runtime_exgs = 0.0
                if num_sidbs < 28:
                    # Run ExGS
                    exgs_result = run_exgs_simulation(sp, layout_coordinates_angstrom)
                    runtime_exgs = exgs_result.simulation_runtime.total_seconds()
                    exgs_total_runtime += runtime_exgs

                # Write to CSV
                writer.writerow({
                    "layout_id": os.path.basename(sqd_file),
                    "num_sidbs": num_sidbs,
                    "runtime_exgs_sec": runtime_exgs,
                    "runtime_quickexact_sec": quickexact_runtime
                })

                file_count += 1

            # Verhältnis für diesen Ordner
            ratio = (exgs_total_runtime / quickexact_total_runtime
                     if quickexact_total_runtime > 0 else float("inf"))

            print(f"ExGS runtime for #SiDBs {num_sidb_folder}: {exgs_total_runtime:.2f} seconds")
            print(f"QuickExact runtime for #SiDBs {num_sidb_folder}: {quickexact_total_runtime:.2f} seconds")
            print(f"(ExGS/QuickExact): {ratio:.4f}")



            summary_results[folder] = {
                "num_layouts": file_count,
                "quickexact_total_runtime_sec": quickexact_total_runtime,
                "exgs_total_runtime_sec": exgs_total_runtime,
                "quickexact_avg_runtime_sec": quickexact_total_runtime / file_count if file_count else 0,
                "exgs_avg_runtime_sec": exgs_total_runtime / file_count if file_count else 0,
                "ratio": ratio,
            }

            total_quickexact_runtime += quickexact_total_runtime
            total_exgs_runtime += exgs_total_runtime
            ratio_list.append(ratio)

    # Globale Zusammenfassung
    global_summary = {
        "total_quickexact_runtime_sec": total_quickexact_runtime,
        "total_exgs_runtime_sec": total_exgs_runtime,
        "average_ratio": sum(ratio_list) / len(ratio_list) if ratio_list else float("nan"),
    }

    return summary_results, global_summary

def main():
    results, global_summary = summarize_runtimes()
    print("\n=== Summary of runtimes per folder ===")
    for folder, data in results.items():
        print(f"Folder: {folder}")
        print(f"  Number of layouts: {data['num_layouts']}")
        print(f"  QuickExact total runtime: {data['quickexact_total_runtime_sec']:.2f} s")
        print(f"  QuickExact avg runtime: {data['quickexact_avg_runtime_sec']:.2f} s")
        print(f"  ExGS total runtime: {data['exgs_total_runtime_sec']:.2f} s")
        print(f"  ExGS avg runtime: {data['exgs_avg_runtime_sec']:.2f} s")
        print(f"  Ratio (ExGS/QuickExact): {data['ratio']:.2f}")
        print("------------------------------------------------------")

    print("\n=== Global Summary ===")
    print(f"Total QuickExact runtime: {global_summary['total_quickexact_runtime_sec']:.2f} s")
    print(f"Total ExGS runtime: {global_summary['total_exgs_runtime_sec']:.2f} s")
    print(f"Average ratio (ExGS/QuickExact): {global_summary['average_ratio']:.2f}")


if __name__ == "__main__":
    main()
