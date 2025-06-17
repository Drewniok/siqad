import os
import glob
import time
from mnt.pyfiction import *  # Ensure this import works in your environment
import pyegs.exhaustive_gs as egs

# Initialize simulation parameters
def initialize_simulation(mu=-0.32, lambda_tf=5.0, epsilon_r=5.6, num_instances=1):
    physical_parameters = sidb_simulation_parameters()
    physical_parameters.base = 2
    physical_parameters.mu_minus = mu
    physical_parameters.lambda_tf = lambda_tf
    physical_parameters.epsilon_r = epsilon_r

    sp = egs.SimParams()
    sp.base = 2
    sp.num_instances = num_instances
    sp.mu = mu

    return physical_parameters, sp

def run_quickexact_simulation(layout, physical_params):
    quickexact_params_inst = quickexact_params()
    quickexact_params_inst.simulation_parameters = physical_params
    quickexact_params_inst.base_number_detection = automatic_base_number_detection.OFF
    result_quickexact = quickexact(layout, quickexact_params_inst)
    return result_quickexact

def run_exgs_simulation(sp, layout_coordinates_angstrom):
    sp.set_db_locs(layout_coordinates_angstrom)
    egs_eng = egs.EGS(sp)

    start_time = time.time()
    egs_eng.invoke()
    simulation_runtime = time.time() - start_time

    pyfiction_exgs_results = sidb_simulation_result_100()
    pyfiction_exgs_results.algorithm_name = "simanneal"
    pyfiction_exgs_results.simulation_runtime = simulation_runtime
    return pyfiction_exgs_results

def load_layout_from_sqd(filepath):
    # Adjust if your actual API differs — this is an example
    layout = read_sqd_layout_100(filepath)
    return layout

def convert_nm_to_angstrom(coords_nm):
    return [[pos[0] * 10, pos[1] * 10] for pos in coords_nm]

def summarize_runtimes(base_folder="../../resources/random_layouts_quickexact_paper/"):
    physical_parameters, sp = initialize_simulation()

    folder_pattern = os.path.join(base_folder, "number_sidbs_*")

    folders = sorted(glob.glob(folder_pattern))

    summary_results = {}

    for folder in folders:
        quickexact_total_runtime = 0.0
        exgs_total_runtime = 0.0
        file_count = 0

        sqd_files = glob.glob(os.path.join(folder, "*.sqd"))
        print(f"Processing folder {folder} with {len(sqd_files)} layouts")

        for sqd_file in sqd_files:
            layout = load_layout_from_sqd(sqd_file)

            # Run QuickExact
            quickexact_result = run_quickexact_simulation(layout, physical_parameters)
            quickexact_runtime = quickexact_result.simulation_runtime.total_seconds()
            quickexact_total_runtime += quickexact_runtime

            # Prepare coordinates for ExGS
            cds = charge_distribution_surface_100(layout)
            all_positions_nm = cds.get_all_sidb_locations_in_nm()
            layout_coordinates_angstrom = convert_nm_to_angstrom(all_positions_nm)

            if layout.num_cells() < 28:
                # Run ExGS
                exgs_result = run_exgs_simulation(sp, layout_coordinates_angstrom)
                exgs_runtime = exgs_result.simulation_runtime.total_seconds()  # in seconds
                exgs_total_runtime += exgs_runtime

            file_count += 1

        print(f"runtime exgs {exgs_total_runtime}")
        print(f"runtime quickexact {quickexact_total_runtime}")
        print(f"exgs/quickexact ratio {exgs_total_runtime / quickexact_total_runtime if quickexact_total_runtime > 0 else float('inf')}")
        print("------------------------------------------------------")

        summary_results[folder] = {
            "num_layouts": file_count,
            "quickexact_total_runtime_sec": quickexact_total_runtime,
            "exgs_total_runtime_sec": exgs_total_runtime,
            "quickexact_avg_runtime_sec": quickexact_total_runtime / file_count if file_count else 0,
            "exgs_avg_runtime_sec": exgs_total_runtime / file_count if file_count else 0,
        }

    return summary_results

def main():
    results = summarize_runtimes()
    print("\n=== Summary of runtimes per folder ===")
    for folder, data in results.items():
        print(f"Folder: {folder}")
        print(f"  Number of layouts: {data['num_layouts']}")
        print(f"  QuickExact total runtime: {data['quickexact_total_runtime_sec']:.2f} s")
        print(f"  QuickExact avg runtime: {data['quickexact_avg_runtime_sec']:.2f} s")
        print(f"  ExGS total runtime: {data['exgs_total_runtime_sec']:.2f} s")
        print(f"  ExGS avg runtime: {data['exgs_avg_runtime_sec']:.2f} s")
        print("------------------------------------------------------")

if __name__ == "__main__":
    main()
