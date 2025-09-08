def summarize_runtimes(base_folder="../../resources/random_layouts_quickexact_paper/"):
    physical_parameters, sp = initialize_simulation()

    folder_pattern = os.path.join(base_folder, "number_sidbs_*")
    folders = sorted(glob.glob(folder_pattern))

    summary_results = {}

    total_quickexact_runtime = 0.0
    total_exgs_runtime = 0.0
    ratio_list = []

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
                exgs_runtime = exgs_result.simulation_runtime.total_seconds()
                exgs_total_runtime += exgs_runtime

            file_count += 1

        # Verhältnis für diesen Ordner
        ratio = (exgs_total_runtime / quickexact_total_runtime
                 if quickexact_total_runtime > 0 else float("inf"))

        print(f"runtime exgs {exgs_total_runtime}")
        print(f"runtime quickexact {quickexact_total_runtime}")
        print(f"exgs/quickexact ratio {ratio}")
        print("------------------------------------------------------")

        summary_results[folder] = {
            "num_layouts": file_count,
            "quickexact_total_runtime_sec": quickexact_total_runtime,
            "exgs_total_runtime_sec": exgs_total_runtime,
            "quickexact_avg_runtime_sec": quickexact_total_runtime / file_count if file_count else 0,
            "exgs_avg_runtime_sec": exgs_total_runtime / file_count if file_count else 0,
            "ratio": ratio,
        }

        # Globale Summen sammeln
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
