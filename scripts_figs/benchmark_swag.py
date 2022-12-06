import os
import sys

sys.path.append("../")

import argparse
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from modules.my_plotting import benchmark_global_skill
from modules.my_plotting import benchmark_global_skills


# Plotting options
import matplotlib

# matplotlib.use('cairo') # Cairo
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white"  # (1,1,1,0)
matplotlib.rcParams["savefig.edgecolor"] = "none"


def main(base_dir, figs_dir, benchmark_dir):
    ##-----------------------------------------------------------------------------.
    # Load different modelling approaches
    RNN_classic = xr.open_dataset(
        os.path.join(
            base_dir,
            "RNN-UNetSpherical-icosahedral-16-k20-MaxAreaPooling-float32-AR6-LinearStep",
            "model_skills/deterministic_global_skill.nc",
        )
    )
    RNN_classic_8 = xr.open_dataset(
        os.path.join(
            base_dir,
            "RNN-UNetSpherical-icosahedral-16-k20-MaxAreaPooling-float32-AR6-LinearStep-8",
            "model_skills/deterministic_global_skill.nc",
        )
    )

    RNN_swag_00 = xr.open_dataset(
        os.path.join(
            base_dir,
            "RNN-UNetSpherical-icosahedral-16-k20-MaxAreaPooling-float32-AR6-LinearStep-SWAG",
            "model_skills/deterministic_global_skill_00.nc",
        )
    )
    RNN_swag_00_lr_001_swalr_0001 = xr.open_dataset(
        os.path.join(
            base_dir,
            "RNN-UNetSpherical-icosahedral-16-k20-MaxAreaPooling-float32-AR6-LinearStep-SWAG-LR001-SWALR0001",
            "model_skills/deterministic_global_skill_00.nc",
        )
    )

    RNN_8_swag_00_lr_0007_swalr_0001 = xr.open_dataset(
        os.path.join(
            base_dir,
            "RNN-UNetSpherical-icosahedral-16-k20-MaxAreaPooling-float32-AR6-LinearStep-SWAG-LR001-SWALR0001",
            "model_skills/deterministic_global_skill_00.nc",
        )
    )

    ##-----------------------------------------------------------------------------.
    # Load Weyn benchmark
    weyn_skills = xr.open_dataset(
        "/mnt/scratch/students/haddad/weather_prediction/data/healpix/metrics/rmses_weyn.nc"
    )
    weyn_skills["lead_time"] = weyn_skills.lead_time.values * np.timedelta64(1, "h")
    weyn_skills = weyn_skills.rename({"lead_time": "leadtime"})
    weyn_skills = weyn_skills.expand_dims({"skill": ["RMSE"]})

    ## Load climatology and persistence baselines

    persistence_skills = xr.open_dataset(
        os.path.join(benchmark_dir, "Persistence_Global_Skills.nc")
    )
    WeeklyClimatology_skills = xr.open_dataset(
        os.path.join(benchmark_dir, "WeeklyClimatology_Global_Skills.nc")
    )
    DailyClimatology_skills = xr.open_dataset(
        os.path.join(benchmark_dir, "DailyClimatology_Global_Skills.nc")
    )
    MonthlyClimatology_skills = xr.open_dataset(
        os.path.join(benchmark_dir, "MonthlyClimatology_Global_Skills.nc")
    )
    HourlyWeeklyClimatology_skills = xr.open_dataset(
        os.path.join(benchmark_dir, "HourlyWeeklyClimatology_Global_Skills.nc")
    )
    HourlyMonthlyClimatology_skills = xr.open_dataset(
        os.path.join(benchmark_dir, "HourlyMonthlyClimatology_Global_Skills.nc")
    )

    ## Define the dictionary of forecast to benchmark
    skills_dict = {
        #    'Planar Projection': planar_skills,
        #    'Cylindrical Projection': cylinder_kills,
        "DeepSphere - Icosahedral": RNN_classic,
        "DeepSphere - Icosahedral 8 epochs": RNN_classic_8,
        "RNN Icosahedral SWA": RNN_swag_00,
        "RNN Icosahedral SWA | LR 0.01 SWA-LR 0.001": RNN_swag_00_lr_001_swalr_0001,
        "RNN Icosahedral (8 epochs) SWA | LR 0.007 SWA-LR 0.0001": RNN_8_swag_00_lr_0007_swalr_0001,
        # 'RNN Healpix State SWAG 0.1': RNN_swag_01,
        "Weyn et al., 2020": weyn_skills,
        "Persistence forecast": persistence_skills,
        "Weekly Climatology": WeeklyClimatology_skills,
        "HourlyWeekly Climatology": HourlyWeeklyClimatology_skills,
        #'Monthly Climatology': MonthlyClimatology_skills,
        #'Daily Climatology': DailyClimatology_skills,
        #'HourlyMonthly Climatology': HourlyMonthlyClimatology_skills,
    }

    # ## Define the dictionary of forecast to benchmark
    # skills_dict1 = {
    #                 #    'Planar Projection': planar_skills,
    #                 #    'Cylindrical Projection': cylinder_kills,
    #             'DeepSphere - Equiangular': graph_skills,
    #             'AR Healpix State': AR_state_skills,
    #             'RNN Healpix State': RNN_state_skills,
    #             'RNN Healpix Anom': RNN_anom_skills,
    #             'RNN Healpix State inc': RNN_state_inc_skills,
    #                 # 'Weyn et al., 2020': weyn_skills,
    #             'Persistence forecast': persistence_skills,
    #             'Weekly Climatology': WeeklyClimatology_skills,
    #             'HourlyWeekly Climatology': HourlyWeeklyClimatology_skills,
    #             #'Monthly Climatology': MonthlyClimatology_skills,
    #             #'Daily Climatology': DailyClimatology_skills,
    #             #'HourlyMonthly Climatology': HourlyMonthlyClimatology_skills,
    #             }

    # Benchmark skills
    benchmark_global_skill(
        skills_dict=skills_dict,
        skill="RMSE",
        variables=["z500", "t850"],
        n_leadtimes=20,
    ).savefig(os.path.join(figs_dir, "Benchmark_Models_SWAG_RMSE.png"))

    # benchmark_global_skills(skills_dict=skills_dict1,
    #                         skills=['BIAS','RMSE','rSD','pearson_R2'],
    #                         variables=['z500','t850'],
    #                         legend_everywhere = True,
    #                         n_leadtimes=20).savefig(os.path.join(figs_dir, "Benchmark_Models_Overview.png"))

    # benchmark_global_skills(skills_dict=skills_dict,
    #                         skills=['relBIAS','MAE','relMAE','diffSD','NSE', 'error_CoV'],
    #                         variables=['z500','t850'],
    #                         legend_everywhere = True,
    #                         n_leadtimes=20)


if __name__ == "__main__":
    ##-----------------------------------------------------------------------------.
    # Define directories
    base_dir = "/mnt/scratch/students/wefeng/"
    exp_dir = "/mnt/scratch/students/haddad/experiments"
    figs_dir = os.path.join(exp_dir, "figs")
    sampling_name = "Icosahedral_400km"
    benchmark_dir = os.path.join(base_dir, "data", sampling_name, "Benchmarks")

    parser = argparse.ArgumentParser(description="Benchmarking SWAG models")
    parser.add_argument("--base_dir", type=str, default=exp_dir)
    parser.add_argument("--figs_dir", type=str, default=figs_dir)
    parser.add_argument("--benchmark_dir", type=str, default=benchmark_dir)

    args = parser.parse_args()

    main(args.base_dir, args.figs_dir, args.benchmark_dir)
