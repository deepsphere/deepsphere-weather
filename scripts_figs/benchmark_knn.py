import os
import sys

sys.path.append("../")
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from modules.my_plotting import benchmark_global_skill
from modules.my_plotting import benchmark_global_skills

# Plotting options
import matplotlib

matplotlib.use("cairo")  # Cairo
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white"  # (1,1,1,0)
matplotlib.rcParams["savefig.edgecolor"] = "none"

##-----------------------------------------------------------------------------.
# Define directories
base_dir = "/data/weather_prediction"
figs_dir = os.path.join(base_dir, "figs")

##-----------------------------------------------------------------------------.
# Load different modelling approaches

k8_skills = xr.open_dataset(
    os.path.join(
        base_dir,
        "experiments_knn",
        "RNN-UNetSpherical-healpix-16-k8-MaxAreaPooling-float32-AR2-LinearStep",
        "model_skills/deterministic_global_skill.nc",
    )
)
k20_skills = xr.open_dataset(
    os.path.join(
        base_dir,
        "experiments_knn",
        "RNN-UNetSpherical-healpix-16-k20-MaxAreaPooling-float32-AR2-LinearStep",
        "model_skills/deterministic_global_skill.nc",
    )
)
k40_skills = xr.open_dataset(
    os.path.join(
        base_dir,
        "experiments_knn",
        "RNN-UNetSpherical-healpix-16-k40-MaxAreaPooling-float32-AR2-LinearStep",
        "model_skills/deterministic_global_skill.nc",
    )
)
k60_skills = xr.open_dataset(
    os.path.join(
        base_dir,
        "experiments_knn",
        "RNN-UNetSpherical-healpix-16-k60-MaxAreaPooling-float32-AR2-LinearStep",
        "model_skills/deterministic_global_skill.nc",
    )
)
k20_old_skills = xr.open_dataset(
    os.path.join(
        base_dir,
        "experiments_samplings",
        "RNN-UNetSpherical-healpix-16-k20-MaxAreaPooling-float32-AR2-LinearStep",
        "model_skills/deterministic_global_skill.nc",
    )
)


##-----------------------------------------------------------------------------.
# Load Weyn benchmark
weyn_skills = xr.open_dataset("/data/weather_prediction/Benchmarks/rmses_weyn.nc")
weyn_skills["lead_time"] = weyn_skills.lead_time.values * np.timedelta64(1, "h")
weyn_skills = weyn_skills.rename({"lead_time": "leadtime"})
weyn_skills = weyn_skills.expand_dims({"skill": ["RMSE"]})

## Load climatology and persistence baselines
sampling_name = "Equiangular_400km"  # All similar btw
benchmark_dir = os.path.join(base_dir, "data", sampling_name, "Benchmarks")
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
    "k = 8": k8_skills,
    "k = 20": k20_skills,
    "k = 40": k40_skills,
    "k = 60": k60_skills,
    "Weyn et al., 2020": weyn_skills,
    "Persistence forecast": persistence_skills,
    "Weekly Climatology": WeeklyClimatology_skills,
    "HourlyWeekly Climatology": HourlyWeeklyClimatology_skills,
    #'Monthly Climatology': MonthlyClimatology_skills,
    #'Daily Climatology': DailyClimatology_skills,
    #'HourlyMonthly Climatology': HourlyMonthlyClimatology_skills,
}

## Define the dictionary of forecast to benchmark (without Weyn)
skills_dict1 = {
    "k = 8": k8_skills,
    "k = 20": k20_skills,
    "k = 40": k40_skills,
    "k = 60": k60_skills,
    # 'Weyn et al., 2020': weyn_skills,
    "Persistence forecast": persistence_skills,
    "Weekly Climatology": WeeklyClimatology_skills,
    "HourlyWeekly Climatology": HourlyWeeklyClimatology_skills,
    #'Monthly Climatology': MonthlyClimatology_skills,
    #'Daily Climatology': DailyClimatology_skills,
    #'HourlyMonthly Climatology': HourlyMonthlyClimatology_skills,
}

colors_dict = {
    "k = 8": "dodgerblue",
    "k = 20": "forestgreen",
    "k = 40": "orange",
    "k = 60": "red",
    "Weyn et al., 2020": "darkviolet",
    "Persistence forecast": "gray",
    "Weekly Climatology": "gray",
    "HourlyWeekly Climatology": "gray",
    "Monthly Climatology": "gray",
    "Daily Climatology": "gray",
    "HourlyMonthly Climatology": "gray",
}

# Benchmark skills
benchmark_global_skill(
    skills_dict=skills_dict,
    skill="RMSE",
    variables=["z500", "t850"],
    colors_dict=colors_dict,
    n_leadtimes=20,
).savefig(os.path.join(figs_dir, "Benchmark_RMSE_knn.png"))


benchmark_global_skills(
    skills_dict=skills_dict1,
    skills=["BIAS", "RMSE", "rSD", "pearson_R2"],
    variables=["z500", "t850"],
    colors_dict=colors_dict,
    legend_everywhere=True,
    n_leadtimes=20,
).savefig(os.path.join(figs_dir, "Benchmark_Overview_knn.png"))
