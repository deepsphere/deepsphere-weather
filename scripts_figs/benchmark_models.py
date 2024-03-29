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
# Load planar, cylinder and DeepSphere (Equiangular) skills
planar_skills = xr.open_dataset(
    os.path.join(
        base_dir,
        "experiments_equiangular",
        "planar",
        "model_skills/deterministic_global_skill.nc",
    )
)
cylinder_kills = xr.open_dataset(
    os.path.join(
        base_dir,
        "experiments_equiangular",
        "cylinder",
        "model_skills/deterministic_global_skill.nc",
    )
)
graph_skills = xr.open_dataset(
    os.path.join(
        base_dir,
        "experiments_equiangular",
        "graph",
        "model_skills/deterministic_global_skill.nc",
    )
)

##-----------------------------------------------------------------------------.
# Load different modelling approaches
RNN_anom_skills = xr.open_dataset(
    os.path.join(
        base_dir,
        "experiments_GG",
        "Anom-RNN-UNetSpherical-healpix-16-k20-MaxAreaPooling-float32-AR6-LinearStep",
        "model_skills/deterministic_global_skill.nc",
    )
)
RNN_state_inc_skills = xr.open_dataset(
    os.path.join(
        base_dir,
        "experiments_GG",
        "RNN-UNetDiffSpherical-healpix-16-k20-MaxAreaPooling-float32-AR6-LinearStep",
        "model_skills/deterministic_global_skill.nc",
    )
)
AR_state_skills = xr.open_dataset(
    os.path.join(
        base_dir,
        "experiments_GG",
        "AR-UNetSpherical-healpix-16-k20-MaxAreaPooling-float32-AR6-LinearStep",
        "model_skills/deterministic_global_skill.nc",
    )
)

# Tmp correction
RNN_state_skills = xr.open_dataset(
    os.path.join(
        base_dir,
        "experiments_GG",
        "RNN-UNetSpherical-healpix-16-k20-MaxAreaPooling-float32-AR6-LinearStep_weight_corrected",
        "model_skills/deterministic_global_skill.nc",
    )
)
RNN_state_skills["leadtime"] = RNN_state_skills["leadtime"].values + np.timedelta64(
    6, "h"
)

##-----------------------------------------------------------------------------.
# Load Weyn benchmark
weyn_skills = xr.open_dataset("/data/weather_prediction/Benchmarks/rmses_weyn.nc")
weyn_skills["lead_time"] = weyn_skills.lead_time.values * np.timedelta64(1, "h")
weyn_skills = weyn_skills.rename({"lead_time": "leadtime"})
weyn_skills = weyn_skills.expand_dims({"skill": ["RMSE"]})

## Load climatology and persistence baselines
sampling_name = "Equiangular_400km"
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
    "State": RNN_state_skills,
    "State increment": RNN_state_inc_skills,
    "Anomalies": RNN_anom_skills,
    "Weyn et al., 2020": weyn_skills,
    "Persistence forecast": persistence_skills,
    "Weekly Climatology": WeeklyClimatology_skills,
    "HourlyWeekly Climatology": HourlyWeeklyClimatology_skills,
    #'Monthly Climatology': MonthlyClimatology_skills,
    #'Daily Climatology': DailyClimatology_skills,
    #'HourlyMonthly Climatology': HourlyMonthlyClimatology_skills,
}

## Define the dictionary of forecast to benchmark
skills_dict1 = {
    "State": RNN_state_skills,
    "State increment": RNN_state_inc_skills,
    "Anomalies": RNN_anom_skills,
    # 'Weyn et al., 2020': weyn_skills,
    "Persistence forecast": persistence_skills,
    "Weekly Climatology": WeeklyClimatology_skills,
    "HourlyWeekly Climatology": HourlyWeeklyClimatology_skills,
    #'Monthly Climatology': MonthlyClimatology_skills,
    #'Daily Climatology': DailyClimatology_skills,
    #'HourlyMonthly Climatology': HourlyMonthlyClimatology_skills,
}

colors_dict = {
    "State": "dodgerblue",
    "State increment": "darkorange",
    "Anomalies": "forestgreen",
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
).savefig(os.path.join(figs_dir, "Benchmark_RMSE_StrategyModels.png"))


benchmark_global_skills(
    skills_dict=skills_dict1,
    skills=["BIAS", "RMSE", "rSD", "pearson_R2"],
    variables=["z500", "t850"],
    colors_dict=colors_dict,
    legend_everywhere=True,
    n_leadtimes=20,
).savefig(os.path.join(figs_dir, "Benchmark_Overview_StrategyModels.png"))

# benchmark_global_skills(skills_dict=skills_dict,
#                         skills=['relBIAS','MAE','relMAE','diffSD','NSE', 'error_CoV'],
#                         variables=['z500','t850'],
#                         legend_everywhere = True,
#                         n_leadtimes=20)


## LONG TERM RMSE ... weyn vs ours
benchmark_global_skill(
    skills_dict=skills_dict,
    skill="RMSE",
    variables=["z500", "t850"],
    colors_dict=colors_dict,
    ylim=None,
    n_leadtimes=39,
).savefig(os.path.join(figs_dir, "Benchmark_LT_RMSE_StrategyModels.png"))
