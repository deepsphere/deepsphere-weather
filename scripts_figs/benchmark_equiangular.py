import os
import sys

sys.path.append("../")
import matplotlib.pyplot as plt
import xarray as xr
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
        "RNN-UNetSpherical-equiangular-[36, 72]-k20-maxPooling-float32-AR6-planar",
        "model_skills/deterministic_global_skill.nc",
    )
)
cylinder_kills = xr.open_dataset(
    os.path.join(
        base_dir,
        "experiments_equiangular",
        "RNN-UNetSpherical-equiangular-[36, 72]-k20-maxPooling-float32-AR6-cylinder",
        "model_skills/deterministic_global_skill.nc",
    )
)
# graph_skills = xr.open_dataset(os.path.join(base_dir, "experiments_equiangular", "graph", "model_skills/deterministic_global_skill.nc"))
graph_skills = xr.open_dataset(
    os.path.join(
        base_dir,
        "experiments_equiangular",
        "RNN-UNetSpherical-equiangular-[36, 72]-k20-MaxPooling-float32-AR6-LinearStep",
        "model_skills/deterministic_global_skill.nc",
    )
)
graph_skills1 = xr.open_dataset(
    os.path.join(
        base_dir,
        "experiments_equiangular",
        "RNN-UNetSpherical-equiangular_test-[36, 72]-k20-InterpPooling-float32-AR6-LinearStep",
        "model_skills/deterministic_global_skill.nc",
    )
)


## Load climatology and persistence baselines
sampling_name = "Equiangular_400km"
benchmark_dir = os.path.join(
    "/data/weather_prediction/data", sampling_name, "Benchmarks"
)
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
    "Planar Projection": planar_skills,
    "Cylindrical Projection": cylinder_kills,
    "DeepSphere - Equiangular": graph_skills,
    "Persistence forecast": persistence_skills,
    "Weekly Climatology": WeeklyClimatology_skills,
    "HourlyWeekly Climatology": HourlyWeeklyClimatology_skills,
    #'Monthly Climatology': MonthlyClimatology_skills,
    #'Daily Climatology': DailyClimatology_skills,
    #'HourlyMonthly Climatology': HourlyMonthlyClimatology_skills,
}

colors_dict = {
    "Planar Projection": "blue",
    "Cylindrical Projection": "aqua",
    "DeepSphere - Equiangular": "dodgerblue",
    "Equiangular": "dodgerblue",
    "Reduced Gaussian": "forestgreen",
    "Cubed": "orange",
    "Icosahedral": "red",
    "Healpix": "fuchsia",
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
).savefig(os.path.join(figs_dir, "Benchmark_RMSE_Equiangular.png"))

benchmark_global_skills(
    skills_dict=skills_dict,
    skills=["BIAS", "RMSE", "rSD", "pearson_R2"],
    variables=["z500", "t850"],
    colors_dict=colors_dict,
    legend_everywhere=True,
    n_leadtimes=20,
).savefig(os.path.join(figs_dir, "Benchmark_Overview_Equiangular.png"))

benchmark_global_skill(
    skills_dict=skills_dict,
    skill="RMSE",
    variables=["z500", "t850"],
    colors_dict=colors_dict,
    ylim=(75, 200),
    n_leadtimes=4,
).savefig(os.path.join(figs_dir, "Benchmark_Zoom_RMSE_Equiangular.png"))

# benchmark_global_skills(skills_dict=skills_dict,
#                         skills=['relBIAS','MAE','relMAE','diffSD','NSE', 'error_CoV'],
#                         variables=['z500','t850'],
#                         legend_everywhere = True,
#                         n_leadtimes=20)

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors_dict = {k: default_colors[i] for i, (k, v) in enumerate(skills_dict.items())}
