import os
import sys

sys.path.append("../")
import xarray as xr
from modules.my_plotting import benchmark_global_skill
from modules.my_plotting import benchmark_global_skills

data_dir = "/data/weather_prediction/data"

# Define samplings
sampling_names = [
    "Healpix_400km",
    "Equiangular_400km",
    "Equiangular_400km_tropics",
    "Icosahedral_400km",
    "O24",
    "Cubed_400km",
]
##-----------------------------------------------------------------------------.
## Load climatology and persistence baselines for each sampling
persistence_dict = {}
WeeklyClimatology_dict = {}
HourlyWeeklyClimatology_dict = {}

for sampling_name in sampling_names:
    benchmark_dir = os.path.join(
        "/data/weather_prediction/data", sampling_name, "Benchmarks"
    )
    persistence_skills = xr.open_dataset(
        os.path.join(benchmark_dir, "Persistence_Global_Skills.nc")
    )
    WeeklyClimatology_skills = xr.open_dataset(
        os.path.join(benchmark_dir, "WeeklyClimatology_Global_Skills.nc")
    )
    HourlyWeeklyClimatology_skills = xr.open_dataset(
        os.path.join(benchmark_dir, "HourlyWeeklyClimatology_Global_Skills.nc")
    )
    # Add to dictionaries
    persistence_dict[sampling_name] = persistence_skills
    WeeklyClimatology_dict[sampling_name] = WeeklyClimatology_skills
    HourlyWeeklyClimatology_dict[sampling_name] = HourlyWeeklyClimatology_skills

##-----------------------------------------------------------------------------.


# Benchmark skills
benchmark_global_skill(
    skills_dict=persistence_dict,
    skill="RMSE",
    variables=["z500", "t850"],
    n_leadtimes=20,
)

benchmark_global_skills(
    skills_dict=persistence_dict,
    skills=["BIAS", "RMSE", "rSD", "pearson_R2"],
    variables=["z500", "t850"],
    legend_everywhere=True,
    n_leadtimes=20,
)
