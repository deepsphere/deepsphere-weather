
import os
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import xarray as xr
from modules.my_plotting import benchmark_global_skill
from modules.my_plotting import benchmark_global_skills

data_dir = "/data/weather_prediction/data"
exp_dir = "/data/weather_prediction/experiments_equiangular"

# Load planar, cylinder and DeepSphere (Equiangular) skills 
planar_skills_fpath = os.path.join(exp_dir, "planar", "model_skills/deterministic_global_skill.nc")
cylinder_skills_fpath = os.path.join(exp_dir, "cylinder", "model_skills/deterministic_global_skill.nc")
graph_skills_fpath = os.path.join(exp_dir, "graph", "model_skills/deterministic_global_skill.nc")

planar_skills = xr.open_dataset(planar_skills_fpath)
cylinder_kills = xr.open_dataset(cylinder_skills_fpath)
graph_skills = xr.open_dataset(graph_skills_fpath)

## Load climatology and persistence baselines 
sampling_name = "Equiangular_400km"
benchmark_dir = os.path.join("/data/weather_prediction/data", sampling_name, "Benchmarks")
persistence_skills = xr.open_dataset(os.path.join(benchmark_dir, "Persistence_Global_Skills.nc"))
WeeklyClimatology_skills = xr.open_dataset(os.path.join(benchmark_dir, "WeeklyClimatology_Global_Skills.nc"))
DailyClimatology_skills = xr.open_dataset(os.path.join(benchmark_dir, "DailyClimatology_Global_Skills.nc"))
MonthlyClimatology_skills = xr.open_dataset(os.path.join(benchmark_dir, "MonthlyClimatology_Global_Skills.nc"))
HourlyWeeklyClimatology_skills = xr.open_dataset(os.path.join(benchmark_dir, "HourlyWeeklyClimatology_Global_Skills.nc"))
HourlyMonthlyClimatology_skills = xr.open_dataset(os.path.join(benchmark_dir, "HourlyMonthlyClimatology_Global_Skills.nc"))

## Define the dictionary of forecast to benchmark 
skills_dict = {'Planar Projection': planar_skills,
               'Cylindrical Projection': cylinder_kills,
               'DeepSphere - Equiangular': graph_skills,
               'Persistence forecast': persistence_skills,
               'Weekly Climatology': WeeklyClimatology_skills,
               'HourlyWeekly Climatology': HourlyWeeklyClimatology_skills,
               #'Monthly Climatology': MonthlyClimatology_skills,
               #'Daily Climatology': DailyClimatology_skills,
               #'HourlyMonthly Climatology': HourlyMonthlyClimatology_skills,
               }

# Benchmark skills 
benchmark_global_skill(skills_dict=skills_dict, 
                       skill="RMSE", 
                       variables=['z500','t850'],
                       n_leadtimes=20)

benchmark_global_skills(skills_dict=skills_dict, 
                        skills=['BIAS','RMSE','rSD','pearson_R2'],
                        variables=['z500','t850'],
                        legend_everywhere = True,
                        n_leadtimes=20)

benchmark_global_skills(skills_dict=skills_dict, 
                        skills=['relBIAS','MAE','relMAE','diffSD','NSE', 'error_CoV'],
                        variables=['z500','t850'],
                        legend_everywhere = True,
                        n_leadtimes=20)