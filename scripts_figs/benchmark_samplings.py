
import os
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from modules.my_plotting import benchmark_global_skill
from modules.my_plotting import benchmark_global_skills

# Plotting options
import matplotlib
matplotlib.use('cairo') # Cairo
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white" # (1,1,1,0)
matplotlib.rcParams["savefig.edgecolor"] = 'none'

##-----------------------------------------------------------------------------.
# Define directories
base_dir = "/data/weather_prediction"
figs_dir = os.path.join(base_dir, "figs")

##-----------------------------------------------------------------------------.
# Load planar, cylinder and DeepSphere (Equiangular) skills 
planar_skills = xr.open_dataset(os.path.join(base_dir, "experiments_equiangular","RNN-UNetSpherical-equiangular-[36, 72]-k20-maxPooling-float32-AR6-planar", "model_skills/deterministic_global_skill.nc"))
cylinder_kills = xr.open_dataset(os.path.join(base_dir, "experiments_equiangular", "RNN-UNetSpherical-equiangular-[36, 72]-k20-maxPooling-float32-AR6-cylinder", "model_skills/deterministic_global_skill.nc"))
graph_skills = xr.open_dataset(os.path.join(base_dir, "experiments_equiangular" , "RNN-UNetSpherical-equiangular-[36, 72]-k20-MaxPooling-float32-AR6-LinearStep", "model_skills/deterministic_global_skill.nc"))

##-----------------------------------------------------------------------------.
# Load different modelling approaches
RNN_cubed_skills = xr.open_dataset(os.path.join(base_dir, "experiments_samplings" , "RNN-UNetSpherical-cubed-24-k20-MaxAreaPooling-float32-AR6-LinearStep", "model_skills/deterministic_global_skill.nc"))
RNN_equi_skills = xr.open_dataset(os.path.join(base_dir, "experiments_equiangular" , "RNN-UNetSpherical-equiangular-[36, 72]-k20-MaxPooling-float32-AR6-LinearStep", "model_skills/deterministic_global_skill.nc"))
RNN_gauss_skills = xr.open_dataset(os.path.join(base_dir, "experiments_samplings" , "RNN-UNetSpherical-gauss-48-k20-MaxAreaPooling-float32-AR6-LinearStep", "model_skills/deterministic_global_skill.nc"))
RNN_healpix_skills = xr.open_dataset(os.path.join(base_dir, "experiments_samplings" , "RNN-UNetSpherical-healpix-16-k20-MaxAreaPooling-float32-AR6-LinearStep", "model_skills/deterministic_global_skill.nc"))
RNN_icos_skills = xr.open_dataset(os.path.join(base_dir, "experiments_samplings" , "RNN-UNetSpherical-icosahedral-16-k20-MaxAreaPooling-float32-AR6-LinearStep", "model_skills/deterministic_global_skill.nc"))

##-----------------------------------------------------------------------------.
# Load Weyn benchmark 
weyn_skills = xr.open_dataset("/data/weather_prediction/Benchmarks/rmses_weyn.nc")
weyn_skills["lead_time"] = weyn_skills.lead_time.values*np.timedelta64(1, 'h')
weyn_skills = weyn_skills.rename({'lead_time':'leadtime'})
weyn_skills = weyn_skills.expand_dims({"skill": ['RMSE']})

## Load climatology and persistence baselines 
sampling_name = "Equiangular_400km" # All similar btw 
benchmark_dir = os.path.join(base_dir, "data", sampling_name, "Benchmarks")
persistence_skills = xr.open_dataset(os.path.join(benchmark_dir, "Persistence_Global_Skills.nc"))
WeeklyClimatology_skills = xr.open_dataset(os.path.join(benchmark_dir, "WeeklyClimatology_Global_Skills.nc"))
DailyClimatology_skills = xr.open_dataset(os.path.join(benchmark_dir, "DailyClimatology_Global_Skills.nc"))
MonthlyClimatology_skills = xr.open_dataset(os.path.join(benchmark_dir, "MonthlyClimatology_Global_Skills.nc"))
HourlyWeeklyClimatology_skills = xr.open_dataset(os.path.join(benchmark_dir, "HourlyWeeklyClimatology_Global_Skills.nc"))
HourlyMonthlyClimatology_skills = xr.open_dataset(os.path.join(benchmark_dir, "HourlyMonthlyClimatology_Global_Skills.nc"))

## Define the dictionary of forecast to benchmark 
skills_dict = {
               # 'Planar Projection': planar_skills,
               # 'Cylindrical Projection': cylinder_kills,
               # 'DeepSphere - Equiangular': graph_skills,
               'Equiangular': RNN_equi_skills,
               'Reduced Gaussian': RNN_gauss_skills,
               'Cubed': RNN_cubed_skills,
               'Healpix': RNN_healpix_skills,
               'Icosahedral': RNN_icos_skills,
               # 'Weyn et al., 2020': weyn_skills,
               'Persistence forecast': persistence_skills,
               'Weekly Climatology': WeeklyClimatology_skills,
               'HourlyWeekly Climatology': HourlyWeeklyClimatology_skills,
               #'Monthly Climatology': MonthlyClimatology_skills,
               #'Daily Climatology': DailyClimatology_skills,
               #'HourlyMonthly Climatology': HourlyMonthlyClimatology_skills,
               }

## Define the dictionary of forecast to benchmark (without Weyn)
skills_dict1 = {
               # 'Planar Projection': planar_skills,
               # 'Cylindrical Projection': cylinder_kills,
               # 'DeepSphere - Equiangular': graph_skills,
               'Equiangular': RNN_equi_skills,
               'Reduced Gaussian': RNN_gauss_skills,
               'Cubed': RNN_cubed_skills,
               'Healpix': RNN_healpix_skills,
               'Icosahedral': RNN_icos_skills,
               # 'Weyn et al., 2020': weyn_skills,
               'Persistence forecast': persistence_skills,
               'Weekly Climatology': WeeklyClimatology_skills,
               'HourlyWeekly Climatology': HourlyWeeklyClimatology_skills,
               #'Monthly Climatology': MonthlyClimatology_skills,
               #'Daily Climatology': DailyClimatology_skills,
               #'HourlyMonthly Climatology': HourlyMonthlyClimatology_skills,
               }

colors_dict = {'Planar Projection':  "blue" ,
               'Cylindrical Projection': "aqua",
               'DeepSphere - Equiangular': "dodgerblue",
               'Equiangular': "dodgerblue",
               'Reduced Gaussian': "forestgreen",
               'Cubed': "orange",
               'Icosahedral': "red",
               'Healpix': "fuchsia",
               'Weyn et al., 2020': "darkviolet",
               'Persistence forecast': "gray",
               'Weekly Climatology': "gray",
               'HourlyWeekly Climatology': "gray",
               'Monthly Climatology': "gray",
               'Daily Climatology': "gray",
               'HourlyMonthly Climatology': "gray",
}

# Benchmark skills 
benchmark_global_skill(skills_dict=skills_dict, 
                       skill="RMSE", 
                       variables=['z500','t850'],
                       colors_dict = colors_dict,
                       n_leadtimes=20).savefig(os.path.join(figs_dir, "Benchmark_RMSE_Samplings.png"))


benchmark_global_skills(skills_dict=skills_dict1, 
                        skills=['BIAS','RMSE','rSD','pearson_R2'],
                        variables=['z500','t850'],
                        colors_dict = colors_dict,
                        legend_everywhere = True,
                        n_leadtimes=20).savefig(os.path.join(figs_dir, "Benchmark_Overview_Samplings.png"))






## Define the dictionary of forecast to benchmark 
skills_dict = {
               # 'Planar Projection': planar_skills,
               # 'Cylindrical Projection': cylinder_kills,
               # 'DeepSphere - Equiangular': graph_skills,
               'Equiangular': RNN_equi_skills,
               'Reduced Gaussian': RNN_gauss_skills,
               'Cubed': RNN_cubed_skills,
               'Healpix': RNN_healpix_skills,
               'Icosahedral': RNN_icos_skills,
               'Weyn et al., 2020': weyn_skills,
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
                       colors_dict = colors_dict,
                       n_leadtimes=20).savefig(os.path.join(figs_dir, "Benchmark_RMSE_Samplings_Weyn.png"))




# Others skills 
# benchmark_global_skills(skills_dict=skills_dict, 
#                         skills=['relBIAS','MAE','relMAE','diffSD','NSE', 'error_CoV'],
#                         variables=['z500','t850'],
#                         legend_everywhere = True,
#                         n_leadtimes=20)

equi_dict = {'DeepSphere - Equiangular': graph_skills,
             'Equiangular': RNN_equi_skills,
             #    'Reduced Gaussian': RNN_gauss_skills,
             #    'Healpix': RNN_healpix_skills,
             #    'Icosahedral': RNN_icos_skills,
            'Weyn et al., 2020': weyn_skills,
            'Persistence forecast': persistence_skills,
            'Weekly Climatology': WeeklyClimatology_skills,
            'HourlyWeekly Climatology': HourlyWeeklyClimatology_skills,
               #'Monthly Climatology': MonthlyClimatology_skills,
               #'Daily Climatology': DailyClimatology_skills,
               #'HourlyMonthly Climatology': HourlyMonthlyClimatology_skills,
               }
benchmark_global_skill(skills_dict=equi_dict, 
                       skill="RMSE", 
                       variables=['z500','t850'],
                       colors_dict = colors_dict,
                       n_leadtimes=20) 