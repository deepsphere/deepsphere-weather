
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
figs_dir = "/data/weather_prediction/experiments_GG/new_old_archi"
 
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
new_dir = "/data/weather_prediction/experiments_GG/new_old_archi"
RNN_not_well_trained1_skills = xr.open_dataset(os.path.join(new_dir,"OLD_fine_tuned1-RNN-AR6-UNetSpherical-Healpix_400km-Graph_knn-k20-MaxAreaPooling","model_skills/deterministic_global_skill.nc"))
RNN_well_trained_skills = xr.open_dataset(os.path.join(new_dir,"OLD_fine_tuned-RNN-AR6-UNetSpherical-Healpix_400km-Graph_knn-k20-MaxAreaPooling","model_skills/deterministic_global_skill.nc"))
RNN_well_trained2_skills = xr.open_dataset(os.path.join(new_dir, "OLD_fine_tuned2-RNN-AR6-UNetSpherical-Healpix_400km-Graph_knn-k20-MaxAreaPooling","model_skills/deterministic_global_skill.nc"))
RNN_well_trained3_skills = xr.open_dataset(os.path.join(new_dir,"OLD_fine_tuned3_-RNN-AR6-UNetSpherical-Healpix_400km-Graph_knn-k20-MaxAreaPooling","model_skills/deterministic_global_skill.nc"))
RNN_well_trained3disal_skills = xr.open_dataset(os.path.join(new_dir,"OLD_fine_tuned1_disal-RNN-AR6-UnetSpherical-Healpix_400km-Graph_knn-k20-MaxAreaPooling","model_skills/deterministic_global_skill.nc"))

RNN_ReZero_skills = xr.open_dataset(os.path.join(new_dir,"OLD_fine_tuned4_ReZero-RNN-AR6-UNetSpherical-Healpix_400km-Graph_knn-k20-MaxAreaPooling","model_skills/deterministic_global_skill.nc"))

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
MonthlyClimatology_skills = xr.open_dataset(os.path.join(benchmark_dir, "MonthlyClimatology_Global_Skills.nc"))
# DailyClimatology_skills = xr.open_dataset(os.path.join(benchmark_dir, "DailyClimatology_Global_Skills.nc"))
# HourlyWeeklyClimatology_skills = xr.open_dataset(os.path.join(benchmark_dir, "HourlyWeeklyClimatology_Global_Skills.nc"))
# HourlyMonthlyClimatology_skills = xr.open_dataset(os.path.join(benchmark_dir, "HourlyMonthlyClimatology_Global_Skills.nc"))

## Define the dictionary of forecast to benchmark 
skills_dict = {
               # 'Planar Projection': planar_skills,
               # 'Cylindrical Projection': cylinder_kills,
               # 'DeepSphere - Equiangular': graph_skills,
            #    'Equiangular': RNN_equi_skills,
            #    'Reduced Gaussian': RNN_gauss_skills,
            #    'Cubed': RNN_cubed_skills,
               'Healpix': RNN_healpix_skills,
            #    'Icosahedral': RNN_icos_skills,
               'Not well trained RNN1': RNN_not_well_trained1_skills,
               'Well_trained RNN': RNN_well_trained_skills,
               'Well_trained RNN2': RNN_well_trained2_skills,
               'Well_trained RNN3': RNN_well_trained3_skills,
               'Well_trained RNN3_disal': RNN_well_trained3disal_skills,
               'ReZero encoder': RNN_ReZero_skills,

               'Weyn et al., 2020': weyn_skills,
               'Persistence forecast': persistence_skills,
               'Monthly Climatology': MonthlyClimatology_skills,
               'Weekly Climatology': WeeklyClimatology_skills,
               #'HourlyWeekly Climatology': HourlyWeeklyClimatology_skills,
               #'Daily Climatology': DailyClimatology_skills,
               #'HourlyMonthly Climatology': HourlyMonthlyClimatology_skills,
               }
skills_dict1 = {
               # 'Planar Projection': planar_skills,
               # 'Cylindrical Projection': cylinder_kills,
               # 'DeepSphere - Equiangular': graph_skills,
            #    'Equiangular': RNN_equi_skills,
            #    'Reduced Gaussian': RNN_gauss_skills,
            #    'Cubed': RNN_cubed_skills,
               'Healpix': RNN_healpix_skills,
            #    'Icosahedral': RNN_icos_skills,
               'Not well trained RNN1': RNN_not_well_trained1_skills,
               'Well_trained RNN': RNN_well_trained_skills,
               'Well_trained RNN2': RNN_well_trained2_skills,
               'Well_trained RNN3': RNN_well_trained3_skills,
               'Well_trained RNN3_disal': RNN_well_trained3disal_skills,
               'ReZero encoder': RNN_ReZero_skills,
               'ReZero encoder': RNN_ReZero_skills,
               'Persistence forecast': persistence_skills,
               'Monthly Climatology': MonthlyClimatology_skills,
               'Weekly Climatology': WeeklyClimatology_skills,
               #'HourlyWeekly Climatology': HourlyWeeklyClimatology_skills,
               #'Daily Climatology': DailyClimatology_skills,
               #'HourlyMonthly Climatology': HourlyMonthlyClimatology_skills,
               }

colors_dict = {'Planar Projection':  "blue" ,
               'Cylindrical Projection': "aqua",
            #    'DeepSphere - Equiangular': "dodgerblue",
            #    'Equiangular': "dodgerblue",
            #    'Reduced Gaussian': "forestgreen",
            #    'Cubed': "orange",
            #    'Icosahedral': "red",
               'Healpix': "fuchsia",
        
               'Not well trained RNN1': "yellow",
               'Well_trained RNN': "black",
               'Well_trained RNN2': "dodgerblue",
               'Well_trained RNN3': "forestgreen",
               'Well_trained RNN3_disal': "orange",
               'ReZero encoder': "red",

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
                       n_leadtimes=20).savefig(os.path.join(figs_dir, "Benchmark_RMSE_Samplings1.png"))


benchmark_global_skills(skills_dict=skills_dict1, 
                        skills=['BIAS','RMSE','rSD','pearson_R2'],
                        variables=['z500','t850'],
                        colors_dict = colors_dict,
                        legend_everywhere = True,
                        n_leadtimes=20).savefig(os.path.join(figs_dir, "Benchmark_Overview_Samplings1.png"))
 