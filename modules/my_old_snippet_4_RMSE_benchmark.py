# RMSE baselines
    rmses_baselines = pickle.load(open('../data/models/baselines/'+'rmse.pkl', 'rb'))
    
    rmses_rasp_direct = rmses_baselines['CNN (direct)']
    rmses_rasp_iter = rmses_baselines['CNN (iterative)']
    rmses_climatology = rmses_baselines['Climatology']
    rmses_weekly_clim = rmses_baselines['Weekly clim.']
    rmses_persistence = rmses_baselines['Persistence']
    rmses_ifs = rmses_baselines['Operational'].sel(lead_time=lead_times)
    rmses_ifs_t42 = rmses_baselines['IFS T42'].sel(lead_time=lead_times)
    rmses_ifs_t63 = rmses_baselines['IFS T63'].sel(lead_time=slice(lead_times[0], lead_times[-1]))
    rmses_weyn = xr.open_dataset(input_dir + 'rmses_weyn.nc').rename({'z500':'z', 't850':'t'})
    rmse_iciar = rmses_baselines['Iciar20']
    rmse_direct = rmses_baselines['Direct_ours']
     
    # RMSE baselines
    rmses_baselines = pickle.load(open('../data/models/baselines/'+'rmse.pkl', 'rb'))
    rmses_weyn = xr.open_dataset('/nfs_home/wefeng/obs/rmses_weyn.nc').rename({'z500':'z', 't850':'t'})
     
    # RMSE baselines
    rmses_baselines = pickle.load(open(input_dir+'mae.pkl', 'rb'))
    rmses_baselines = pickle.load(open(input_dir+'acc.pkl', 'rb'))  
 
    # RMSE
    rmses_weyn = xr.open_dataset(input_dir + 'rmses_weyn.nc')
    rmses_rasp = xr.open_dataset(input_dir + 'rmses_rasp.nc')
    rmses_climatology = xr.open_dataset(input_dir + 'rmses_clim.nc')
    rmses_weekly_clim = xr.open_dataset(input_dir + 'rmses_weekly_clim.nc')
    rmses_persistence = xr.open_dataset(input_dir + 'rmses_persistence.nc')
    rmses_tigge = xr.open_dataset(input_dir + 'rmses_tigge.nc')
    
    # RMSE baselines
    rmses_baselines = pickle.load(open(input_dir+'acc.pkl', 'rb'))
    
    rmses_rasp_direct = rmses_baselines['CNN (direct)']
    rmses_rasp_iter = rmses_baselines['CNN (iterative)']
    rmses_climatology = rmses_baselines['Climatology']
    rmses_weekly_clim = rmses_baselines['Weekly clim.']
    rmses_persistence = rmses_baselines['Persistence']
    rmses_ifs = rmses_baselines['Operational'].sel(lead_time=lead_times)
    rmses_ifs_t42 = rmses_baselines['IFS T42'].sel(lead_time=lead_times)
    rmses_ifs_t63 = rmses_baselines['IFS T63'].sel(lead_time=slice(lead_times[0], lead_times[-1]))
    rmses_weyn = xr.open_dataset(input_dir + 'rmses_weyn.nc')

    axs[0].plot(lead_times0, rmses_persistence.z.values, label='Persistence', linestyle='--')
    axs[0].plot(lead_times0, [rmses_climatology.z.values]*len(lead_times0), label='Global climatology', linestyle='--')
    axs[0].plot(lead_times0, [rmses_weekly_clim.z.values]*len(lead_times0), label='Weekly climatology', linestyle='--')
    axs[0].plot(lead_times0, rmses_ifs.z.values, label='Operational IFS', linestyle='--')
    #axs[0].plot(lead_times0, rmses_ifs_t42.z.values, label='IFS T42', linestyle='--')
    #axs[0].plot(rmses_ifs_t63.lead_time.values, rmses_ifs_t63.z.values, label='IFS T63', linestyle='--')
    axs[0].scatter([72, 120], rmses_rasp_direct.z.values, label='Rasp 2020 (direct)', color='maroon')
    axs[0].plot(lead_times0, rmses_rasp_iter.z.values, label='Rasp 2020 (iter)', linestyle='-')
    #axs[0].plot(lead_times0, rmses_weyn.z.values, label='Weyn 2020', linestyle='-')
    axs[0].plot(lead_times, rmses_spherical.z.values, label='Ours', color='black', marker='o')


    axs[0, 0].plot(lead_times, rmses_persistence.z.values, label='Persistence', linestyle='--')
    axs[0, 0].plot(lead_times, rmses_climatology.z.values, label='Global climatology', linestyle='--')
    axs[0, 0].plot(lead_times, rmses_weekly_clim.z.values, label='Weekly climatology', linestyle='--')
    axs[0, 0].plot(lead_times, rmses_tigge.z.values, label='Operational IFS', linestyle='--')
    axs[0, 0].plot(lead_times, rmses_weyn.z.values, label='Weyn 2020', linestyle='--')
    axs[0, 0].scatter([72, 120], rmses_rasp.z.values, label='Rasp 2020', marker='_', linewidth=5, s=60, color='maroon')
    axs[0, 0].plot(lead_times, rmse.z.values, label='Spherical', color='black')
