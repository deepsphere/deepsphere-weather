

ds_obs = xr.open_zarr(os.path.join(data_sampling_dir, "Data","dynamic", "time_chunked", "dynamic.zarr")) 

ds_pred = ds_forecasts.isel(time=slice(0,365*4))

ds_pred, ds_obs = xr.align(ds_pred, ds_obs)
ds_pred = ds_pred.compute()
ds_obs = ds_obs.compute()

# ds_pred = ds_pred.isel(time=slice(0,50))
# ds_pred, ds_obs = xr.align(ds_pred, ds_obs)

# ds_obs = hourly_weekly_anomaly_scaler.transform(ds_obs)
# ds_pred = hourly_weekly_anomaly_scaler.transform(ds_pred)

# ds_err = ds_pred - ds_obs 

# %matplotlib inline

time_groups = None 

# time_groups = {"hour": 1 }      
# time_groups = ["dayofyear"]
# time_groups = ["weekofyear"]
# time_groups = ["month"]

hovmoller_pred = HovmollerDiagram(ds_pred, 
                                  time_dim = "time", 
                                  time_groups = time_groups,
                                  spatial_dim = "lat", bin_width = 5,
                                  time_average_before_binning = True)
hovmoller_obs = HovmollerDiagram(ds_obs, 
                             time_dim = "time", 
                             time_groups = time_groups,
                             spatial_dim = "lat", bin_width = 5,
                             time_average_before_binning = True)

hovmoller_diff = hovmoller_pred - hovmoller_obs

# hovmoller_err = HovmollerDiagram(ds_err, 
#                                  time_dim = "time", 
#                                  time_groups = time_groups,
#                                  spatial_dim = "lat", bin_width = 5,
#                                  time_average_before_binning = True)
# diff_hov = hovmoller_diff - hovmoller_err
 
### Overview of hovmoller plots
vars = list(hovmoller_obs.data_vars.keys())
vars = ["z500", "t850"]
fig, axs = plt.subplots(len(vars),3)
axs = axs.flatten()
for i, var in enumerate(vars):
    # Plot obs 
    hovmoller_obs[var].plot(ax=axs[i*3], add_colorbar=False)
    axs[i*3].set_ylabel("Latitude")
    if i == len(vars):
        axs[i*3].set_title("Observed")
    else: 
        axs[i*3].set_title("")
    if i != len(vars):
        axs[i*3].get_xaxis().set_visible(False)
    # Plot pred
    hovmoller_pred[var].plot(ax=axs[i*3+1], add_colorbar=True)
    axs[i*3+1].set_ylabel("")
    if i == len(vars):
        axs[i*3+1].set_title("Predicted")
    else: 
        axs[i*3+1].set_title("")
    if i != len(vars):
        axs[i*3+1].axis('off')
    # Plot difference
    hovmoller_diff[var].plot(ax=axs[i*3+2], add_colorbar=True)
    axs[i*3+2].set_ylabel("")
    if i == 0:
        axs[i*3+2].set_title("Difference")
    else: 
        axs[i*3+2].set_title("")
    if i != len(vars):
        axs[i*3+2].axis('off')
plt.show()



fig, ax = plt.subplots()
hovmoller_obs['q500'].plot(ax=ax)
ax.set_ylabel("Latitude")
ax.set_title("Observed")
plt.show()

fig, ax = plt.subplots()
hovmoller_pred['z500'].plot(ax=ax)
ax.set_ylabel("Latitude")
ax.set_title("Predicted")
plt.show()

fig, ax = plt.subplots()
hovmoller_diff['z500'].plot(ax=ax)
ax.set_ylabel("Latitude")
plt.show()

# fig, ax = plt.subplots()
# hovmoller_err['z500'].plot(ax=ax)
# ax.set_ylabel("Latitude")
# plt.show()

# fig, ax = plt.subplots()
# diff_hov['z500'].plot(ax=ax)
# ax.set_ylabel("Latitude")
# plt.show()

spatial_dim = "lat"
time_dim = "time"
bin_edges=None
bin_width=5
time_groups=time_groups
time_average_before_binning=True
variable_dim=None

# fig, ax = plt.subplots()
# hovmoller_err['z500'].plot(x="lon_bins", y="time", ax=ax)
# ax.set_xlabel("Longitude")
# plt.show()

# Check time_average_before_binning when having >1 year data 
hovmoller_pred1 = HovmollerDiagram(ds, 
                                  time_dim = "time", 
                                  time_groups = time_groups,
                                  spatial_dim = "lat", bin_width = 5,
                                  time_average_before_binning = False)
d = hovmoller_pred - hovmoller_pred1
fig, ax = plt.subplots()
d['z500'].plot(ax=ax)
ax.set_ylabel("Latitude")
plt.show() 
 # plot 
 # contour 
 # conoturf 

## - Hovmoller 
# Diurnal cycle (over 1 week) 
# Annual simulation (original t_res) 
# Annual simulation (daily mean) 

# Animation:  left side: map over time,  right side: howmoller diagram 

hovmoller_pred = HovmollerDiagram(da, 
                                  time_dim = "time", 
                                  time_groups = "month",
                                  spatial_dim = "lat", bin_width = 5,
                                  time_average_before_binning = True) # xarray bug ... 

hovmoller_pred = HovmollerDiagram(da, 
                                  time_dim = "time", 
                                  time_groups = time_groups,
                                  spatial_dim = "lat", bin_width = 5,
                                  time_average_before_binning = True,
                                  variable_dim = "variable")