##----------------------------------------------------------------------------.                              
# Check Datasets have mesh attached 
if 'mesh' not in list(ds_forecast.coords.keys()):
    raise ValueError("No 'mesh' coordinate in ds_forecast.")
if 'mesh' not in list(ds_obs.coords.keys()):
    raise ValueError("No 'mesh' coordinate in ds_obs.")
##----------------------------------------------------------------------------.
# Retrieve forecast reference time 
forecast_reference_time = str(ds_forecast['forecast_reference_time'].values.astype('datetime64[s]')) # remove nanosecs

# Load data into memory
ds_forecast = ds_forecast.load()

# Retrieve valid time 
valid_time = ds_forecast['forecast_reference_time'].values + ds_forecast['leadtime'].values
ds_forecast = ds_forecast.assign_coords({'time': ('leadtime', valid_time)})
ds_forecast = ds_forecast.swap_dims({'leadtime': 'time'})

# Subset observations and load in memory
ds_obs = ds_obs.sel(time=ds_forecast['time'].values)
ds_obs = ds_obs.load()

# Compute error 
ds_error = ds_forecast - ds_obs 
ds_error = ds_error.assign_coords({'mesh': ("node", ds_obs['mesh'].values)})

# Create a dictionary with relevant infos  
ds_dict = {"pred": ds_forecast, "obs": ds_obs, "error": ds_error}

# Retrieve common variables to plot 
variables = list(ds_forecast.data_vars.keys())

##----------------------------------------------------------------------------.
# Check GIF fpath 
if not os.path.exists(os.path.dirname(GIF_fpath)):
    os.makedirs(os.path.dirname(GIF_fpath))
    
# Check GIF fpath ends with .gif
if GIF_fpath[-4:] != ".gif":
    print("Added .gif to GIF_fpath.")
    GIF_fpath = GIF_fpath + ".gif"
    
# Create temporary directory to store temporary GIF image frames

##----------------------------------------------------------------------------.
# Create GIF image frames for each leadtime 
for i in range(len(ds_forecast['leadtime'])):
    # Select frame super title 
    tmp_leadtime = str(ds_forecast['leadtime'].values[i].astype('timedelta64[h]'))
    tmp_valid_time = str(ds_forecast['time'].values[i].astype('datetime64[s]'))
    suptitle_str = "Forecast reference time: {}, Leadtime: {}".format(forecast_reference_time, tmp_leadtime)
    ##------------------------------------------------------------------------.
    # Create figure 
    fig, axs = plt.subplots(nrows=len(variables), ncols=3, 
                    figsize=(18, 4*len(variables)),
                    subplot_kw={'projection': ccrs.Robinson()})
    fig.suptitle(suptitle_str)
    # fig.subplots_adjust(wspace=0.1, hspace=0.2)
    ##------------------------------------------------------------------------.
    # Initialize 
    axs = axs.flatten()
    ax_count = 0
    ##------------------------------------------------------------------------.
    # Plot each variable
    for var in variables:
        # Plot obs 
        tmp_obs = ds_dict['obs'][var].isel(time=i)
        xsphere._plot(tmp_obs,
                        ax=axs[ax_count],
                        edgecolors = edgecolors,
                        antialiased = antialiased,
                        vmin=get_var_clim(var,'state')[0],
                        vmax=get_var_clim(var,'state')[1],
                        cmap=get_var_cmap(var,'state')
                        )
        axs[ax_count].set_title(None)
        axs[ax_count].coastlines(alpha=0.2)
        # Plot pred 
        tmp_pred = ds_dict['pred'][var].isel(time=i)
        s_p = xsphere._plot(tmp_pred,
                            ax=axs[ax_count+1],
                            edgecolors = edgecolors, 
                            antialiased = antialiased,
                            vmin=get_var_clim(var,'state')[0],
                            vmax=get_var_clim(var,'state')[1],
                            cmap=get_var_cmap(var,'state'),
                            )
        axs[ax_count+1].set_title(None)    
        axs[ax_count+1].coastlines(alpha=0.2)
        # - Add state colorbar
        cbar = fig.colorbar(s_p, ax=axs[[ax_count, ax_count+1]], 
                            orientation="horizontal", 
                            extend = 'both',
                            aspect=aspect_cbar)       
        cbar.set_label(var.upper())
        cbar.ax.xaxis.set_label_position('top')
        # Plot error 
        tmp_error = ds_dict['error'][var].isel(time=i)
        e_p = xsphere._plot(tmp_error,
                            ax=axs[ax_count+2],
                            edgecolors = edgecolors, 
                            antialiased = antialiased,
                            vmin=get_var_clim(var,'error')[0],
                            vmax=get_var_clim(var,'error')[1],
                            cmap=get_var_cmap(var,'error'),
                            # add_colorbar = True, 
                            # cbar_kwargs={'orientation': 'horizontal',
                            #              'label': var.upper() + " Error"}
                            )
        axs[ax_count+2].set_title(None)
        axs[ax_count+2].coastlines(alpha=0.2)
        # - Add error colorbar
        # cb = plt.colorbar(e_p, ax=axs[ax_count+2], orientation="horizontal") # pad=0.15)
        # cb.set_label(label=var.upper() + " Error") # size='large', weight='bold'
        cbar_err = fig.colorbar(e_p, ax=axs[ax_count+2],
                                orientation="horizontal",
                                extend = 'both',
                                aspect = aspect_cbar/2)      
        cbar_err.set_label(var.upper() + " Error")
        cbar_err.ax.xaxis.set_label_position('top')
        # Add plot labels 
        # if ax_count == 0: 
        axs[ax_count].set_title("Observed")     
        axs[ax_count+1].set_title("Predicted")  
        axs[ax_count+2].set_title("Error")
        # Update ax_count 
        ax_count += 3
    ##------------------------------------------------------------------------.    
    # Save figure in temporary directory 
    fig.savefig(os.path.join(tmp_dir, '{:04}.png'.format(i)), dpi=200)
##----------------------------------------------------------------------------. 
# Load all figures (in PIL PngImageFile)
im_fpaths = glob.glob(tmp_dir + "/" + '*.png')
im_fpaths.sort()
img, *l_imgs = [Image.open(fpath) for fpath in im_fpaths]
##----------------------------------------------------------------------------.
# Create a GIF
img.save(fp = GIF_fpath, 
        format='GIF', 
        append_images = l_imgs,
        save_all=True, 
        optimize=False,
        duration=0.5*1000,  # The time to display the current frame of the GIF, in milliseconds
        loop=0) # Number of times the GIF should loop 



##  gif format only support 256 color variations
# https://homehack.nl/create-animated-gifs-from-mp4-with-ffmpeg/
# https://medium.com/swlh/how-to-create-animated-gifs-with-ffmpeg-29467362cdc1
# https://medium.com/@Peter_UXer/small-sized-and-beautiful-gifs-with-ffmpeg-25c5082ed733