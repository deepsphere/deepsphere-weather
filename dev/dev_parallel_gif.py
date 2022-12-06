       
def create_gif_forecast_evolution(gif_fpath,
                                  dict_data, 
                                  # Plot options 
                                  aspect_cbar = 40,
                                  antialiased = False,
                                  edgecolors = None,
                                  # GIF options 
                                  fps = 10,
                                  create_gif=True):
    ##------------------------------------------------------------------------.
    # Check dict_data 
    if not isinstance(dict_data, dict):
        raise TypeError("dict_data must be a dictionary with sub-keys ('data','clim','cmap').")
    
    for k in dict_data.keys(): 
        # Check there is a subdictionary with keys ('data','clim','cmap')
        if not isinstance(dict_data[k], dict): 
            raise ValueError("For {!r}', you must specify a subdictionary with keys ('data','clim','cmap').".format(k)) 
        required_subkeys = np.array(['data','clim','cmap'])
        subkeys = list(dict_data[k].keys())
        missing_subkeys = required_subkeys[np.isin(required_subkeys, subkeys, invert=True)]
        if len(missing_subkeys) > 0: 
            raise ValueError("{!r} miss the following subkeys: {}".format(k, missing_subkeys))
        
        # Check data are in the expected format 
        x = dict_data[k]['data']
        if not x.sphere.has_mesh():
            raise ValueError("No 'mesh' coordinate in data.") 
        if not isinstance(x, (xr.DataArray, xr.Dataset)):
            raise TypeError("Provide a list of xr.DataArray or xr.Dataset objects.") 
        if not xr_has_dim(x, dim="time"): 
            raise ValueError("Provide xarray objects with the 'time' dimension.")
        if isinstance(x, xr.Dataset):
            if xr_n_vars(x) != 1: 
                raise ValueError("Unless providing a single xr.Dataset, each xr.Dataset in the provided list must have an unique variable.")
            # Convert to DataArray
            dict_data[k]['data'] = x[xr_Dataset_vars(x)[0]]
    
    ##------------------------------------------------------------------------.
    # Align DataArray         
    l_DataArrays = [dict_data[k]['data'] for k in dict_data.keys()]                    
    l_DataArrays = list(xr.align(*l_DataArrays))
    
    ##------------------------------------------------------------------------.
    # Load the data
    for i, k in enumerate(list(dict_data.keys())): 
        dict_data[k]['data'] = l_DataArrays[i].compute()
        
    ##------------------------------------------------------------------------.
    # Compute hovmollers
    print("- Computing hovmoller diagram")
    dict_hovmollers = {}
    for k, d in dict_data.items(): 
        dict_hovmollers[k] = HovmollerDiagram(d['data'], 
                                                time_dim = "time", 
                                                time_groups = None,
                                                spatial_dim = "lat", bin_width = 5,
                                                time_average_before_binning = True)
        
    ##------------------------------------------------------------------------.
    # Check GIF fpath 
    if not os.path.exists(os.path.dirname(gif_fpath)):
        os.makedirs(os.path.dirname(gif_fpath))
    
    # Remove gif file format 
    if gif_fpath[-4:] == ".gif":
        gif_fpath = gif_fpath[:-4]
    
    # Create temporary directory to store temporary GIF image frames
    tmp_dir = tempfile.mkdtemp()
    
    ##-----------------------------------------------------------------------.
    # Retrieve timesteps 
    timesteps = l_DataArrays[0]['time'].values 
    n_timesteps = len(timesteps)
    
    # Retrieve number of DataArrays to plot 
    n_rows = len(dict_data)
    
    ##-----------------------------------------------------------------------.
    # i = n_timesteps-1
    # antialiased = False
    # edgecolors = None
    ##-----------------------------------------------------------------------.
    ### Define figure generation 
    @dask.delayed()
    def figure_generation(i, dict_data, dict_hovmollers, cwd): #
        # This hack is required because otherwise the xarray accessor is lost
        # os.chdir(cwd)
        # import modules.xsphere
        ##----------------------------------------------------------------------.
        tmp_timestep = timesteps[i]
        tmp_timestep_str = str(tmp_timestep.astype('datetime64[m]'))
    
        ##----------------------------------------------------------------------.
        # Define figure settings
        # pix = 1/plt.rcParams['figure.dpi'] figsize=(1920*pix,1080*pix),
        figsize = (17, 4*n_rows)
        crs_proj = ccrs.Robinson()
        fig = plt.figure(figsize = figsize) 
        
        # Define figure grids
        gs = gridspec.GridSpec(nrows=n_rows, ncols=2, 
                               # height_ratios=[1, 6],
                               hspace=0.03, wspace=0.1)
        axs = []
        for row in range(n_rows):
            # Left plot (geographic map)
            axs.append(fig.add_subplot(gs[row, 0], projection=crs_proj))
            # Right plot (hovmoller plot)
            axs.append(fig.add_subplot(gs[row, 1]))
        
        ##---------------------------------------------------------------------.
        # Plot each DataArray 
        for row, (k, d) in enumerate(dict_data.items()):   
            # Retrieve infos 
            data = d['data']
            vmin, vmax = d['clim']
            cmap = d['cmap']
            clabel = k 
            cbar_kwargs={'label': clabel}
            # Plot field map  
            tmp_map = data.isel(time=i)
            tmp_map.sphere.plot(ax=axs[row*2],
                                edgecolors = edgecolors,
                                antialiased = antialiased,
                                vmin=vmin, 
                                vmax=vmax,
                                cmap=cmap, 
                               )
            axs[row*2].set_title(None)
            axs[row*2].coastlines(alpha=0.2)
            # Plot hovmoller 
            tmp_hovmoller = dict_hovmollers[k] 
            tmp_hovmoller = tmp_hovmoller.where(tmp_hovmoller.time <= tmp_timestep)
            # mask tmp_hovmoller till > i 
            _ = tmp_hovmoller.plot(ax=axs[row*2 + 1],
                                   vmin=vmin, 
                                   vmax=vmax,
                                   cmap=cmap, 
                                   add_colorbar=True,
                                   cbar_kwargs=cbar_kwargs,
                                  )
            axs[row*2 + 1].set_ylabel("Latitude")  
            axs[row*2 + 1].set_title(None)
            # Display title ony on top plot 
            if row == 0: 
               axs[row*2 + 0].set_title(tmp_timestep_str)
               axs[row*2 + 1].set_title("Hovmoller diagram")   
            # Display time axis only on the bottom plot 
            if row != n_rows - 1:
               axs[row*2 + 1].get_xaxis().set_visible(False)     
     
        ##---------------------------------------------------------------------.    
        # Save figure in temporary directory 
        fig.savefig(os.path.join(tmp_dir, '{:04}.png'.format(i)))
        # Close the figure 
        plt.close(fig)
        return None
        
    ### Create GIF image frames for each timestep in parallel     
    cwd = os.getcwd()
    tasks = [figure_generation(i, dict_data, dict_hovmollers, cwd) for i in  np.arange(0,100)]
    # tasks = [figure_generation(i, dict_data, dict_hovmollers, cwd) for i in  np.arange(0,n_timesteps)]
    with ProgressBar():
        # dask.compute(tasks,'processes', num_workers=50)
        dask.compute(tasks, scheduler='threading', num_workers=25) 
        # dask.compute(tasks, scheduler='synchronous') # TO DEBUG

 
    # with multiprocessing.Pool(processes=8) as pool:
    #     # _ = list(tqdm.tqdm(pool.imap(figure_generation, args), total=n_timesteps))
    #     pool.map(figure_generation, args)
        
    ##-------------------------------------------------------------------------.
    ## Create MP4 and GIF with FFMPEG 
    # --> YouTube and Vimeo won’t really appreciate video with < 0.5 FPS
    # --> Duplicate frames by specifying (again) the desired FPS before -codec 
    # -y : overwrite existing file 
    # -r:v 30 : write at 30 frames per seconds 
    # -r 4: write 4 frames per seconds 
    # -r:v 1/4 : write a frame every 4 seconds 
    # -codec:v lix264
    
    # Create MP4
    cmd = 'ffmpeg -r:v {} -i "{}/%04d.png" -codec:v libx264 -preset placebo -an -y "{}.mp4"'.format(fps, tmp_dir, gif_fpath)
    subprocess.run(cmd, shell=True)
    
    # Create GIF
    if create_gif:
        cmd = 'ffmpeg -i {}.mp4 -vf "fps={},scale=2560:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 -y {}.gif'.format(gif_fpath, fps, gif_fpath)
        subprocess.run(cmd, shell=True)
    
    ##-------------------------------------------------------------------------.
    # Remove temporary images 
    shutil.rmtree(tmp_dir)
    
 
    
 
 
    
 