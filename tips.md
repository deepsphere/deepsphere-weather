## How to mount from remote server into local machine

#### For macOS users
Download "FUSE for macOS" and SSHFS from the [FUSE for macOS website](https://osxfuse.github.io/)

In your local terminal, create a directory where you want to mount the files
```
mkidr my_project
```

Then, create the access
```
sshfs username@lts2gdk0.epfl.ch: path/to/folder my_project/
```

## xArray 
When standardizing data based on monthly/weekly anomalies: 

First, data to be standardized needs to be grouped based on the temporal dim we are interested in: 

```ds.groupby({time.month}) or ds.groupby({time.week})```

Then, if done in a single step (ds - ds_mean)/ds_std it won't work since it will create an extra dimension. 
The only way is: 

```
ds = ds.groupby({time.month}) - ds_mean.to_array(dim='level)
ds = ds.groupby({time.month}) / ds_std.to_array(dim='level)
ds.compute()
```
