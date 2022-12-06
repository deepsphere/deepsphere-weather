import pickle
import os

os.chdir("/home/ghiggi/Projects/deepsphere-weather")


fpath = "/data/weather_prediction/experiments_GG/new/RNN-AR6-UNetSpherical-Healpix_400km-Graph_knn-k20-MaxAreaPooli/model_weights/AR_TrainingInfo.pickle"

with open(fpath, "rb") as handle:
    ar_training_info = pickle.load(handle)

ar_training_info.plots("/home/ghiggi/Projects/deepsphere-weather/")
ar_training_info.plots(ylim=(0, 0.6))

ar_training_info.plot_loss_per_ar_iteration(
    ar_iteration=1,
    linestyle="dashed",
    linewidth=0.3,
    xlim=(2000, 10000),
    ylim=(0.005, 0.014),
    plot_training=False,
    plot_labels=True,
    plot_legend=True,
    add_ar_weights_updates=True,
)
