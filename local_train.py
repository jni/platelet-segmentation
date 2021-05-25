import numpy as np
import os
import torch
import napari
import train
import zarr


# ------------------
# Stable Local Paths
# ------------------
# Directory for training data and network output
data_dir = '/Users/jni/data/platelets-deep'
# Path for original image volumes for which GT was generated
image_paths = [os.path.join(data_dir,
    '191113_IVMTR26_Inj3_exp3_t58_cang_training_image.zarr')] #,
# Path for GT labels volumes
labels_paths = [os.path.join(data_dir,
    '191113_IVMTR26_Inj3_exp3_t58_cang_training_labels.zarr')]#,


# -----------------------
# CHANGE THESE EACH TIME:
# ---------------------------------------------------------------------------------------
METHOD = 'get'
out_dir = os.path.join(data_dir, '210525_training_0')
suffix = 'z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl'
# ---------------------------------------------------------------------------------------


if METHOD == 'get':
    # --------------------
    # CAN CHANGE THESE TOO
    # --------------------
    n_each = 100
    channels = ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'centreness', 'centreness-log')
    validation_prop = 0.2
    scale = (4, 1, 1)
    epochs = 4
    lr = .01
    loss_function = 'BCELoss'
    chan_weights = (1., 2., 2.) # only used for weighted BCE
    weights = None # can load and continue training
    update_every = 20 # how many batches before printing loss
    # -----------
    # Train U-net
    # -----------
    # with newly generated label chunks
    unet = train.train_unet_get_labels(
                                       out_dir,
                                       image_paths,
                                       labels_paths,
                                       suffix=suffix,
                                       n_each=n_each,
                                       channels=channels,
                                       validation_prop=validation_prop,
                                       scale=scale,
                                       epochs=epochs,
                                       lr=lr,
                                       loss_function=loss_function,
                                       chan_weights=chan_weights,
                                       weights=weights,
                                       update_every=update_every
                                       )
    image = zarr.open(image_paths[0])
    viewer = napari.view_image(image)
    napari.run()


if METHOD == 'load':
    # to be continued ...
    pass
