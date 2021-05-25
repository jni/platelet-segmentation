# coding: utf-8
from tqdm import tqdm
import itertools
import torch
import numpy as np
import nd2_dask as nd2
import napari
from skimage.exposure import rescale_intensity

import unet

u = unet.UNet(in_channels=1, out_channels=5)
u.load_state_dict(torch.load('/data/platelets-deep/210525_141407_basic_z-1_y-1_x-1_m_centg/212505_142127_unet_210525_141407_basic_z-1_y-1_x-1_m_centg.pt'))
layer_list = nd2.nd2_reader.nd2_reader('/data/platelets/200519_IVMTR69_Inj4_dmso_exp3.nd2')

IGNORE_CUDA = False

if torch.cuda.is_available() and not IGNORE_CUDA:
    u.cuda()

t_idx = 114

source_vol = layer_list[2][0]
vol2predict = rescale_intensity(
        np.asarray(source_vol[t_idx])
        ).astype(np.float32)
prediction_output = np.zeros((5,) + vol2predict.shape, dtype=np.float32)

chunk_start_0 = (0, 6, 12, 18, 23)
chunk_start_1 = (0, 128, 256)
chunk_start_2 = (0, 128, 256)
crops_0 = [(0, 8), (2, 8), (2, 8), (2, 8), (3, 10)]
crops_1 = [(0, 192), (64, 192), (64, 256)]
crops_2 = [(0, 192), (64, 192), (64, 256)]
chunk_starts = list(itertools.product(chunk_start_0, chunk_start_1, chunk_start_2))
chunk_crops = list(itertools.product(crops_0, crops_1, crops_2))
size = (10, 256, 256)


for i, (start, crop) in tqdm(enumerate(zip(chunk_starts, chunk_crops))):
    sl = tuple(slice(start, start+step) for start, step
               in zip(start, size))
    tensor = torch.from_numpy(vol2predict[sl][np.newaxis, np.newaxis])
    if torch.cuda.is_available() and not IGNORE_CUDA:
        tensor = tensor.cuda()
    predicted_array = u(tensor).cpu().detach().numpy()
    # add slice(None) for the 5 channels
    cr = (slice(None),) + tuple(slice(i, j) for i, j in crop)
    prediction_output[(slice(None),) + sl][cr] = predicted_array[(0,) + cr]

viewer = napari.Viewer()
l0 = viewer._add_layer_from_data(*layer_list[0])
l1 = viewer._add_layer_from_data(*layer_list[1])
l2 = viewer._add_layer_from_data(*layer_list[2])

offsets = -0.5 * np.asarray(l0.scale)[-3:] * np.eye(5, 3)
viewer.add_image(
        prediction_output,
        channel_axis=0,
        name=['z-aff', 'y-aff', 'x-aff', 'mask', 'centroids'],
        scale=l0.scale[-3:],
        translate=list(np.asarray(l0.translate[-3:]) + offsets),
        colormap=['bop purple', 'bop orange', 'bop orange', 'gray', 'gray'],
        visible=[False, False, False, True, False],
        )
viewer.dims.set_point(0, t_idx)

napari.run()
