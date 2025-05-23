#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import multiprocessing
from multiprocessing.pool import Pool
from typing import Tuple, Union, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.configuration import default_num_processes
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class, nnUNetBaseDataset
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager
from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder, \
    get_filenames_of_train_images_and_targets

color_cycle = (
    "000000",
    "4363d8",
    "f58231",
    "3cb44b",
    "e6194B",
    "911eb4",
    "ffe119",
    "bfef45",
    "42d4f4",
    "f032e6",
    "000075",
    "9A6324",
    "808000",
    "800000",
    "469990",
)


def hex_to_rgb(hex: str):
    assert len(hex) == 6
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))


def generate_overlay(input_image: np.ndarray, segmentation: np.ndarray, mapping: dict = None,
                     color_cycle: Tuple[str, ...] = color_cycle,
                     overlay_intensity: float = 0.6):
    """
    image can be 2d greyscale or 2d RGB (color channel in last dimension!)

    Segmentation must be label map of same shape as image (w/o color channels)

    mapping can be label_id -> idx_in_cycle or None

    returned image is scaled to [0, 255] (uint8)!!!
    """
    # create a copy of image
    image = np.copy(input_image)

    if image.ndim == 2:
        image = np.tile(image[:, :, None], (1, 1, 3))
    elif image.ndim == 3:
        if image.shape[2] == 1:
            image = np.tile(image, (1, 1, 3))
        else:
            raise RuntimeError(f'if 3d image is given the last dimension must be the color channels (3 channels). '
                               f'Only 2D images are supported. Your image shape: {image.shape}')
    else:
        raise RuntimeError("unexpected image shape. only 2D images and 2D images with color channels (color in "
                           "last dimension) are supported")

    # rescale image to [0, 255]
    image = image - image.min()
    image = image / image.max() * 255

    # create output
    if mapping is None:
        uniques = np.sort(pd.unique(segmentation.ravel()))  # np.unique(segmentation)
        mapping = {i: c for c, i in enumerate(uniques)}

    for l in mapping.keys():
        image[segmentation == l] += overlay_intensity * np.array(hex_to_rgb(color_cycle[mapping[l]]))

    # rescale result to [0, 255]
    image = image / image.max() * 255
    return image.astype(np.uint8)


def generate_heatmap_overlay(image: np.ndarray, heatmap: np.ndarray, colormap_name: str = 'viridis',
                             alpha: float = 0.7, range_image: Tuple[float, float] = None,
                             range_heatmap: Tuple[float, float] = None) -> np.ndarray:
    """
    Generates a heatmap overlay on a 2D grayscale image.

    Args:
        image (np.ndarray): 2D grayscale numpy array.
        heatmap (np.ndarray): 2D numpy array (same spatial dimensions as image) for uncertainty.
        colormap_name (str): Matplotlib colormap name.
        alpha (float): Transparency of the heatmap.
        range_image (Tuple[float, float], optional): Min/max for image normalization. Defaults to None.
        range_heatmap (Tuple[float, float], optional): Min/max for heatmap normalization. Defaults to None.

    Returns:
        np.ndarray: The overlay image as uint8 array [0, 255].
    """
    if image.ndim != 2:
        raise ValueError(f"Input image must be 2D grayscale. Got shape: {image.shape}")
    if heatmap.ndim != 2:
        raise ValueError(f"Input heatmap must be 2D. Got shape: {heatmap.shape}")
    if image.shape != heatmap.shape:
        raise ValueError(f"Image and heatmap must have the same spatial dimensions. "
                         f"Got image shape: {image.shape}, heatmap shape: {heatmap.shape}")

    # Normalize image
    if range_image is None:
        img_min, img_max = np.min(image), np.max(image)
    else:
        img_min, img_max = range_image
    
    if img_max == img_min: # Avoid division by zero if image is flat
        image_norm = np.zeros_like(image, dtype=np.float32)
    else:
        image_norm = (image - img_min) / (img_max - img_min)
    image_norm = np.clip(image_norm, 0, 1)
    image_rgb = np.repeat(image_norm[:, :, np.newaxis], 3, axis=2) # Convert to RGB

    # Normalize heatmap
    if range_heatmap is None:
        map_min, map_max = np.min(heatmap), np.max(heatmap)
    else:
        map_min, map_max = range_heatmap

    if map_max == map_min: # Avoid division by zero if heatmap is flat
        heatmap_norm = np.zeros_like(heatmap, dtype=np.float32)
    else:
        heatmap_norm = (heatmap - map_min) / (map_max - map_min)
    heatmap_norm = np.clip(heatmap_norm, 0, 1)

    # Apply colormap
    cmap = plt.get_cmap(colormap_name)
    heatmap_rgb = cmap(heatmap_norm)[..., :3]  # Take only RGB, discard alpha from cmap output

    # Blend
    overlay = (1 - alpha) * image_rgb + alpha * heatmap_rgb
    overlay_clipped = np.clip(overlay, 0, 1)
    
    # Scale to [0, 255] uint8
    overlay_uint8 = (overlay_clipped * 255).astype(np.uint8)
    
    return overlay_uint8


def plot_heatmap_overlay(image_file: str, heatmap_file: str, image_reader_writer: BaseReaderWriter, output_file: str,
                         slice_selection_fn=select_slice_to_plot2, colormap_name: str = 'viridis',
                         alpha: float = 0.7):
    """
    Plots a 2D heatmap overlay for a selected slice from 3D image and heatmap volumes.

    Args:
        image_file (str): Path to the 3D image file.
        heatmap_file (str): Path to the 3D heatmap file.
        image_reader_writer (BaseReaderWriter): Instance to read image and heatmap files.
        output_file (str): Path to save the generated overlay image.
        slice_selection_fn (function): Function to select the slice index.
                                       Defaults to select_slice_to_plot2.
        colormap_name (str): Name of the colormap for the heatmap. Defaults to 'viridis'.
        alpha (float): Transparency of the heatmap overlay. Defaults to 0.7.
    """
    import matplotlib.pyplot as plt

    image, props_img = image_reader_writer.read_images((image_file,))
    image = image[0]  # Assuming single channel image

    # Reading heatmap - assuming it's like a single channel image.
    # If it's a different format or needs special handling, this part might need adjustment.
    # For example, if it's a .npz or .npy, a different reader logic would be needed.
    # For now, assume it can be read similarly to an image channel.
    try:
        heatmap, props_hm = image_reader_writer.read_images((heatmap_file,)) # Try reading as image
        heatmap = heatmap[0] # Assuming single channel heatmap
    except Exception as e:
        # Fallback for non-image formats, e.g. simple npy/npz
        if heatmap_file.endswith('.npy'):
            heatmap = np.load(heatmap_file)
        elif heatmap_file.endswith('.npz'):
            # Assuming the key in npz is 'data' or the first one if not 'data'
            with np.load(heatmap_file) as npz_file:
                if 'data' in npz_file:
                    heatmap = npz_file['data']
                elif npz_file.files: # Take the first array if 'data' key doesn't exist
                     heatmap = npz_file[npz_file.files[0]]
                else:
                    raise IOError(f"Cannot read heatmap from NPZ file: {heatmap_file}. No arrays found or 'data' key missing.") from e
        else:
            raise IOError(f"Unsupported heatmap file format: {heatmap_file}. Error: {e}")


    if image.ndim != 3:
        raise ValueError(f"Image must be 3D. Got shape: {image.shape} for file {image_file}")
    if heatmap.ndim != 3:
        # If heatmap is 4D (e.g. N,C,H,W,D with N=C=1), try squeezing.
        if heatmap.ndim == image.ndim + 1 and heatmap.shape[0] == 1: # N,H,W,D vs H,W,D
             heatmap = heatmap[0]
        elif heatmap.ndim == image.ndim + 2 and heatmap.shape[0]==1 and heatmap.shape[1]==1: # N,C,H,W,D vs H,W,D
             heatmap = heatmap[0,0]
        else:
            raise ValueError(f"Heatmap must be 3D (or squeezable to 3D). Got shape: {heatmap.shape} for file {heatmap_file}")
    
    if image.shape != heatmap.shape:
        # Attempt to reconcile if one has an extra channel dim of 1
        squeezed_image_shape = image.shape
        squeezed_heatmap_shape = heatmap.shape
        if image.ndim == heatmap.ndim + 1 and image.shape[0] == 1:
             squeezed_image_shape = image.squeeze(0).shape
        elif heatmap.ndim == image.ndim + 1 and heatmap.shape[0] == 1:
             squeezed_heatmap_shape = heatmap.squeeze(0).shape

        if squeezed_image_shape != squeezed_heatmap_shape:
            raise ValueError(f"Image and heatmap must have the same spatial dimensions after potential squeeze. "
                             f"Original Image shape: {image.shape}, Original Heatmap shape: {heatmap.shape}. "
                             f"Squeezed Image shape: {squeezed_image_shape}, Squeezed Heatmap shape: {squeezed_heatmap_shape}."
                             f"Files: {image_file}, {heatmap_file}")
        if image.ndim > heatmap.ndim: image = image.squeeze(0)
        if heatmap.ndim > image.ndim: heatmap = heatmap.squeeze(0)


    # Select slice using the image. If heatmap is more relevant, this could be changed.
    # select_slice_to_plot2 needs a segmentation, but we don't have one here.
    # Using a simpler slice selection: middle slice, or adapt select_slice_to_plot to not require seg.
    # For now, let's use a simple middle slice selection if slice_selection_fn is the default one expecting seg.
    if slice_selection_fn.__name__ == 'select_slice_to_plot2' or slice_selection_fn.__name__ == 'select_slice_to_plot':
        # These functions expect a segmentation argument which we don't have for heatmap context.
        # Defaulting to a middle slice for the first spatial dimension (often axial).
        selected_slice_idx = image.shape[0] // 2
        print(f"Warning: slice_selection_fn '{slice_selection_fn.__name__}' expects segmentation. "
              f"Defaulting to middle slice ({selected_slice_idx}) for axis 0.")
    else:
        # This assumes slice_selection_fn can work with (image, heatmap) or just image
        try:
            selected_slice_idx = slice_selection_fn(image, heatmap) # if it can use heatmap
        except TypeError:
            selected_slice_idx = slice_selection_fn(image) # if it only uses image


    image_slice = image[selected_slice_idx]
    heatmap_slice = heatmap[selected_slice_idx]

    overlay_uint8 = generate_heatmap_overlay(image_slice, heatmap_slice, colormap_name, alpha)

    try:
        plt.imsave(output_file, overlay_uint8)
    except Exception as e:
        print(f"Error saving heatmap overlay to {output_file}: {e}")
        raise

def multiprocessing_plot_heatmap_overlay(list_of_image_files: List[str], list_of_heatmap_files: List[str],
                                         image_reader_writer: BaseReaderWriter, list_of_output_files: List[str],
                                         slice_selection_fn_list: List, colormap_name_list: List[str],
                                         alpha_list: List[float], num_processes: int = 8):
    """
    Helper function to run plot_heatmap_overlay in parallel.
    """
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        r = p.starmap_async(plot_heatmap_overlay, zip(
            list_of_image_files, list_of_heatmap_files,
            [image_reader_writer] * len(list_of_output_files),
            list_of_output_files,
            slice_selection_fn_list,
            colormap_name_list,
            alpha_list
        ))
        r.get()


def generate_heatmap_overlays_from_files(list_of_image_files: List[str], list_of_heatmap_files: List[str],
                                         output_folder: str, image_reader_writer_class: type,
                                         slice_selection_fn=select_slice_to_plot2,
                                         num_processes: int = 8, colormap_name: str = 'viridis',
                                         alpha: float = 0.7, output_filename_prefix: str = "heatmap_"):
    """
    Generates heatmap overlays for a list of image and heatmap files.

    Args:
        list_of_image_files (List[str]): List of paths to image files.
        list_of_heatmap_files (List[str]): List of paths to heatmap files.
        output_folder (str): Folder to save the generated overlay images.
        image_reader_writer_class (type): The class of the image reader/writer to use.
        slice_selection_fn (function): Function to select slice.
        num_processes (int): Number of parallel processes.
        colormap_name (str): Colormap for heatmap.
        alpha (float): Transparency for heatmap.
        output_filename_prefix (str): Prefix for output filenames.
    """
    if len(list_of_image_files) != len(list_of_heatmap_files):
        raise ValueError("list_of_image_files and list_of_heatmap_files must have the same length.")

    maybe_mkdir_p(output_folder)

    output_files = []
    for i in range(len(list_of_image_files)):
        base_img_name = os.path.basename(list_of_image_files[i]).split('.')[0]
        output_files.append(join(output_folder, f"{output_filename_prefix}{base_img_name}_{i}.png"))
    
    image_reader_writer_instance = image_reader_writer_class()

    multiprocessing_plot_heatmap_overlay(
        list_of_image_files,
        list_of_heatmap_files,
        image_reader_writer_instance,
        output_files,
        [slice_selection_fn] * len(output_files), # Use the same slice selection function for all
        [colormap_name] * len(output_files),      # Use the same colormap for all
        [alpha] * len(output_files),              # Use the same alpha for all
        num_processes=num_processes
    )


def select_slice_to_plot(image: np.ndarray, segmentation: np.ndarray) -> int:
    """
    image and segmentation are expected to be 3D

    selects the slice with the largest amount of fg (regardless of label)

    we give image so that we can easily replace this function if needed
    """
    fg_mask = segmentation != 0
    fg_per_slice = fg_mask.sum((1, 2))
    selected_slice = int(np.argmax(fg_per_slice))
    return selected_slice


def select_slice_to_plot2(image: np.ndarray, segmentation: np.ndarray) -> int:
    """
    image and segmentation are expected to be 3D (or 1, x, y)

    selects the slice with the largest amount of fg (how much percent of each class are in each slice? pick slice
    with highest avg percent)

    we give image so that we can easily replace this function if needed
    """
    classes = [i for i in np.sort(pd.unique(segmentation.ravel())) if i > 0]
    fg_per_slice = np.zeros((image.shape[0], len(classes)))
    for i, c in enumerate(classes):
        fg_mask = segmentation == c
        fg_per_slice[:, i] = fg_mask.sum((1, 2))
        fg_per_slice[:, i] /= fg_per_slice.sum()
    fg_per_slice = fg_per_slice.mean(1)
    return int(np.argmax(fg_per_slice))


def plot_overlay(image_file: str, segmentation_file: str, image_reader_writer: BaseReaderWriter, output_file: str,
                 overlay_intensity: float = 0.6):
    import matplotlib.pyplot as plt

    image, props = image_reader_writer.read_images((image_file, ))
    image = image[0]
    seg, props_seg = image_reader_writer.read_seg(segmentation_file)
    seg = seg[0]

    assert image.shape == seg.shape, "image and seg do not have the same shape: %s, %s" % (
        image_file, segmentation_file)

    assert image.ndim == 3, 'only 3D images/segs are supported'

    selected_slice = select_slice_to_plot2(image, seg)
    # print(image.shape, selected_slice)

    overlay = generate_overlay(image[selected_slice], seg[selected_slice], overlay_intensity=overlay_intensity)

    plt.imsave(output_file, overlay)


def plot_overlay_preprocessed(dataset: nnUNetBaseDataset, k: str, output_folder: str, overlay_intensity: float = 0.6, channel_idx=0):
    import matplotlib.pyplot as plt
    data, seg, _, properties = dataset.load_case(k)

    assert channel_idx < (data.shape[0]), 'This dataset only supports channel index up to %d' % (data.shape[0] - 1)

    image = data[channel_idx]
    seg = seg[0]
    selected_slice = select_slice_to_plot2(image, seg)

    seg = np.copy(seg[selected_slice])
    seg[seg < 0] = 0
    overlay = generate_overlay(image[selected_slice], seg, overlay_intensity=overlay_intensity)

    plt.imsave(join(output_folder, k + '.png'), overlay)


def multiprocessing_plot_overlay(list_of_image_files, list_of_seg_files, image_reader_writer,
                                 list_of_output_files, overlay_intensity,
                                 num_processes=8):
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        r = p.starmap_async(plot_overlay, zip(
            list_of_image_files, list_of_seg_files, [image_reader_writer] * len(list_of_output_files),
            list_of_output_files, [overlay_intensity] * len(list_of_output_files)
        ))
        r.get()


def multiprocessing_plot_overlay_preprocessed(dataset: nnUNetBaseDataset, output_folder, overlay_intensity,
                                              num_processes=8, channel_idx=0):
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        r = []
        for k in dataset.identifiers:
            r.append(
                p.starmap_async(plot_overlay_preprocessed,
                                ((
                                    dataset, k, output_folder, overlay_intensity, channel_idx
                                 ),))
            )
        _ = [i.get() for i in r]


def generate_overlays_from_raw(dataset_name_or_id: Union[int, str], output_folder: str,
                               num_processes: int = 8, channel_idx: int = 0, overlay_intensity: float = 0.6):
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    folder = join(nnUNet_raw, dataset_name)
    dataset_json = load_json(join(folder, 'dataset.json'))
    dataset = get_filenames_of_train_images_and_targets(folder, dataset_json)

    image_files = [v['images'][channel_idx] for v in dataset.values()]
    seg_files = [v['label'] for v in dataset.values()]

    assert all([isfile(i) for i in image_files])
    assert all([isfile(i) for i in seg_files])

    maybe_mkdir_p(output_folder)
    output_files = [join(output_folder, i + '.png') for i in dataset.keys()]

    image_reader_writer = determine_reader_writer_from_dataset_json(dataset_json, image_files[0])()
    multiprocessing_plot_overlay(image_files, seg_files, image_reader_writer, output_files, overlay_intensity, num_processes)


def generate_overlays_from_preprocessed(dataset_name_or_id: Union[int, str], output_folder: str,
                                        num_processes: int = 8, channel_idx: int = 0,
                                        configuration: str = None,
                                        plans_identifier: str = 'nnUNetPlans',
                                        overlay_intensity: float = 0.6):
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    folder = join(nnUNet_preprocessed, dataset_name)
    if not isdir(folder): raise RuntimeError("run preprocessing for that task first")

    plans = load_json(join(folder, plans_identifier + '.json'))
    if configuration is None:
        if '3d_fullres' in plans['configurations'].keys():
            configuration = '3d_fullres'
        else:
            configuration = '2d'
    cm = ConfigurationManager(plans['configurations'][configuration])
    preprocessed_folder = join(folder, cm.data_identifier)

    if not isdir(preprocessed_folder):
        raise RuntimeError(f"Preprocessed data folder for configuration {configuration} of plans identifier "
                           f"{plans_identifier} ({dataset_name}) does not exist. Run preprocessing for this "
                           f"configuration first!")

    dc = infer_dataset_class(preprocessed_folder)
    dataset = dc(preprocessed_folder)

    maybe_mkdir_p(output_folder)
    multiprocessing_plot_overlay_preprocessed(dataset, output_folder, overlay_intensity=overlay_intensity,
                                              num_processes=num_processes, channel_idx=channel_idx)


def entry_point_generate_overlay():
    import argparse
    parser = argparse.ArgumentParser("Plots png overlays of the slice with the most foreground (for segmentations) "
                                     "or a selected/middle slice (for heatmaps). Note that this disregards spacing information!")
    
    # Arguments for both segmentation and heatmap overlays
    parser.add_argument('-o', type=str, help="Output folder", required=True)
    parser.add_argument('-np', type=int, default=default_num_processes, required=False,
                        help=f"Number of processes used. Default: {default_num_processes}")
    parser.add_argument('-channel_idx', type=int, default=0, required=False,
                        help="Channel index used for input image (0 = _0000). Default: 0")

    # Subparsers for different modes (segmentation vs. heatmap)
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Mode of operation: "segmentation" or "heatmap"')

    # Parser for segmentation overlays
    parser_seg = subparsers.add_parser('segmentation', help='Generate segmentation overlays.')
    parser_seg.add_argument('-d', type=str, help="Dataset name or id for segmentation overlays.", required=True)
    parser_seg.add_argument('--use_raw', action='store_true', required=False, 
                            help="If set, use raw data for segmentation overlays. Else, use preprocessed.")
    parser_seg.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                            help='Plans identifier for preprocessed data. Only used if --use_raw is not set. Default: nnUNetPlans')
    parser_seg.add_argument('-c', type=str, required=False, default=None,
                            help='Configuration name for preprocessed data. Only used if --use_raw is not set. Default: None = 3d_fullres if available, else 2d')
    parser_seg.add_argument('-overlay_intensity', type=float, required=False, default=0.6,
                            help='Segmentation overlay intensity. Higher = brighter/less transparent. Default: 0.6')

    # Parser for heatmap overlays
    parser_heat = subparsers.add_parser('heatmap', help='Generate heatmap overlays.')
    parser_heat.add_argument('-images', type=str, nargs='+', required=True,
                             help='List of image files for heatmap overlays.')
    parser_heat.add_argument('-heatmaps', type=str, nargs='+', required=True,
                             help='List of heatmap files (e.g., uncertainty maps). Must correspond to -images.')
    # TODO: The BaseReaderWriter needs to be determined differently for direct file inputs.
    # This might require specifying reader type or trying to infer from dataset.json if a dataset context is still used.
    # For now, let's assume a generic reader might be needed or specified.
    # This part is tricky as image_reader_writer_class comes from dataset_json in other functions.
    # A simplified approach for now: Assume SimpleITKMultiThreadeIO or similar can be used if files are NIFTI.
    # This needs robust handling. For now, we'll pass a placeholder or require user to specify.
    parser_heat.add_argument('--reader_writer_class_name', type=str, default="SimpleITKIO", # Placeholder, needs better solution
                             help="Name of the nnU-Net ImageIO class to use (e.g., SimpleITKIO). Default: SimpleITKIO")
    parser_heat.add_argument('-colormap', type=str, default='viridis', required=False,
                             help="Matplotlib colormap name for the heatmap. Default: viridis")
    parser_heat.add_argument('-alpha', type=float, default=0.7, required=False,
                             help="Transparency of the heatmap overlay (0=transparent, 1=opaque). Default: 0.7")
    parser_heat.add_argument('--output_prefix', type=str, default="heatmap_",
                             help="Prefix for output heatmap overlay filenames. Default: 'heatmap_'")


    args = parser.parse_args()

    if args.mode == 'segmentation':
        if args.use_raw:
            generate_overlays_from_raw(args.d, args.o, args.np, args.channel_idx,
                                       overlay_intensity=args.overlay_intensity)
        else:
            generate_overlays_from_preprocessed(args.d, args.o, args.np, args.channel_idx, args.c, args.p,
                                                overlay_intensity=args.overlay_intensity)
    elif args.mode == 'heatmap':
        if len(args.images) != len(args.heatmaps):
            parser_heat.error("Number of -images and -heatmaps must be identical.")
        
        # Dynamically get the reader/writer class
        # This is a simplified way; a robust solution might involve a small factory or explicit mapping
        try:
            from nnunetv2.imageio.reader_writer_registry import reader_writer_perform_import
            reader_writer_class = reader_writer_perform_import(args.reader_writer_class_name)
            if reader_writer_class is None:
                raise ImportError
        except ImportError:
            print(f"Could not import specified reader_writer_class: {args.reader_writer_class_name}. Please ensure it's a valid nnU-Net ImageIO class name.")
            return

        generate_heatmap_overlays_from_files(
            list_of_image_files=args.images,
            list_of_heatmap_files=args.heatmaps,
            output_folder=args.o,
            image_reader_writer_class=reader_writer_class, # Pass the class itself
            num_processes=args.np,
            colormap_name=args.colormap,
            alpha=args.alpha,
            output_filename_prefix=args.output_prefix
            # slice_selection_fn could be added as an arg if needed
        )


if __name__ == '__main__':
    entry_point_generate_overlay()
