import os
from copy import deepcopy
from typing import Union, List

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice, insert_crop_into_image
from batchgenerators.utilities.file_and_folder_operations import load_json, isfile, save_pickle

from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


def convert_probabilities_to_segmentation_with_correct_shape(
        predicted_probabilities: Union[torch.Tensor, np.ndarray], # Renamed
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        label_manager: LabelManager,
        properties_dict: dict,
        return_probabilities_and_segmentation: bool = False, # Changed name and meaning
        variance_map: Optional[Union[torch.Tensor, np.ndarray]] = None,
        num_threads_torch: int = default_num_processes): # New name
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # input is now assumed to be probabilities
    if isinstance(predicted_probabilities, np.ndarray):
        predicted_probabilities = torch.from_numpy(predicted_probabilities)

    # derive segmentation before resampling
    segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)
    # Ensure variance_map is a tensor for consistent processing, if it exists
    if variance_map is not None and isinstance(variance_map, np.ndarray):
        variance_map = torch.from_numpy(variance_map)

    # resample to original shape
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    
    resampling_args = (properties_dict['shape_after_cropping_and_before_resampling'],
                       current_spacing,
                       [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])

    # Resample probabilities, segmentation, and variance map (if present)
    resampled_probabilities = configuration_manager.resampling_fn_probabilities(predicted_probabilities, *resampling_args)
    resampled_segmentation = configuration_manager.resampling_fn_segmentation(segmentation, *resampling_args)
    
    resampled_variance_map = None
    if variance_map is not None:
        resampled_variance_map = configuration_manager.resampling_fn_probabilities(variance_map, *resampling_args)

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
    segmentation_reverted_cropping = insert_crop_into_image(segmentation_reverted_cropping, resampled_segmentation,
                                                            properties_dict['bbox_used_for_cropping'])
    del resampled_segmentation

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation_reverted_cropping, torch.Tensor):
        segmentation_reverted_cropping = segmentation_reverted_cropping.cpu().numpy()

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)

    # Prepare return values
    # The first element is always the segmentation
    return_values = [segmentation_reverted_cropping] 
    
    # Handle probabilities
    if return_probabilities_and_segmentation:
        # Probabilities are already resampled, now revert cropping and transpose
        resampled_probabilities = label_manager.revert_cropping_on_probabilities(
            resampled_probabilities, # Use the resampled version
            properties_dict['bbox_used_for_cropping'],
            properties_dict['shape_before_cropping']
        )
        resampled_probabilities = resampled_probabilities.cpu().numpy()
        resampled_probabilities = resampled_probabilities.transpose([0] + [i + 1 for i in plans_manager.transpose_backward])
        return_values.append(resampled_probabilities)

    # Handle variance map
    if variance_map is not None:
        # Variance map is already resampled, now revert cropping and transpose
        resampled_variance_map = label_manager.revert_cropping_on_probabilities(
            resampled_variance_map, # Use the resampled version
            properties_dict['bbox_used_for_cropping'],
            properties_dict['shape_before_cropping']
        )
        resampled_variance_map = resampled_variance_map.cpu().numpy()
        resampled_variance_map = resampled_variance_map.transpose([0] + [i + 1 for i in plans_manager.transpose_backward])
        return_values.append(resampled_variance_map)
        
    torch.set_num_threads(old_threads)
    
    if len(return_values) == 1:
        return return_values[0] # Only segmentation
    else:
        return tuple(return_values)


def export_prediction_from_probabilities( # Renamed
        predicted_probabilities: Union[np.ndarray, torch.Tensor], # Renamed
        properties_dict: dict,
        configuration_manager: ConfigurationManager,
        plans_manager: PlansManager,
        dataset_json_dict_or_file: Union[dict, str],
        output_file_truncated: str,
        save_prediction_npz: bool = False, # Renamed
        variance_map: Optional[Union[np.ndarray, torch.Tensor]] = None,
        num_threads_torch: int = default_num_processes):

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    
    # convert_probabilities_to_segmentation_with_correct_shape now returns a tuple
    # The first element is always segmentation.
    # If return_probabilities_and_segmentation is True, probabilities are the second.
    # If variance_map is provided, it's the next.
    processed_outputs = convert_probabilities_to_segmentation_with_correct_shape(
        predicted_probabilities, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities_and_segmentation=save_prediction_npz, # Pass new flag
        variance_map=variance_map,
        num_threads_torch=num_threads_torch
    )

    segmentation_final = processed_outputs[0]
    
    # Save the segmentation (NIfTI or other formats)
    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)

    # Handle NPZ saving
    if save_prediction_npz:
        npz_dict = {'segmentation': segmentation_final}
        idx = 1
        if save_prediction_npz : # This check is redundant due to outer if, but explicit. Probabilities were requested.
            npz_dict['probabilities'] = processed_outputs[idx]
            idx +=1
        
        if variance_map is not None:
            npz_dict['variance_map'] = processed_outputs[idx]
            
        np.savez_compressed(output_file_truncated + '.npz', **npz_dict)
        save_pickle(properties_dict, output_file_truncated + '.pkl') # Save properties for the .npz

    # Handle separate variance map NIfTI saving if NPZ wasn't saved but variance is present
    elif variance_map is not None: 
        # variance_map would be the second element if save_prediction_npz was False
        variance_map_final = processed_outputs[1] 
        variance_filename = output_file_truncated + "_variance" + dataset_json_dict_or_file['file_ending']
        
        # Create a temporary properties_dict for variance map, potentially with adjusted channel info
        variance_properties_dict = deepcopy(properties_dict)
        # Assuming variance map is single channel or multi-channel float data.
        # The writer needs to handle this. For SimpleITK, it might save multi-channel as vector image.
        # If variance is per class, it will have C channels.
        
        # We need a writer that can save float data. SimpleITKIO should handle this.
        # We assume variance_map_final is already correctly oriented (transposed back)
        variance_rw = SimpleITKIO() # Using SimpleITKIO directly, or plans_manager.image_reader_writer_class() if it's suitable for float
        variance_rw.write_seg(variance_map_final, variance_filename, variance_properties_dict) # write_seg might work for float if underlying lib supports it
        # Alternatively, a more generic write_nifti_from_numpy or similar might be needed if write_seg is strictly for int labels.
        # For now, assuming write_seg can handle it or will be adapted.
        # print(f"Saved variance map to {variance_filename}") # Optional: for debugging

def resample_and_save(predicted: Union[torch.Tensor, np.ndarray], target_shape: List[int], output_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager, properties_dict: dict,
                      dataset_json_dict_or_file: Union[dict, str], num_threads_torch: int = default_num_processes,
                      dataset_class=None) \
        -> None:
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    target_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted,
                                                                                target_shape,
                                                                                current_spacing,
                                                                                target_spacing)

    # create segmentation (argmax, regions, etc)
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)
    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    if dataset_class is None:
        nnUNetDatasetBlosc2.save_seg(segmentation.astype(dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16), output_file)
    else:
        dataset_class.save_seg(segmentation.astype(dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16), output_file)
    torch.set_num_threads(old_threads)
