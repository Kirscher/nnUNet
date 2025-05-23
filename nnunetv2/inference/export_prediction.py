import os
from copy import deepcopy
from typing import Union, List

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice, insert_crop_into_image
from batchgenerators.utilities.file_and_folder_operations import load_json, isfile, save_pickle

from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                variance_map: Optional[Union[torch.Tensor, np.ndarray]] = None,
                                                                num_threads_torch: int = default_num_processes):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

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

    predicted_logits = configuration_manager.resampling_fn_probabilities(predicted_logits, *resampling_args)
    
    if variance_map is not None:
        # Resample variance map similarly. Assuming it's float data, trilinear interp is fine.
        # The resampling function might need to handle multi-channel data appropriately if variance_map is [C, H, W, D]
        variance_map_resampled = configuration_manager.resampling_fn_probabilities(variance_map, *resampling_args)


    # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch
    if not return_probabilities:
        # this has a faster computation path becasue we can skip the softmax in regular (not region based) trainig
        segmentation = label_manager.convert_logits_to_segmentation(predicted_logits)
    else:
        predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
        segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)
    del predicted_logits

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
    segmentation_reverted_cropping = insert_crop_into_image(segmentation_reverted_cropping, segmentation, properties_dict['bbox_used_for_cropping'])
    del segmentation

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation_reverted_cropping, torch.Tensor):
        segmentation_reverted_cropping = segmentation_reverted_cropping.cpu().numpy()

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)

    # Prepare return values
    return_values = [segmentation_reverted_cropping]

    if return_probabilities:
        # revert cropping for probabilities
        predicted_probabilities = label_manager.revert_cropping_on_probabilities(predicted_probabilities,
                                                                                 properties_dict[
                                                                                     'bbox_used_for_cropping'],
                                                                                 properties_dict[
                                                                                     'shape_before_cropping'])
        predicted_probabilities = predicted_probabilities.cpu().numpy()
        # revert transpose for probabilities
        predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in
                                                                           plans_manager.transpose_backward])
        return_values.append(predicted_probabilities)

    if variance_map is not None:
        # Revert cropping for variance map
        # Assuming variance_map_resampled is a tensor [C, H, W, D] or [C, H, W]
        # Need a similar revert_cropping_on_probabilities or adapt it if it works for generic multi-channel data
        variance_map_reverted_cropping = label_manager.revert_cropping_on_probabilities(
            variance_map_resampled,
            properties_dict['bbox_used_for_cropping'],
            properties_dict['shape_before_cropping']
        )
        variance_map_reverted_cropping = variance_map_reverted_cropping.cpu().numpy()
        # Revert transpose for variance map
        variance_map_reverted_cropping = variance_map_reverted_cropping.transpose([0] + [i + 1 for i in
                                                                                    plans_manager.transpose_backward])
        return_values.append(variance_map_reverted_cropping)
        
    torch.set_num_threads(old_threads)
    
    if len(return_values) == 1:
        return return_values[0]
    else:
        return tuple(return_values)


def export_prediction_from_logits(predicted_logits: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                  save_probabilities: bool = False,
                                  variance_map: Union[np.ndarray, torch.Tensor] = None,
                                  num_threads_torch: int = default_num_processes):
    # if isinstance(predicted_logits, str):
    #     tmp = deepcopy(predicted_logits)
    #     if predicted_logits.endswith('.npy'):
    #         predicted_logits = np.load(predicted_logits)
    #     elif predicted_logits.endswith('.npz'):
    #         predicted_logits = np.load(predicted_logits)['softmax']
    #     os.remove(tmp)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_logits, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities, num_threads_torch=num_threads_torch,
        variance_map=variance_map # Pass variance_map here
    )
    # del predicted_logits # We might need it if save_probabilities is true, for the npz file

    # save
    if save_probabilities:
        if variance_map is not None:
            segmentation_final, probabilities_final, variance_map_final = ret
            np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final, variance=variance_map_final)
            save_pickle(properties_dict, output_file_truncated + '.pkl') # Save properties once
            del probabilities_final, variance_map_final
        else:
            segmentation_final, probabilities_final = ret
            np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
            save_pickle(properties_dict, output_file_truncated + '.pkl')
            del probabilities_final
    else: # not save_probabilities
        if variance_map is not None: # variance can be present even if not saving probabilities
            segmentation_final, variance_map_final = ret # Expecting two return values now
            # Save variance map separately if it's provided and not saving probabilities (as it's not in the npz)
            variance_filename = output_file_truncated + "_variance" + dataset_json_dict_or_file['file_ending']
            rw_variance = plans_manager.image_reader_writer_class() # Or appropriate writer for multi-channel float data
            # The variance map needs to be resampled and transposed like the segmentation/probabilities.
            # This logic is handled in convert_predicted_logits_to_segmentation_with_correct_shape
            # Ensure variance_map_final is what you expect (numpy array, correct orientation)
            rw_variance.write_seg(variance_map_final, variance_filename, properties_dict) # May need a different writer if not int seg
            del variance_map_final
        else:
            segmentation_final = ret
    
    del ret # delete ret at the end

    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)
    
    # Clean up predicted_logits if it's a large array and no longer needed.
    del predicted_logits
    if variance_map is not None and not save_probabilities: # If variance map was handled and saved separately
        pass # variance_map_final was already deleted
    elif variance_map is not None: # If passed but not handled by a specific path yet (e.g. save_probabilities=False but variance still needs saving)
        # This case should ideally be covered by the logic above.
        # If variance map is returned by convert_predicted_logits_to_segmentation_with_correct_shape
        # and not saved with probabilities, it needs its own saving mechanism.
        # This is somewhat redundant with the save_probabilities=False, variance_map is not None block
        pass




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
