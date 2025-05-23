import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Union, List, Dict, Tuple, Optional

# Placeholder for Dice and IoU functions, will implement or import later if not directly available
# For now, simple implementations will be added if needed within the functions themselves.

def dice_score(pred: np.ndarray, true: np.ndarray, k: int = 1) -> float:
    """
    Dice coefficient for a single class k.
    pred: boolean or 0/1 array for class k predictions
    true: boolean or 0/1 array for class k ground truth
    """
    intersection = np.sum(pred[true==k]) * 2.0
    denominator = np.sum(pred) + np.sum(true)
    if denominator == 0:
        return 1.0 # Or 0.0, depending on convention for empty sets
    return intersection / denominator

def iou_score(pred: np.ndarray, true: np.ndarray, k: int = 1) -> float:
    """
    IoU for a single class k.
    pred: boolean or 0/1 array for class k predictions
    true: boolean or 0/1 array for class k ground truth
    """
    intersection = np.sum(pred[true==k])
    union = np.sum(pred) + np.sum(true) - intersection
    if union == 0:
        return 1.0 # Or 0.0, depending on convention for empty sets
    return intersection / union

class MetricNotCalculableError(Exception):
    """Custom exception for when a metric cannot be calculated (e.g. division by zero)."""
    pass

def calculate_ece(probabilities: np.ndarray, ground_truth_segmentations: np.ndarray, num_bins: int = 10) -> float:
    """
    Calculates the Expected Calibration Error (ECE).

    Args:
        probabilities (np.ndarray): Softmax probabilities. Shape: (N, C, H, W, D) or (N, C, H, W).
        ground_truth_segmentations (np.ndarray): Integer ground truth labels.
                                                 Shape: (N, 1, H, W, D) or (N, 1, H, W), or (N, H, W, D), (N, H, W).
        num_bins (int): Number of bins to use for confidence values.

    Returns:
        float: The Expected Calibration Error.
    """
    if not np.issubdtype(ground_truth_segmentations.dtype, np.integer):
        # This warning can be converted to an error if strict type checking is required.
        print(f"Warning: ground_truth_segmentations is not integer type (dtype: {ground_truth_segmentations.dtype}). Casting to int.")
        ground_truth_segmentations = ground_truth_segmentations.astype(np.int32)

    if probabilities.shape[0] != ground_truth_segmentations.shape[0]:
        raise ValueError("Batch size N must be the same for probabilities and ground_truth_segmentations.")

    # Adjust ground_truth_segmentations shape: (N, 1, H, W, D) or (N, 1, H, W) -> (N, H, W, D) or (N, H, W)
    if ground_truth_segmentations.ndim == probabilities.ndim and ground_truth_segmentations.shape[1] == 1:
        ground_truth_segmentations = np.squeeze(ground_truth_segmentations, axis=1)
    elif ground_truth_segmentations.ndim == probabilities.ndim -1: # GT is (N,H,W,D) and probs is (N,C,H,W,D)
        pass # Shape is already fine
    else:
        raise ValueError(
            f"Shape mismatch or unsupported shape for ground_truth_segmentations. Probs shape: {probabilities.shape}, GT shape: {ground_truth_segmentations.shape}"
        )


    predicted_labels = np.argmax(probabilities, axis=1) # Shape: (N, H, W, D) or (N, H, W)
    confidences = np.max(probabilities, axis=1)         # Shape: (N, H, W, D) or (N, H, W)

    # Flatten arrays
    flat_predicted_labels = predicted_labels.ravel()
    flat_confidences = confidences.ravel()
    flat_ground_truth = ground_truth_segmentations.ravel()

    if flat_predicted_labels.shape != flat_confidences.shape or flat_predicted_labels.shape != flat_ground_truth.shape:
        # This should not happen if input shapes are handled correctly
        raise ValueError(
            f"Flattened array shapes do not match. This indicates an issue with input processing. "
            f"Pred: {flat_predicted_labels.shape}, Conf: {flat_confidences.shape}, GT: {flat_ground_truth.shape}"
        )

    total_num_voxels = flat_confidences.shape[0]
    if total_num_voxels == 0:
        return 0.0 # Or raise MetricNotCalculableError

    bin_limits = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for j in range(num_bins):
        bin_lower_bound = bin_limits[j]
        bin_upper_bound = bin_limits[j+1]

        # Find indices of voxels where confidences fall into the current bin
        # For the last bin, include the upper bound (1.0)
        if j == num_bins - 1:
            indices_in_bin = (flat_confidences >= bin_lower_bound) & (flat_confidences <= bin_upper_bound)
        else:
            indices_in_bin = (flat_confidences >= bin_lower_bound) & (flat_confidences < bin_upper_bound)
        
        num_voxels_in_bin = np.sum(indices_in_bin)

        if num_voxels_in_bin == 0:
            continue

        bin_conf = flat_confidences[indices_in_bin]
        bin_pred_labels = flat_predicted_labels[indices_in_bin]
        bin_gt_labels = flat_ground_truth[indices_in_bin]

        accuracy_in_bin = np.mean((bin_pred_labels == bin_gt_labels).astype(float))
        average_confidence_in_bin = np.mean(bin_conf)
        
        ece += (num_voxels_in_bin / total_num_voxels) * np.abs(accuracy_in_bin - average_confidence_in_bin)

    return ece

def generate_uncertainty_error_curve_data(uncertainty_maps: np.ndarray, 
                                          predicted_segmentations: np.ndarray, 
                                          ground_truth_segmentations: np.ndarray, 
                                          num_threshold_steps: int = 20, 
                                          uncertainty_metric_name: str = 'Variance') -> dict:
    """
    Generates data for plotting Uncertainty-Error Curves.

    Args:
        uncertainty_maps (np.ndarray): Per-voxel uncertainty. 
                                     Shape: (N, [C_uncertainty,] H, W, D) or (N, [C_uncertainty,] H, W).
                                     If multi-channel (C_uncertainty > 1), it's averaged along channel 1.
        predicted_segmentations (np.ndarray): Final predicted labels. 
                                            Shape: (N, 1, H, W, D) or (N, 1, H, W) or (N, H, W, D) or (N, H, W).
        ground_truth_segmentations (np.ndarray): Integer ground truth labels.
                                               Shape: (N, 1, H, W, D) or (N, 1, H, W) or (N, H, W, D) or (N, H, W).
        num_threshold_steps (int): Number of steps for removing uncertain voxels.
        uncertainty_metric_name (str): Name of the uncertainty metric (for potential future use, e.g. title).

    Returns:
        dict: Contains 'fraction_voxels_removed', 'dice_scores', 'iou_scores'.
              Each list in the dictionary will store the batch-averaged scores for each fraction.
    """
    
    # Ensure GT and Pred are integer type
    if not np.issubdtype(ground_truth_segmentations.dtype, np.integer):
        ground_truth_segmentations = ground_truth_segmentations.astype(np.int32)
    if not np.issubdtype(predicted_segmentations.dtype, np.integer):
        predicted_segmentations = predicted_segmentations.astype(np.int32)

    # Reshape inputs to (N, H, W, D) or (N, H, W)
    if predicted_segmentations.ndim == uncertainty_maps.ndim and predicted_segmentations.shape[1] == 1: # N,1,H,W(,D)
        predicted_segmentations = np.squeeze(predicted_segmentations, axis=1)
    elif predicted_segmentations.ndim == uncertainty_maps.ndim -1 and uncertainty_maps.shape[1] > 1: # Pred is N,H,W(,D), Unc is N,C,H,W(,D)
        pass # Pred shape is fine
    elif predicted_segmentations.ndim == uncertainty_maps.ndim: # Pred is N,H,W(,D), Unc is N,H,W(,D) or N,C,H,W(,D) with C=1
         if uncertainty_maps.ndim > predicted_segmentations.ndim and uncertainty_maps.shape[1] == 1: # Unc is N,1,H,W(,D)
            uncertainty_maps = np.squeeze(uncertainty_maps, axis=1)
         elif uncertainty_maps.ndim == predicted_segmentations.ndim: # Both N,H,W(,D)
            pass
    elif predicted_segmentations.ndim == uncertainty_maps.ndim +1 and predicted_segmentations.shape[1] ==1 : # Pred N,1,H,W(D), Unc N,H,W(D)
        predicted_segmentations = np.squeeze(predicted_segmentations, axis=1)

    if ground_truth_segmentations.ndim == uncertainty_maps.ndim and ground_truth_segmentations.shape[1] == 1:
        ground_truth_segmentations = np.squeeze(ground_truth_segmentations, axis=1)
    elif ground_truth_segmentations.ndim == uncertainty_maps.ndim -1 and uncertainty_maps.shape[1] > 1:
        pass
    elif ground_truth_segmentations.ndim == uncertainty_maps.ndim:
        if uncertainty_maps.ndim > ground_truth_segmentations.ndim and uncertainty_maps.shape[1] == 1:
            uncertainty_maps = np.squeeze(uncertainty_maps, axis=1)
        elif uncertainty_maps.ndim == ground_truth_segmentations.ndim:
            pass
    elif ground_truth_segmentations.ndim == uncertainty_maps.ndim +1 and ground_truth_segmentations.shape[1] ==1 :
        ground_truth_segmentations = np.squeeze(ground_truth_segmentations, axis=1)


    # Handle multi-channel uncertainty maps by averaging
    # Assuming uncertainty_maps could be (N, C_uncertainty, H, W, D) or (N, C_uncertainty, H, W)
    # And predicted_segmentations is (N, H, W, D) or (N, H, W) after squeezing
    if uncertainty_maps.ndim == predicted_segmentations.ndim + 1 and uncertainty_maps.shape[1] > 1:
        uncertainty_maps = np.mean(uncertainty_maps, axis=1) # Now (N, H, W, D) or (N, H, W)
    elif uncertainty_maps.ndim == predicted_segmentations.ndim + 1 and uncertainty_maps.shape[1] == 1: # N,1,H,W,(D)
        uncertainty_maps = np.squeeze(uncertainty_maps, axis=1)


    if not (uncertainty_maps.shape == predicted_segmentations.shape == ground_truth_segmentations.shape):
        raise ValueError(f"Shape mismatch after initial processing: "
                         f"Uncertainty: {uncertainty_maps.shape}, "
                         f"Predicted: {predicted_segmentations.shape}, "
                         f"Ground Truth: {ground_truth_segmentations.shape}")

    num_samples = uncertainty_maps.shape[0]
    
    # Store scores for each fraction step, to be averaged later
    all_dice_scores_at_steps = [[] for _ in range(num_threshold_steps + 1)]
    all_iou_scores_at_steps = [[] for _ in range(num_threshold_steps + 1)]
    fractions_removed_at_steps = np.linspace(0, 1, num_threshold_steps + 1)

    for n in range(num_samples):
        flat_uncertainty = uncertainty_maps[n].ravel()
        flat_predicted = predicted_segmentations[n].ravel()
        flat_gt = ground_truth_segmentations[n].ravel()

        # Sort by uncertainty (descending), so most uncertain are first
        sorted_indices_by_uncertainty = np.argsort(flat_uncertainty)[::-1] 
        num_total_voxels = len(flat_uncertainty)

        if num_total_voxels == 0:
            for step_idx in range(num_threshold_steps + 1):
                all_dice_scores_at_steps[step_idx].append(0.0) # Or some indicator for invalid
                all_iou_scores_at_steps[step_idx].append(0.0)
            continue


        for step_idx, fraction_to_remove in enumerate(fractions_removed_at_steps):
            num_voxels_to_remove = int(fraction_to_remove * num_total_voxels)
            
            # If removing all voxels, handle explicitly
            if num_voxels_to_remove >= num_total_voxels:
                indices_to_keep = np.array([], dtype=np.int64)
            else:
                indices_to_keep = sorted_indices_by_uncertainty[num_voxels_to_remove:]

            current_dice_scores_for_sample = []
            current_iou_scores_for_sample = []

            if len(indices_to_keep) == 0:
                # All voxels removed or no voxels to begin with
                all_dice_scores_at_steps[step_idx].append(0.0) # Assuming 0 if no voxels
                all_iou_scores_at_steps[step_idx].append(0.0)
                continue

            confident_predictions = flat_predicted[indices_to_keep]
            confident_gt = flat_gt[indices_to_keep]
            
            # Get unique labels present in the confident ground truth, excluding background (0)
            unique_gt_labels = np.unique(confident_gt)
            foreground_labels = unique_gt_labels[unique_gt_labels > 0]

            if len(foreground_labels) == 0: # No foreground instances in GT subset
                # This case means either only background is left, or GT was empty of foreground.
                # Dice/IoU is typically 1 if pred also shows no foreground, or 0 if pred shows some.
                # For simplicity, if no FG in GT, and no FG in pred, consider it perfect for that part.
                # If no FG in GT, but some in pred, it's 0.
                # A common approach: if GT is all background, Dice is 1 if pred is also all background, else 0.
                # Here, we average over foreground classes. If none exist, the average is over an empty set.
                # Let's define this as 1.0 if prediction is also all background, else 0.0
                # This is tricky. Let's append 1.0 if confident_predictions only contains background, else 0.0
                # Or, if there are no foreground labels, this sample contributes nothing to the average Dice/IoU for this step.
                # We will average valid Dice/IoU scores. If a sample has no FG labels, its Dice/IoU for this step is undefined for FG.
                # For now, if no foreground_labels, we'll append a value that indicates this, or handle in averaging.
                # Let's append nan, and use np.nanmean later.
                current_dice_scores_for_sample.append(np.nan) 
                current_iou_scores_for_sample.append(np.nan)
            else:
                for label_k in foreground_labels:
                    pred_k = (confident_predictions == label_k)
                    true_k = (confident_gt == label_k)
                    current_dice_scores_for_sample.append(dice_score(pred_k, true_k, k=1)) # k=1 because pred_k/true_k are already boolean for the class
                    current_iou_scores_for_sample.append(iou_score(pred_k, true_k, k=1))
            
            # Average Dice/IoU for the current sample at this removal step
            # Use nanmean to ignore NaNs if no foreground labels were present
            avg_dice_for_sample_step = np.nanmean(current_dice_scores_for_sample) if current_dice_scores_for_sample else 0.0
            avg_iou_for_sample_step = np.nanmean(current_iou_scores_for_sample) if current_iou_scores_for_sample else 0.0
            
            all_dice_scores_at_steps[step_idx].append(avg_dice_for_sample_step if not np.isnan(avg_dice_for_sample_step) else 0.0)
            all_iou_scores_at_steps[step_idx].append(avg_iou_for_sample_step if not np.isnan(avg_iou_for_sample_step) else 0.0)

    # Average scores across all samples for each step
    final_avg_dice_scores = [np.mean(scores_at_step) if scores_at_step else 0.0 for scores_at_step in all_dice_scores_at_steps]
    final_avg_iou_scores = [np.mean(scores_at_step) if scores_at_step else 0.0 for scores_at_step in all_iou_scores_at_steps]
    
    # Ensure fractions_removed_at_steps is a list for JSON serialization if needed later
    curve_data = {
        'fraction_voxels_removed': fractions_removed_at_steps.tolist(),
        'dice_scores': final_avg_dice_scores,
        'iou_scores': final_avg_iou_scores,
        'uncertainty_metric_name': uncertainty_metric_name
    }

    return curve_data

def plot_uncertainty_error_curve(curve_data: dict, 
                                 output_filepath: str, 
                                 metric_to_plot: str = 'Dice', 
                                 plot_iou: bool = True):
    """
    Plots the Uncertainty-Error Curve.

    Args:
        curve_data (dict): Dictionary containing 'fraction_voxels_removed', 
                           'dice_scores', 'iou_scores', and 'uncertainty_metric_name'.
        output_filepath (str): Path to save the generated plot.
        metric_to_plot (str): Which primary metric to plot ('Dice' or 'IoU'). This will be the main line.
        plot_iou (bool): If True and metric_to_plot is 'Dice', also plots IoU as a secondary line. 
                         If metric_to_plot is 'IoU', Dice will be plotted as secondary if available.
    """
    fractions = curve_data['fraction_voxels_removed']
    
    plt.figure(figsize=(10, 6))

    primary_metric_scores = None
    primary_metric_label = ""
    secondary_metric_scores = None
    secondary_metric_label = ""

    if metric_to_plot.lower() == 'dice':
        primary_metric_scores = curve_data.get('dice_scores')
        primary_metric_label = 'Dice Score'
        if plot_iou and 'iou_scores' in curve_data:
            secondary_metric_scores = curve_data.get('iou_scores')
            secondary_metric_label = 'IoU Score'
    elif metric_to_plot.lower() == 'iou':
        primary_metric_scores = curve_data.get('iou_scores')
        primary_metric_label = 'IoU Score'
        if plot_iou and 'dice_scores' in curve_data: # Plot Dice if primary is IoU
            secondary_metric_scores = curve_data.get('dice_scores')
            secondary_metric_label = 'Dice Score'
    else:
        raise ValueError(f"metric_to_plot must be 'Dice' or 'IoU', got {metric_to_plot}")

    if primary_metric_scores is None:
        raise ValueError(f"Scores for primary metric '{metric_to_plot}' not found in curve_data.")

    plt.plot(fractions, primary_metric_scores, marker='o', linestyle='-', label=primary_metric_label)

    if secondary_metric_scores is not None:
        plt.plot(fractions, secondary_metric_scores, marker='x', linestyle='--', label=secondary_metric_label)
    
    uncertainty_name = curve_data.get('uncertainty_metric_name', 'Uncertainty')
    plt.title(f'{uncertainty_name}-Error Curve ({primary_metric_label})')
    plt.xlabel("Fraction of Most Uncertain Voxels Removed")
    plt.ylabel(f"Segmentation Score")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05]) # Allow a bit of margin above 1.0 for visual clarity
    plt.grid(True, linestyle='--')
    plt.legend(loc='lower left') # Typically scores increase as uncertain voxels are removed
    plt.tight_layout()

    try:
        plt.savefig(output_filepath)
        print(f"Uncertainty-Error curve plot saved to {output_filepath}")
    except Exception as e:
        print(f"Error saving plot to {output_filepath}: {e}")
    plt.close()
