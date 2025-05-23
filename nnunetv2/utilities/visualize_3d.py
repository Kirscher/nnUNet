import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import argparse
import os

def plot_orthogonal_views(image_path: str, 
                          segmentation_path: str, 
                          uncertainty_path: Optional[str] = None, 
                          output_png_path: Optional[str] = None, 
                          slice_indices: Optional[Tuple[int, int, int]] = None, 
                          main_title: Optional[str] = None, 
                          uncertainty_colormap: str = 'viridis', 
                          uncertainty_alpha: float = 0.5, 
                          segmentation_contour_color: str = 'r', 
                          segmentation_linewidth: float = 0.5):
    """
    Plots orthogonal views (axial, sagittal, coronal) of an image, its segmentation, 
    and optionally an uncertainty map.

    Args:
        image_path (str): Path to the NIfTI image file.
        segmentation_path (str): Path to the NIfTI segmentation file.
        uncertainty_path (Optional[str], optional): Path to the NIfTI uncertainty map file. Defaults to None.
        output_png_path (Optional[str], optional): Path to save the output PNG. If None, shows plot. Defaults to None.
        slice_indices (Optional[Tuple[int, int, int]], optional): Tuple of (sagittal, coronal, axial) slice indices. 
                                                                  If None, central slices are used. Defaults to None.
        main_title (Optional[str], optional): Main title for the plot. Defaults to None.
        uncertainty_colormap (str, optional): Colormap for the uncertainty heatmap. Defaults to 'viridis'.
        uncertainty_alpha (float, optional): Alpha value for the uncertainty heatmap overlay. Defaults to 0.5.
        segmentation_contour_color (str, optional): Color for the segmentation contours. Defaults to 'r'.
        segmentation_linewidth (float, optional): Linewidth for segmentation contours. Defaults to 0.5.
    """
    try:
        img_nib = nib.load(image_path)
        image_data = img_nib.get_fdata()
    except Exception as e:
        print(f"Error loading image file {image_path}: {e}")
        return

    try:
        seg_nib = nib.load(segmentation_path)
        segmentation_data = seg_nib.get_fdata()
    except Exception as e:
        print(f"Error loading segmentation file {segmentation_path}: {e}")
        return

    uncertainty_data = None
    if uncertainty_path:
        try:
            unc_nib = nib.load(uncertainty_path)
            uncertainty_data = unc_nib.get_fdata()
            if image_data.shape != uncertainty_data.shape:
                print(f"Warning: Image shape {image_data.shape} and uncertainty shape {uncertainty_data.shape} mismatch. Skipping uncertainty overlay.")
                uncertainty_data = None
        except Exception as e:
            print(f"Error loading uncertainty file {uncertainty_path}: {e}. Proceeding without uncertainty.")
            uncertainty_data = None
    
    if image_data.shape != segmentation_data.shape:
        # Try to reconcile if one has an extra channel dim of 1 at the end (e.g. (H,W,D,1) vs (H,W,D))
        if image_data.ndim == segmentation_data.ndim + 1 and image_data.shape[-1] == 1:
            image_data = image_data[...,0]
        elif segmentation_data.ndim == image_data.ndim + 1 and segmentation_data.shape[-1] == 1:
            segmentation_data = segmentation_data[...,0]
        
        if image_data.shape != segmentation_data.shape: # Check again
            print(f"Error: Image shape {img_nib.shape} and segmentation shape {seg_nib.shape} do not match after attempting reconciliation.")
            return


    if slice_indices is None:
        slice_indices = tuple(s // 2 for s in image_data.shape)
    
    if len(slice_indices) != 3 :
        print(f"Error: slice_indices must be a tuple of 3 integers for (sagittal, coronal, axial) views. Got: {slice_indices}")
        # Attempt to use central if incorrect format given
        slice_indices = tuple(s // 2 for s in image_data.shape)
        print(f"Defaulting to central slices: {slice_indices}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Increased figure size for better layout

    # --- Axial View ---
    # Sagittal index for X, Coronal index for Y, Axial index for Z
    # image_data shape is typically (Sag, Cor, Ax) if loaded by nibabel from standard NIFTI
    ax_idx, cor_idx, sag_idx = slice_indices[2], slice_indices[1], slice_indices[0] 

    # Axial: fixed Z (ax_idx), vary X (sag_idx-dim) and Y (cor_idx-dim)
    img_slice_ax = np.rot90(image_data[:, :, ax_idx])
    seg_slice_ax = np.rot90(segmentation_data[:, :, ax_idx])
    
    axes[0].imshow(img_slice_ax, cmap='gray')
    unique_labels_ax = np.unique(seg_slice_ax)
    if len(unique_labels_ax) > 1: # Check if there's anything other than background
        axes[0].contour(seg_slice_ax, levels=unique_labels_ax[1:], colors=segmentation_contour_color, linewidths=segmentation_linewidth)
    
    if uncertainty_data is not None:
        unc_slice_ax = np.rot90(uncertainty_data[:, :, ax_idx])
        axes[0].imshow(unc_slice_ax, cmap=uncertainty_colormap, alpha=uncertainty_alpha, 
                       vmin=0, vmax=np.max(uncertainty_data) if uncertainty_data.size > 0 and np.max(uncertainty_data) > 0 else 1)
    axes[0].set_title(f"Axial View (Slice {ax_idx})")
    axes[0].axis('off')

    # --- Sagittal View ---
    # Sagittal: fixed X (sag_idx), vary Y (cor_idx-dim) and Z (ax_idx-dim)
    img_slice_sag = np.rot90(image_data[sag_idx, :, :])
    seg_slice_sag = np.rot90(segmentation_data[sag_idx, :, :])

    axes[1].imshow(img_slice_sag, cmap='gray')
    unique_labels_sag = np.unique(seg_slice_sag)
    if len(unique_labels_sag) > 1:
        axes[1].contour(seg_slice_sag, levels=unique_labels_sag[1:], colors=segmentation_contour_color, linewidths=segmentation_linewidth)

    if uncertainty_data is not None:
        unc_slice_sag = np.rot90(uncertainty_data[sag_idx, :, :])
        axes[1].imshow(unc_slice_sag, cmap=uncertainty_colormap, alpha=uncertainty_alpha,
                       vmin=0, vmax=np.max(uncertainty_data) if uncertainty_data.size > 0 and np.max(uncertainty_data) > 0 else 1)
    axes[1].set_title(f"Sagittal View (Slice {sag_idx})")
    axes[1].axis('off')

    # --- Coronal View ---
    # Coronal: fixed Y (cor_idx), vary X (sag_idx-dim) and Z (ax_idx-dim)
    img_slice_cor = np.rot90(image_data[:, cor_idx, :])
    seg_slice_cor = np.rot90(segmentation_data[:, cor_idx, :])

    axes[2].imshow(img_slice_cor, cmap='gray')
    unique_labels_cor = np.unique(seg_slice_cor)
    if len(unique_labels_cor) > 1:
        axes[2].contour(seg_slice_cor, levels=unique_labels_cor[1:], colors=segmentation_contour_color, linewidths=segmentation_linewidth)
    
    if uncertainty_data is not None:
        unc_slice_cor = np.rot90(uncertainty_data[:, cor_idx, :])
        axes[2].imshow(unc_slice_cor, cmap=uncertainty_colormap, alpha=uncertainty_alpha,
                       vmin=0, vmax=np.max(uncertainty_data) if uncertainty_data.size > 0 and np.max(uncertainty_data) > 0 else 1)
    axes[2].set_title(f"Coronal View (Slice {cor_idx})")
    axes[2].axis('off')

    if main_title:
        fig.suptitle(main_title, fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96] if main_title else None) # Adjust layout to make space for suptitle

    if output_png_path:
        try:
            plt.savefig(output_png_path)
            print(f"Orthogonal views saved to {output_png_path}")
        except Exception as e:
            print(f"Error saving plot to {output_png_path}: {e}")
    else:
        plt.show()
    
    plt.close(fig) # Close the figure to free memory

def main():
    parser = argparse.ArgumentParser(description="Plot orthogonal views of a 3D medical image and its segmentation/uncertainty.")
    parser.add_argument("-i", "--image_path", type=str, required=True, help="Path to the NIfTI image file.")
    parser.add_argument("-s", "--segmentation_path", type=str, required=True, help="Path to the NIfTI segmentation file.")
    parser.add_argument("-u", "--uncertainty_path", type=str, default=None, help="Optional: Path to the NIfTI uncertainty map file.")
    parser.add_argument("-o", "--output_png_path", type=str, default=None, help="Optional: Path to save the output PNG. If not provided, the plot will be shown.")
    parser.add_argument("--slice_indices", type=int, nargs=3, default=None, metavar=('SAG', 'COR', 'AX'),
                        help="Optional: Tuple of (sagittal, coronal, axial) slice indices. E.g., 100 120 90. If None, central slices are used.")
    parser.add_argument("--title", type=str, default=None, help="Optional: Main title for the plot.")
    parser.add_argument("--uncertainty_cmap", type=str, default='viridis', help="Colormap for the uncertainty heatmap. Default: 'viridis'.")
    parser.add_argument("--uncertainty_alpha", type=float, default=0.5, help="Alpha value for the uncertainty heatmap overlay. Default: 0.5.")
    parser.add_argument("--seg_color", type=str, default='r', help="Color for the segmentation contours. Default: 'r'.")
    parser.add_argument("--seg_lw", type=float, default=0.5, help="Linewidth for segmentation contours. Default: 0.5.")

    args = parser.parse_args()

    # Convert slice_indices to tuple if provided
    slice_indices_tuple = tuple(args.slice_indices) if args.slice_indices else None

    plot_orthogonal_views(
        image_path=args.image_path,
        segmentation_path=args.segmentation_path,
        uncertainty_path=args.uncertainty_path,
        output_png_path=args.output_png_path,
        slice_indices=slice_indices_tuple,
        main_title=args.title,
        uncertainty_colormap=args.uncertainty_cmap,
        uncertainty_alpha=args.uncertainty_alpha,
        segmentation_contour_color=args.seg_color,
        segmentation_linewidth=args.seg_lw
    )

if __name__ == "__main__":
    main()
