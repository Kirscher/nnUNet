import argparse
import os
import torch
import nnunetv2 # Required for path initialization
from nnunetv2.run.run_training import run_training # Import the Python function

def run_training_ensemble_entry():
    parser = argparse.ArgumentParser(description="Run nnU-Net ensemble training.")
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('fold', type=str,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4 or \'all\'.')
    
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    
    parser.add_argument('--num_ensemble_members', type=int, required=True,
                        help="Number of ensemble members to train.")
    parser.add_argument('--start_seed', type=int, default=42,
                        help="Starting random seed for the first ensemble member. Subsequent members will use seed + member_idx. Default: 42")
    
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Specify the number of GPUs to use for training. Default: 1')
    parser.add_argument('--deterministic_augmentations', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to use deterministic data augmentations. Recommended for reproducibility.')
    parser.add_argument('--cudnn_deterministic', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to make CuDNN use deterministic algorithms. May impact performance. Recommended for reproducibility.')
    parser.add_argument('--disable_checkpointing', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and '
                             'you dont want to flood your hard drive with checkpoints.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the training should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!")
    
    parser.add_argument('--npz', action='store_true', required=False,
                        help='[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted '
                             'segmentations). Needed for finding the best ensemble.')
    parser.add_argument('--c', action='store_true', required=False,
                        help='[OPTIONAL] Continue training for ensemble members. Checks for existing checkpoints in member-specific folders.')
    parser.add_argument('--val_best', action='store_true', required=False,
                        help='[OPTIONAL] If set, the validation will be performed with the checkpoint_best instead '
                             'of checkpoint_final. NOT COMPATIBLE with --disable_checkpointing! '
                             'WARNING: This will use the same \'validation\' folder as the regular validation '
                             'with no way of distinguishing the two!')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only '
                             'be used when actually training. Beta. Use with caution.')

    args = parser.parse_args()

    # Determine torch device
    if args.device == 'cpu':
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        torch_device_object = torch.device('cpu')
    elif args.device == 'cuda':
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch_device_object = torch.device('cuda')
    elif args.device == 'mps':
        torch_device_object = torch.device('mps')
    else:
        raise ValueError(f"Device {args.device} not supported. Choose from 'cuda', 'cpu', 'mps'.")

    for member_idx in range(args.num_ensemble_members):
        current_member_seed = args.start_seed + member_idx
        member_suffix = f"ensemble_member_{member_idx}"

        print(f"\n====================================================================")
        print(f"Starting training for ensemble member {member_idx + 1}/{args.num_ensemble_members}")
        print(f"Seed: {current_member_seed}")
        print(f"Output Suffix: {member_suffix}")
        print(f"====================================================================\n")

        run_training(
            dataset_name_or_id=args.dataset_name_or_id,
            configuration=args.configuration,
            fold=args.fold,
            trainer_class_name=args.tr,
            plans_identifier=args.p,
            pretrained_weights=args.pretrained_weights,
            num_gpus=args.num_gpus,
            export_validation_probabilities=args.npz,
            continue_training=args.c,
            only_run_validation=False, # Ensemble script is for training
            disable_checkpointing=args.disable_checkpointing,
            val_with_best=args.val_best,
            device=torch_device_object,
            seed=current_member_seed,
            deterministic_augmentations=args.deterministic_augmentations,
            cudnn_deterministic=args.cudnn_deterministic,
            output_folder_suffix=member_suffix
        )
        print(f"\n--------------------------------------------------------------------")
        print(f"Finished training for ensemble member {member_idx + 1}/{args.num_ensemble_members}")
        print(f"--------------------------------------------------------------------\n")

if __name__ == '__main__':
    # Set OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS for consistency with run_training.py
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # Reduces the number of threads used for compiling. More threads don't help and can cause problems
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    run_training_ensemble_entry()
