import multiprocessing
import os
import socket
from typing import Union, Optional

import nnunetv2
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from torch.backends import cudnn


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def set_random_seeds(seed: Optional[int]):
    if seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_trainer_from_args(dataset_name_or_id: Union[int, str],
                          configuration: str,
                          fold: int,
                          trainer_name: str = 'nnUNetTrainer',
                          plans_identifier: str = 'nnUNetPlans',
                          seed: Optional[int] = None,
                          output_folder_suffix: Optional[str] = None, # New argument
                          device: torch.device = torch.device('cuda')):
    # load nnunet class and do sanity checks
    nnunet_trainer = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                trainer_name, 'nnunetv2.training.nnUNetTrainer')
    if nnunet_trainer is None:
        raise RuntimeError(f'Could not find requested nnunet trainer {trainer_name} in '
                           f'nnunetv2.training.nnUNetTrainer ('
                           f'{join(nnunetv2.__path__[0], "training", "nnUNetTrainer")}). If it is located somewhere '
                           f'else, please move it there.')
    assert issubclass(nnunet_trainer, nnUNetTrainer), 'The requested nnunet trainer class must inherit from ' \
                                                    'nnUNetTrainer'

    # handle dataset input. If it's an ID we need to convert to int from string
    if dataset_name_or_id.startswith('Dataset'):
        pass
    else:
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(f'dataset_name_or_id must either be an integer or a valid dataset name with the pattern '
                             f'DatasetXXX_YYY where XXX are the three(!) task ID digits. Your '
                             f'input: {dataset_name_or_id}')

    # initialize nnunet trainer
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
    nnunet_trainer = nnunet_trainer(plans=plans, configuration=configuration, fold=fold,
                                    dataset_json=dataset_json, master_seed=seed,
                                    output_folder_suffix=output_folder_suffix, # Pass suffix
                                    device=device)
    return nnunet_trainer


def maybe_load_checkpoint(nnunet_trainer: nnUNetTrainer, continue_training: bool, validation_only: bool,
                          pretrained_weights_file: str = None):
    if continue_training and pretrained_weights_file is not None:
        raise RuntimeError('Cannot both continue a training AND load pretrained weights. Pretrained weights can only '
                           'be used at the beginning of the training.')
    if continue_training:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_latest.pth')
        # special case where --c is used to run a previously aborted validation
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_best.pth')
        if not isfile(expected_checkpoint_file):
            print(f"WARNING: Cannot continue training because there seems to be no checkpoint available to "
                               f"continue from. Starting a new training...")
            expected_checkpoint_file = None
    elif validation_only:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            raise RuntimeError(f"Cannot run validation because the training is not finished yet!")
    else:
        if pretrained_weights_file is not None:
            if not nnunet_trainer.was_initialized:
                nnunet_trainer.initialize()
            load_pretrained_weights(nnunet_trainer.network, pretrained_weights_file, verbose=True)
        expected_checkpoint_file = None

    if expected_checkpoint_file is not None:
        nnunet_trainer.load_checkpoint(expected_checkpoint_file)


def setup_ddp(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def run_ddp(rank, dataset_name_or_id, configuration, fold, trainer_class_name, plans_identifier,
            disable_checkpointing_flag, continue_training_flag, only_run_validation_flag,
            pretrained_weights_file, export_validation_probabilities_flag, val_with_best_flag,
            world_size, seed, deterministic_augmentations, cudnn_deterministic_flag,
            output_folder_suffix_ddp): # New argument for DDP
    setup_ddp(rank, world_size)
    torch.cuda.set_device(torch.device('cuda', rank))

    if seed is not None:
        set_random_seeds(seed + rank) # Offset seed by rank for DDP

    if torch.cuda.is_available(): # This check is technically redundant here as DDP implies CUDA
        if cudnn_deterministic_flag:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    # No specific else for cudnn_deterministic_flag here, as DDP is CUDA-only.
    # A warning would have been printed by the main process if CUDA was unavailable but flag was set.

    # Pass the correct device for DDP
    current_device = torch.device('cuda', rank)
    # Pass seed and output_folder_suffix to get_trainer_from_args for DDP
    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_class_name,
                                           plans_identifier, seed=seed,
                                           output_folder_suffix=output_folder_suffix_ddp, # Pass suffix
                                           device=current_device)

    if hasattr(nnunet_trainer, 'set_deterministic_augmentations') and callable(
            getattr(nnunet_trainer, 'set_deterministic_augmentations')):
        nnunet_trainer.set_deterministic_augmentations(deterministic_augmentations)
    elif deterministic_augmentations:
        print(
            f"Warning: --deterministic_augmentations was set, but the trainer {trainer_class_name} "
            f"does not have a 'set_deterministic_augmentations' method."
        )

    if disable_checkpointing_flag:
        nnunet_trainer.disable_checkpointing = disable_checkpointing_flag

    assert not (continue_training_flag and only_run_validation_flag), \
        f'Cannot set --c and --val flag at the same time. Dummy.'

    maybe_load_checkpoint(nnunet_trainer, continue_training_flag, only_run_validation_flag, pretrained_weights_file)

    if not only_run_validation_flag:
        nnunet_trainer.run_training()

    if val_with_best_flag:
        nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
    nnunet_trainer.perform_actual_validation(export_validation_probabilities_flag)
    cleanup_ddp()


def run_training(dataset_name_or_id: Union[str, int],
                 configuration: str, fold: Union[int, str],
                 trainer_class_name: str = 'nnUNetTrainer',
                 plans_identifier: str = 'nnUNetPlans',
                 pretrained_weights: Optional[str] = None,
                 num_gpus: int = 1,
                 export_validation_probabilities: bool = False,
                 continue_training: bool = False,
                 only_run_validation: bool = False,
                 disable_checkpointing: bool = False,
                 val_with_best: bool = False,
                 seed: Optional[int] = None,
                 deterministic_augmentations: bool = False,
                 cudnn_deterministic: bool = False,
                 output_folder_suffix: Optional[str] = None, # New argument
                 device: torch.device = torch.device('cuda')):
    if plans_identifier == 'nnUNetPlans':
        print("\n############################\n"
              "INFO: You are using the old nnU-Net default plans. We have updated our recommendations. "
              "Please consider using those instead! "
              "Read more here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md"
              "\n############################\n")
    if isinstance(fold, str):
        if fold != 'all':
            try:
                fold = int(fold)
            except ValueError as e:
                print(f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!')
                raise e

    if val_with_best: # This check should be before DDP block
        assert not disable_checkpointing, '--val_best is not compatible with --disable_checkpointing'

    if num_gpus > 1:
        assert device.type == 'cuda', f"DDP training (triggered by num_gpus > 1) is only implemented for cuda devices. Your device: {device}"
        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ.keys():
            port = str(find_free_network_port())
            print(f"using port {port}")
            os.environ['MASTER_PORT'] = port

        # Arguments for mp.spawn are already correctly ordered as per the previous diff.
        # (dataset_name_or_id, configuration, fold, trainer_class_name, plans_identifier,
        #  disable_checkpointing, continue_training, only_run_validation, pretrained_weights,
        #  export_validation_probabilities, val_with_best, num_gpus,
        #  seed, deterministic_augmentations, cudnn_deterministic)
        mp.spawn(run_ddp,
                 args=(dataset_name_or_id,
                       configuration,
                       fold,
                       trainer_class_name, # maps to trainer_class_name in run_ddp
                       plans_identifier, # maps to plans_identifier in run_ddp
                       disable_checkpointing, # maps to disable_checkpointing_flag
                       continue_training,     # maps to continue_training_flag
                       only_run_validation,   # maps to only_run_validation_flag
                       pretrained_weights,    # maps to pretrained_weights_file
                       export_validation_probabilities, # maps to export_validation_probabilities_flag
                       val_with_best,         # maps to val_with_best_flag
                       num_gpus,              # maps to world_size
                       seed,                  # maps to seed
                       deterministic_augmentations, # maps to deterministic_augmentations_flag
                       cudnn_deterministic,   # maps to cudnn_deterministic_flag
                       output_folder_suffix   # maps to output_folder_suffix_ddp
                       ),
                 nprocs=num_gpus,
                 join=True)
    else: # Single GPU or CPU training
        if seed is not None:
            set_random_seeds(seed) # Set seeds before trainer initialization

        if torch.cuda.is_available(): # Set CuDNN flags before trainer initialization
            if cudnn_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
        elif cudnn_deterministic: # if on CPU/MPS and cudnn_deterministic is set
             print("Warning: --cudnn_deterministic is set, but no CUDA device is available. This flag only affects CuDNN.")

        # Pass seed and output_folder_suffix to get_trainer_from_args for single GPU/CPU
        nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_class_name,
                                               plans_identifier, seed=seed,
                                               output_folder_suffix=output_folder_suffix, # Pass suffix
                                               device=device)

        if hasattr(nnunet_trainer, 'set_deterministic_augmentations') and callable(
            getattr(nnunet_trainer, 'set_deterministic_augmentations')):
            nnunet_trainer.set_deterministic_augmentations(deterministic_augmentations)
        elif deterministic_augmentations:
            print(f"Warning: --deterministic_augmentations was set, but the trainer {trainer_class_name} "
                  f"does not have a 'set_deterministic_augmentations' method.")

        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing

        assert not (continue_training and only_run_validation), f'Cannot set --c and --val flag at the same time. Dummy.'

        maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation, pretrained_weights)

        if not only_run_validation:
            nnunet_trainer.run_training()

        if val_with_best:
            nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
        nnunet_trainer.perform_actual_validation(export_validation_probabilities)


def run_training_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('fold', type=str,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only '
                             'be used when actually training. Beta. Use with caution.')
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Specify the number of GPUs to use for training')
    parser.add_argument('--npz', action='store_true', required=False,
                        help='[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted '
                             'segmentations). Needed for finding the best ensemble.')
    parser.add_argument('--c', action='store_true', required=False,
                        help='[OPTIONAL] Continue training from latest checkpoint')
    parser.add_argument('--val', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.')
    parser.add_argument('--val_best', action='store_true', required=False,
                        help='[OPTIONAL] If set, the validation will be performed with the checkpoint_best instead '
                             'of checkpoint_final. NOT COMPATIBLE with --disable_checkpointing! '
                             'WARNING: This will use the same \'validation\' folder as the regular validation '
                             'with no way of distinguishing the two!')
    parser.add_argument('--disable_checkpointing', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and '
                             'you dont want to flood your hard drive with checkpoints.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                    help="Use this to set the device the training should run with. Available options are 'cuda' "
                         "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                         "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!")
    parser.add_argument('--seed', type=int, default=None, required=False,
                        help='[OPTIONAL] Set a global random seed for reproducibility. Default: None')
    parser.add_argument('--deterministic_augmentations', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to use deterministic data augmentations. Recommended for reproducibility.')
    parser.add_argument('--cudnn_deterministic', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to make CuDNN use deterministic algorithms. May impact performance. Recommended for reproducibility.')
    parser.add_argument('--output_folder_suffix', type=str, default=None, required=False,
                        help='[OPTIONAL] Suffix to append to the output folder name. Useful for ensemble member distinction.')
    args = parser.parse_args()

    assert args.device in ['cpu', 'cuda', 'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    run_training(args.dataset_name_or_id, args.configuration, args.fold, args.tr, args.p, args.pretrained_weights,
                 args.num_gpus, args.npz, args.c, args.val, args.disable_checkpointing, args.val_best,
                 args.seed, args.deterministic_augmentations, args.cudnn_deterministic,
                 args.output_folder_suffix, # New argument
                 device=device)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # reduces the number of threads used for compiling. More threads don't help and can cause problems
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = 1
    # multiprocessing.set_start_method("spawn")
    run_training_entry()
