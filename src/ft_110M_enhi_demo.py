import os
import torch
import lightning.pytorch as pl

# Configuration and Data Classes
from omegaconf import OmegaConf, MISSING
from dataclasses import dataclass
from typing import Optional, Tuple

# NeMo Utilities and Models
from nemo.utils import model_utils
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.utils.trainer_utils import resolve_trainer_cfg
from nemo.utils.exp_manager import exp_manager

# --- Environment Setup ---
# Set the visible CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Flag to enable 8-bit optimizer for memory optimization
bnb_optim = False


# --- Bitsandbytes (bnb) 8-bit Optimizer Registration (Optional) ---
if bnb_optim:
    # Support 8-bit optimizer for limited memory GPUs (e.g., 24GB on RTX 4090)
    import bitsandbytes as bnb
    from nemo.core.optim.optimizers import register_optimizer

    @dataclass
    class OptimizerParams:
        """Base Optimizer params."""
        lr: Optional[float] = MISSING

    @dataclass
    class AdamW8bitParams(OptimizerParams):
        """Default configuration for the 8-bit AdamW optimizer."""
        betas: Tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-08
        weight_decay: float = 0
        amsgrad: bool = False

    # Register the custom 8-bit optimizer with NeMo
    register_optimizer('adamw8bit', bnb.optim.AdamW8bit, AdamW8bitParams())
    print("INFO: bitsandbytes AdamW8bit optimizer registered.")


# --- Model Loading and Initialization ---

# Load the configuration file
config = OmegaConf.load("./ft_110M_enhi.yaml")
cfg = model_utils.convert_model_config_to_dict_config(config)

# Load the pre-trained Hybrid RNN-T/CTC model
# 'nvidia/parakeet-tdt_ctc-110m' is an example checkpoint
asr_model = EncDecHybridRNNTCTCBPEModel.from_pretrained('nvidia/parakeet-tdt_ctc-110m')
print(f"INFO: Loaded pre-trained model.")


# --- Vocabulary Expansion and Weight Transfer ---

# 1. Reserve original weights of the vocabulary-dependent layers
print("STEP 1: Reserving original layer weights...")
ori_decoder_prediction_embed = asr_model.decoder.prediction.embed
ori_decoder_prediction_dec_rnn = asr_model.decoder.prediction.dec_rnn
ori_joint_pred = asr_model.joint.pred
ori_joint_enc = asr_model.joint.enc
# The output Linear layer in the Joint Network (index 2 in the Sequential)
ori_joint_joint_net_Linear = asr_model.joint.joint_net[2]
# The output Conv1d layer in the CTC Decoder
ori_ctc_decoder_decoder_layers_Conv1d = asr_model.ctc_decoder.decoder_layers[0] 

prev_vocab_size = asr_model.tokenizer.vocab_size

# 2. Change/Expand vocabulary for the new language/tokenizer
print(f"STEP 2: Changing vocabulary from size {prev_vocab_size} to new size...")
# This call re-initializes the vocabulary-related layers (Embeddings, Final Linear/Conv layers)
asr_model.change_vocabulary(
    new_tokenizer_dir = "./en1024_hi256", 
    new_tokenizer_type = "bpe",
    bnb_optim = bnb_optim
)
cur_vocab_size = asr_model.tokenizer.vocab_size
print(asr_model)
print(f"INFO: New vocabulary size: {cur_vocab_size}")


# 3. Transfer original weights back to the expanded layers
if cur_vocab_size != prev_vocab_size:
    print("STEP 3: Transferring weights for knowledge preservation...")

    with torch.no_grad():
        # 3.1 RNN-T Decoder Layers
        # Embedding: Copy weights for the first 1024 tokens (original vocab) and the last token (e.g., <sos>)
        asr_model.decoder.prediction.embed.weight[: 1024] = ori_decoder_prediction_embed.weight[: 1024]
        asr_model.decoder.prediction.embed.weight[-1] = ori_decoder_prediction_embed.weight[-1]
        
        # Prediction RNN: Re-assign the original RNN layer (if it was replaced during change_vocabulary)
        asr_model.decoder.prediction.dec_rnn = ori_decoder_prediction_dec_rnn

        # 3.2 Joint Network Layers
        # Prediction/Encoder Nets: Re-assign the original layers
        asr_model.joint.pred = ori_joint_pred 
        asr_model.joint.enc = ori_joint_enc 

        # Joint Net Linear (Output): Copy weights/biases for original vocab ([:1024]) and special tokens ([-6:])
        # The range [-6:] accounts for 5 duration tokens + 1 padding/blank token, common in TDT models.
        asr_model.joint.joint_net[2].weight[: 1024] = ori_joint_joint_net_Linear.weight[: 1024]
        asr_model.joint.joint_net[2].bias[: 1024] = ori_joint_joint_net_Linear.bias[: 1024]
        asr_model.joint.joint_net[2].weight[-6:] = ori_joint_joint_net_Linear.weight[-6:]
        asr_model.joint.joint_net[2].bias[-6:] = ori_joint_joint_net_Linear.bias[-6:]

        # 3.3 CTC Decoder Layer (Output)
        # Conv1d: Copy weights/biases for original vocab ([:1024]) and the last token (e.g., CTC Blank)
        asr_model.ctc_decoder.decoder_layers[0].weight[:1024] = ori_ctc_decoder_decoder_layers_Conv1d.weight[:1024]
        asr_model.ctc_decoder.decoder_layers[0].weight[-1] = ori_ctc_decoder_decoder_layers_Conv1d.weight[-1]
        asr_model.ctc_decoder.decoder_layers[0].bias[:1024] = ori_ctc_decoder_decoder_layers_Conv1d.bias[:1024]
        asr_model.ctc_decoder.decoder_layers[0].bias[-1] = ori_ctc_decoder_decoder_layers_Conv1d.bias[-1]

# Clean up memory by deleting the original weight variables
del ori_decoder_prediction_embed, ori_decoder_prediction_dec_rnn, ori_joint_pred, ori_joint_enc, ori_joint_joint_net_Linear, ori_ctc_decoder_decoder_layers_Conv1d
print("INFO: Original weight variables cleaned up.")


### 4. 🧠 Data and Optimization Setup
print("STEP 4: Setting up training data and optimization...")
# Setup training, validation, and test data using configurations from the YAML file
asr_model.setup_training_data(cfg.model.train_ds)
asr_model.setup_multiple_validation_data(cfg.model.validation_ds)
if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
    asr_model.setup_test_data(cfg.model.test_ds)

# Setup optimization schedule and optimizer (will use the registered 8-bit if bnb_optim=True)
asr_model.setup_optimization(cfg.model.optim)


# --- 5. Trainer Initialization and Training ---

print("STEP 5: Initializing Lightning Trainer and starting fine-tuning...")

# Initialize a PyTorch Lightning Trainer using configurations
trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))

# Setup experiment manager for logging, checkpoints, etc.
exp_manager(trainer, cfg.get("exp_manager", None))

# Start the fine-tuning process
trainer.fit(asr_model)