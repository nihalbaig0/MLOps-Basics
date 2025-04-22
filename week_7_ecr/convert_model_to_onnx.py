import torch
import hydra
import logging
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from model import ColaModel
from data import DataModule

logger = logging.getLogger(__name__)

@hydra.main(config_path="./configs", config_name="config")
def convert_model(cfg):
    # Allow EarlyStopping and ModelCheckpoint as safe globals
    torch.serialization.add_safe_globals([EarlyStopping, ModelCheckpoint])

    # Get the root directory
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/models/best-checkpoint.ckpt"
    logger.info(f"Loading pre-trained model from: {model_path}")

    try:
        # Load the checkpoint
        cola_model = ColaModel.load_from_checkpoint(model_path, map_location="cpu")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

    # Set model to evaluation mode
    cola_model.eval()

    # Initialize data module
    logger.info("Setting up data module")
    data_model = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    data_model.prepare_data()
    data_model.setup()

    # Get input sample from the data loader
    input_batch = next(iter(data_model.train_dataloader()))
    input_sample = {
        "input_ids": input_batch["input_ids"][0].unsqueeze(0).to("cpu"),
        "attention_mask": input_batch["attention_mask"][0].unsqueeze(0).to("cpu"),
    }

    # Export the model to ONNX
    output_path = f"{root_dir}/models/model.onnx"
    logger.info(f"Converting the model to ONNX format at: {output_path}")
    try:
        torch.onnx.export(
            cola_model,  # Model being run
            (
                input_sample["input_ids"],
                input_sample["attention_mask"],
            ),  # Model input (tuple for multiple inputs)
            output_path,  # Where to save the ONNX model
            export_params=True,
            opset_version=14,  # Updated to opset 14 for scaled_dot_product_attention
            do_constant_folding=True,  # Optimize by folding constants
            input_names=["input_ids", "attention_mask"],  # Input names
            output_names=["output"],  # Output names
            dynamic_axes={
                "input_ids": {0: "batch_size"},  # Variable length axes
                "attention_mask": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        logger.info(f"Model converted successfully. ONNX model saved at: {output_path}")
    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {str(e)}")
        raise

if __name__ == "__main__":
    convert_model()