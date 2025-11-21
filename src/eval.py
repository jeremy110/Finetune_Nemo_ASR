import os
import json

import torch
import lightning.pytorch as pl
from omegaconf import OmegaConf
from nemo.utils import model_utils
from nemo.utils.trainer_utils import resolve_trainer_cfg

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer


torch.set_float32_matmul_precision('medium')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def validate(model_path):
    config = OmegaConf.load("./ft_110M_enhi.yaml")
    cfg = model_utils.convert_model_config_to_dict_config(config)
    asr_model = EncDecHybridRNNTCTCBPEModel.restore_from(model_path)
    asr_model.eval()
    asr_model.setup_multiple_validation_data(cfg.model.validation_ds)
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    trainer.validate(asr_model)

def inferece(manifest, model_path):

    asr_model = EncDecHybridRNNTCTCBPEModel.restore_from(model_path)
    asr_model.eval()
    hypotheses = asr_model.transcribe(manifest, batch_size = 4, num_workers = 4)
    extracted_texts = [item.text for item in hypotheses]

    for text in extracted_texts:
        print(text)

    with open(manifest, 'r', encoding = 'utf-8') as f_i:
        with open('./hyp.json', 'w', encoding = 'utf-8') as f_o:
            lines = f_i.readlines()
            for idx, line in enumerate(lines):
                line = json.loads(line)
                text = line['text']

                res = {
                    "text": text,
                    "pred_text": extracted_texts[idx]
                }
                json.dump(res, f_o, ensure_ascii = False)
                f_o.write('\n')

    output_manifest_w_wer, total_res, _ = cal_write_wer(
        pred_manifest = './hyp.json',
        use_cer = False,
        output_filename = './out.json',
        )
    print(total_res)

if __name__ == "__main__":
    model_path = "ckpt.nemo"
    manifest = ""
    validate(model_path)
    inferece(manifest, model_path)