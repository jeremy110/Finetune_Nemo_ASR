# Layer Normalization Analysis for fastconformer

According to the Zipformer paper, several issues related to the use of **Layer Normalization** were discussed. To investigate whether similar problems occur in our models, we provide a dedicated script, `norm_analyze.py`, to analyze both:

- Our **self-trained models**
- **Pretrained models**

The analysis results (log_analyze.txt) indicate that **similar normalization issues are indeed present** in both cases.

## Next Step

As a follow-up experiment, we plan to **replace the final normalization layer (`norm_out`) with `BiasNorm`** and evaluate whether this modification can alleviate the observed issues.

## Reference

**ZIPFORMER: A Faster and Better Encoder for Automatic Speech Recognition**  
Paper link:  
https://arxiv.org/pdf/2310.11230
