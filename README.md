# EBFT: Effective and Block-Wise Fine-Tuning for Sparse LLMs


### fine-tuning

##### 1. LlamaV1-7B
```shell
python main.py --model decapoda-research/llama-7b-hf \
    --prune_method wanda \
    --density 0.5 \
    --sparsity_type unstructured \
    --learning_rate 0.0002 \
    --eval_zero_shot \
```

# Acknowledgments

Our implementation partially reuses [Wanda's code](https://github.com/locuslab/wanda).
