from lib.options import args
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as Llama_Huggingface
from lib.modeling_llama import LlamaForCausalLM
from importlib.metadata import version
import math
import sys
from lib.prune_ft import prune_wanda, prune_magnitude, prune_sparsegpt, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_zero_shot
import logging
logger = logging.getLogger(__name__)
# from transformers import Trainer, default_data_collator, HfArgumentParser, TrainingArguments
print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())
# log_format = '%(asctime)s %(message)s'
# logging.basicConfig(stream=sys.stdout, level=logging.INFO,
#     format=log_format, datefmt='%m/%d %I:%M:%S %p')
# fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
# fh.setFormatter(logging.Formatter(log_format))
# logging.getLogger().addHandler(fh)
def get_llm(model_name, cache_dir="llm_weights"):
    model = LlamaForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="cpu"
    )
    model.seqlen = model.config.max_position_embeddings 
    return model
def get_llm_fp16(model_name, cache_dir="llm_weights"):
    model = Llama_Huggingface.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )
    model.seqlen = model.config.max_position_embeddings 
    return model
def main():
    # import time
    # time.sleep(18000)
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True,)
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.density == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.density != 1:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
    model = get_llm_fp16(args.save_model, args.cache_dir)
    ppl_test = eval_ppl("wikitext2", model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")
    #os.system(f"rm -rf {args.save_model}")
    logging.info('method:%s type:%s density:%f learning rate:%f wikitext2 ppl %f:',args.prune_method,args.sparsity_type,args.density,args.learning_rate,ppl_test)
    #logging.info('wikitext2 ppl %f', ppl_test)
    # save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    # with open(save_filepath, "w") as f:
    #     print("method\tactual_sparsity\tppl_test", file=f, flush=True)
    #     print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["race","arc_challenge","piqa","arc_easy","openbookqa","winogrande","hellaswag","boolq","storycloze_2018"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)
        logging.info('method:%s type:%s density:%f learning rate:%f zeroshot result %f:',args.prune_method,args.sparsity_type,args.density,args.learning_rate,results)


if __name__ == '__main__':
    main()
