import os
import time 
import heapq 
import pandas as pd
import torch 
import torch.nn as nn 
from torch.cuda.amp import autocast
from .tensor_dataloader import TensorData, TensorDataLoader, TensorData_infer
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from .finetune import train
from .linear_type import LinearMasked
import random
import numpy as np
def find_layers(module, layers=[nn.Linear], masked_layers=[LinearMasked], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers or type(module) in masked_layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        #print(child)
        res.update(find_layers(
            child, layers=layers, masked_layers=masked_layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 
def prepare_calibration_input(args, model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, args.seqlen, model.config.hidden_size), dtype=dtype, device="cpu")
    inps.requires_grad = False
    i=0
    for batch in dataloader:
        try:
            cache = model(batch[0].to(device),flag=True)
            inps[i] = cache[0].to("cpu")
            i += 1
        except ValueError:
            pass 
    outs = torch.zeros_like(inps)
    attention_mask = cache[2].to(device)
    position_ids = cache[1].to(device)
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    table_dict = []
    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=args.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device)
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            device = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(device), outs.to(device), attention_mask.to(device), position_ids.to(device)
        for name in subset:
            subset[name].prune_rate = 0
        layer.to(device)
        with torch.no_grad():
            with autocast():
                for j in range(0, args.nsamples, args.infer_batch_size):
                    outs[j:j+args.infer_batch_size] = layer(inps[j:j+args.infer_batch_size].to(device), attention_mask=attention_mask.expand(args.infer_batch_size, -1, -1, -1), position_ids=position_ids, device=device)[0].to("cpu")
        for name in subset:
            W = subset[name].weight.data 
            M = subset[name].mask.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=True)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*(1-args.density))].cpu()
                W_mask = (W_metric>thresh)

            M[W_mask] = 1
            subset[name].prune_rate = 1 - args.density
        init_loss, final_loss = train(layer, inps, outs, dataloader, args, device, attention_mask=attention_mask, position_ids=position_ids, layer_index=i)
        table_dict.append({
            "init_loss": init_loss,
            "final_loss": final_loss,
            "flip_mask_num": 0,
            "layer_idx" : i,
        })
        with torch.no_grad():
            for name in subset:
                mask_copy1 = subset[name].mask.clone()
                subset[name].weight.data = mask_copy1 * subset[name].weight.data
                subset[name].prune_rate = 0
        torch.cuda.empty_cache()
        with torch.no_grad():
            with autocast():
                for j in range(0, args.nsamples, args.infer_batch_size):
                    outs[j:j+args.infer_batch_size] = layer(inps[j:j+args.infer_batch_size].to(device), attention_mask=attention_mask.expand(args.infer_batch_size, -1, -1, -1), position_ids=position_ids, device=device)[0].to("cpu")
        inps, outs = outs, inps
        layer = layer.to('cpu')
        torch.cuda.empty_cache()
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    df = pd.DataFrame.from_records(table_dict, index="layer_idx")
    df.to_csv(os.path.join(args.save, "table0.csv"))
    print(df)

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, epochs=10):
    table_dict = []
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=args.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name],device)
            subset[name].prune_rate = 0

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        layer.to(device)

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        with torch.no_grad():
            with autocast():
                for j in range(0, args.nsamples, args.infer_batch_size):
                    outs[j:j+args.infer_batch_size] = layer(inps[j:j+args.infer_batch_size].to(device), attention_mask=attention_mask.expand(args.infer_batch_size, -1, -1, -1), position_ids=position_ids, device=device)[0].to("cpu")
        for h in handles:
            h.remove()
        wanda_mask_indices = {}
        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            #W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                final_indices = None
                with torch.no_grad():
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:,ii:(ii+prune_m)].float()
                            indices = ii+torch.topk(tmp, prune_n,dim=1, largest=True)[1].cuda()
                            subset[name].mask.scatter_(1, indices, 1)
                            if final_indices is None:
                                final_indices = indices.cpu()
                            else:
                                final_indices = torch.concat((final_indices,indices.cpu()),1)
                wanda_mask_indices[name] = final_indices
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True, descending=True)

                if args.use_variant:
                    # wanda variant 
                    print(f"no implementation")
                else:
                    #unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.density)]
                    wanda_mask_indices[name] = indices
                    with torch.no_grad():
                        subset[name].mask.scatter_(1, indices.cuda(), 1)
            subset[name].prune_rate = 1 - args.density
            #subset[name].weight.data[W_mask] = 0  ## set weights to zero
         
        #通过重构误差来微调掩码
        init_loss, final_loss = train(layer, inps, outs, dataloader, args, device, attention_mask=attention_mask, position_ids=position_ids, layer_index=i)
        table_dict.append({
            "init_loss": init_loss,
            "final_loss": final_loss,
            "flip_mask_num": 0,
            "layer_idx" : i,
        })
        with torch.no_grad():
            for name in subset:
                mask_copy1 = subset[name].mask.clone()
                subset[name].weight.data = mask_copy1 * subset[name].weight.data
                subset[name].prune_rate = 0
        torch.cuda.empty_cache()
        with torch.no_grad():
            with autocast():
                for j in range(0, args.nsamples, args.infer_batch_size):
                    outs[j:j+args.infer_batch_size] = layer(inps[j:j+args.infer_batch_size].to(device), attention_mask=attention_mask.expand(args.infer_batch_size, -1, -1, -1), position_ids=position_ids, device=device)[0].to("cpu")
        inps, outs = outs, inps
        layer = layer.to('cpu')
        torch.cuda.empty_cache()
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    df = pd.DataFrame.from_records(table_dict, index="layer_idx")
    df.to_csv(os.path.join(args.save, "table.csv"))
    print(df)


def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    table_dict = []
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=args.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, dev)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name],dev)
            subset[name].prune_rate = 0

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        layer.to(dev)
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(1 - args.density, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()
            subset[name].prune_rate = 1 - args.density
         
        #通过重构误差来微调掩码
        
        init_loss, final_loss = train(layer, inps, outs, dataloader, args, dev, attention_mask=attention_mask, position_ids=position_ids, layer_index=i)
        table_dict.append({
            "init_loss": init_loss,
            "final_loss": final_loss,
            "flip_mask_num": 0,
            "layer_idx" : i,
        })
        with torch.no_grad():
            for name in subset:
                mask_copy1 = subset[name].mask.clone()
                subset[name].weight.data = mask_copy1 * subset[name].weight.data
                subset[name].prune_rate = 0
        torch.cuda.empty_cache()
        starttime = time.time()
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask, position_ids=position_ids)[0].to("cpu")
        endtime = time.time()
        print(endtime-starttime)
        layer = layer.to('cpu')
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    df = pd.DataFrame.from_records(table_dict, index="layer_idx")
    df.to_csv(os.path.join(args.save, "table.csv"))
    print(df)


