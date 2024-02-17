from transformers.utils import logging
from torch.optim import AdamW,Adam
from transformers.optimization import get_linear_schedule_with_warmup
import math
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import time
from .tensor_dataloader import TensorData, TensorDataLoader, TensorData_infer

scaler = GradScaler()
logger = logging.get_logger(__name__)

def val(layer, inps, outs, dataloader, args, device, attention_mask, position_ids, layer_index=None):
    ret_loss = 0
    len_dataloader = len(dataloader)
    __attention_mask = attention_mask.expand(args.infer_batch_size,-1,-1,-1)
    tensordata = TensorData(inps, outs, device)
    tensordata_loader = TensorDataLoader(tensordata, args.infer_batch_size, shuffle=False, num_workers=0).get_loader()
    criterion = nn.MSELoss(reduction="mean").cuda()
    with torch.no_grad():
        layer.eval()
        for inputs, outs in tensordata_loader:
            with autocast():
                outputs = layer(inputs, attention_mask=attention_mask.expand(len(inputs),-1,-1,-1), position_ids=position_ids)[0]
                loss = criterion(outputs, outs)

            ret_loss += (loss.detach().cpu().item()) * len(inputs)
    return ret_loss / len(inps)

def train(layer, inps, outs, dataloader, args, device, attention_mask, position_ids, layer_index=None):
    init_loss = val(layer, inps, outs, dataloader, args, device, attention_mask, position_ids, layer_index=layer_index)
    len_dataloader = len(dataloader)
    num_update_steps_per_epoch = len_dataloader // args.batch_size
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    max_steps = math.ceil(args.epochs * num_update_steps_per_epoch)
    mark_only_weights_as_trainable(layer)
    #mark_only_mask_as_trainable(layer)
    optimizer,lr_scheduler = prepare_optimizer_and_scheduler(layer, args, max_steps)
    criterion = nn.MSELoss(reduction="mean").cuda()
    losses = []
    lrs= []
    mlrs= []
    start_time = time.time()
    tensordata = TensorData(inps, outs, device)
    tensordata_loader = TensorDataLoader(tensordata, args.batch_size, shuffle=True, num_workers=0).get_loader()
    for epoch in range(0, args.epochs):
        layer.train()
        print("epoch {}".format(epoch))
        for inputs, outps in tensordata_loader:
            with autocast():
                outputs = layer(inputs,attention_mask=attention_mask.expand(len(inputs),-1,-1,-1), position_ids=position_ids)[0]
                loss = criterion(outputs, outps)
                lr = lr_scheduler.get_last_lr()[0]
                lrs.append(lr)
            # 用scaler，scale loss(FP16)，backward得到scaled的梯度(FP16)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                        layer.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()
            layer.zero_grad()
            losses.append(loss.detach().cpu().item())
    torch.cuda.empty_cache()
    end_time = time.time()
    print("time cost of finetuning each layer: {}".format(end_time-start_time))
    os.makedirs('plots', exist_ok=True)
    plt.plot(list(range(0,len(losses))), losses)
    plt.xlabel('training steps')
    plt.ylabel('loss')
    plt.savefig(f'plots/001/loss_plot_{layer_index}.png')
    plt.clf()
    plt.plot(list(range(0,len(lrs))), lrs)
    plt.xlabel('training steps')
    plt.ylabel('lr')
    plt.savefig(f'plots/001/lr_plot_{layer_index}.png')
    plt.clf()
    # plt.plot(list(range(0,len(mlrs))), mlrs)
    # plt.xlabel('training steps')
    # plt.ylabel('mlr')
    # plt.savefig(f'plots/001/mlr_plot_{layer_index}.png')
    # plt.clf()
    final_loss = val(layer, inps, outs, dataloader, args, device,attention_mask, position_ids, layer_index=layer_index)
    print(init_loss)
    print("*********")
    print(final_loss)
    return init_loss, final_loss


def prepare_optimizer_and_scheduler(layer, args, max_steps):
    def log_params(param_groups, des):
        for i, grouped_parameters in enumerate(param_groups):
            logger.info(
                f"{des}, number of params: {sum(p.nelement() for p in grouped_parameters['params'])}, weight_decay: {grouped_parameters['weight_decay']}, lr: {grouped_parameters['lr']}")

    main_model_params = [
        {
            "params": [p for n, p in layer.named_parameters() if 'mask' not in n],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate
        },
    ]
    log_params(main_model_params, "weight params")
    optimizer = AdamW(
        main_model_params,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps
    )
    return optimizer, lr_scheduler
    
def mark_only_mask_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if 'mask' not in n:
            p.requires_grad = False


def mark_only_weights_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if 'mask' in n:
            p.requires_grad = False