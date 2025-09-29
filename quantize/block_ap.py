import torch
import torch.nn as nn
import torch.nn.functional as F
import quantize.int_linear_fake as int_linear_fake
import quantize.int_linear_real as int_linear_real
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import math
import utils
import pdb
import gc
from quantize.utils import (
    quant_parameters, weight_parameters, trainable_parameters,
    set_quant_state, quant_inplace, set_quant_parameters,
    set_weight_parameters, trainable_parameters_num, get_named_linears, set_op_by_name)
import time
from datautils_block import BlockTrainDataset
from torch.utils.data import DataLoader
import shutil
import os
import json


@torch.no_grad()
def build_block_inputs(model, base_dataset, block_idx, device, attention_mask=None, position_ids=None):
    """
    base_dataset: iterable or dataset of tensors at entry-of-layer-0 (CPU)
    returns: a dataset/list of inputs for block `block_idx` (CPU tensors)
    """
    layers = model.model.layers
    out_list = []
    for inps in base_dataset:
        x = inps.to(device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        # forward through prefix [0..block_idx-1] using current model state
        for j in range(block_idx):
            x = layers[j].to(device)(
                x, attention_mask=attention_mask, position_ids=position_ids)[0]
        out_list.append(x.detach().cpu())
    return out_list


def block_ap(
    model,
    args,
    trainloader,
    valloader,
    logger=None,
):
    logger.info("Starting ...")
    if args.off_load_to_disk:
        logger.info(
            "offload the training dataset to disk, saving CPU memory, but may slowdown the training due to additional I/O...")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # step 1: move embedding layer and first layer to target device, only suppress llama models now
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = torch.float16

    # step 2: init dataset
    flag = time.time()
    if args.off_load_to_disk:
        fp_train_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_train'
        fp_val_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_val'
        quant_train_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_train'
        quant_val_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_val'
        for path in [fp_train_cache_path, fp_val_cache_path, quant_train_cache_path, quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
    else:
        fp_train_cache_path = None
        fp_val_cache_path = None
        quant_train_cache_path = None
        quant_val_cache_path = None
    fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen,
                                      model.config.hidden_size, args.batch_size, dtype, cache_path=fp_train_cache_path, off_load_to_disk=args.off_load_to_disk)
    fp_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen,
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=fp_val_cache_path, off_load_to_disk=args.off_load_to_disk)

    # step 3: catch the input of thefirst layer
    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.attention_mask = None
            self.position_ids = None

        def forward(self, inp, **kwargs):
            self.dataset.update_data(self.index, inp.squeeze(0).to('cpu'))
            self.index += 1
            if self.attention_mask is None:
                self.attention_mask = kwargs["attention_mask"]
            if self.position_ids is None:
                self.position_ids = kwargs["position_ids"]
            raise ValueError

    # step 3.1: catch the input of training set
    layers[0] = Catcher(layers[0], fp_train_inps)
    iters = len(trainloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([trainloader[j][0] for j in range(
                i*args.batch_size, (i+1)*args.batch_size)], dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module

    # step 3.2: catch the input of validation set
    layers[0] = Catcher(layers[0], fp_val_inps)
    iters = len(valloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([valloader[j][0] for j in range(
                i*args.batch_size, (i+1)*args.batch_size)], dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    attention_mask = layers[0].attention_mask
    position_ids = layers[0].position_ids
    layers[0] = layers[0].module
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(
            args.batch_size, 1, 1, 1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    # step 4: move embedding layer and first layer to cpu
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    # step 5: copy fp input as the quant input, they are same at the first layer
    if args.off_load_to_disk:
        # copy quant input from fp input, they are same in first layer
        shutil.copytree(fp_train_cache_path, quant_train_cache_path)
        shutil.copytree(fp_val_cache_path, quant_val_cache_path)
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen,
                                             model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path, off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen,
                                           model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path, off_load_to_disk=args.off_load_to_disk)
    else:
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen,
                                             model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path, off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen,
                                           model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path, off_load_to_disk=args.off_load_to_disk)
        for index, data in enumerate(fp_train_inps):
            quant_train_inps.update_data(index, data)
        for index, data in enumerate(fp_val_inps):
            quant_val_inps.update_data(index, data)

    # step 6: start training
    loss_func = torch.nn.MSELoss()

    # NEW: Sensitivity-based layer ordering
    if args.layer_ordering == 'sensitivity':
        logger.info("Using sensitivity-based layer ordering.")
        if args.sensitivity_file is None or not os.path.exists(args.sensitivity_file):
            raise ValueError(
                "Sensitivity file not provided or not found. Please specify with --sensitivity_file.")

        with open(args.sensitivity_file, 'r') as f:
            sensitivity_data = json.load(f)

        sensitivity_scores = torch.tensor(
            sensitivity_data['sensitivity_scores'])
        layer_indices = torch.argsort(
            sensitivity_scores, descending=True).tolist()

        logger.info(f"Layer training order: {layer_indices}")

    elif args.layer_ordering == 'random':
        logger.info("Using random layer ordering.")
        layer_indices = torch.randperm(len(layers)).tolist()
    else:
        logger.info("Using original sequential layer ordering.")
        layer_indices = list(range(len(layers)))

    # Fast-track validation: only train the debug block
    if args.debug_block is not None:
        if args.debug_block not in layer_indices:
            logger.warning(
                f"Debug block {args.debug_block} not in the planned training order. Prepending it for debugging.")
            layer_indices = [args.debug_block]
        else:
            layer_indices = [args.debug_block]

    for i, block_index in enumerate(layer_indices):
        logger.info(
            f"=== Training block {block_index} (order {i+1}/{len(layers)}) ===")
        # step 6.1: replace torch.nn.Linear with QuantLinear for QAT
        layer = layers[block_index]
        qlayer = copy.deepcopy(layer)
        # 1. Replace all nn.Linear layers with QuantLinear
        for name, module in qlayer.named_modules():
            if isinstance(module, torch.nn.Linear):
                quantlinear = int_linear_fake.QuantLinear(
                    module, args.wbits, args.group_size)
                set_op_by_name(qlayer, name, quantlinear)
                del module

        # 2. CRITICAL: Initialize quant params *after* all replacements are done
        for module in qlayer.modules():
            if isinstance(module, int_linear_fake.QuantLinear):
                module.init_quant_params()

        # CRITICAL: Cast layer to float32 BEFORE optimizer creation
        qlayer = qlayer.float().to(dev)
        assert all(p.dtype == torch.float32 for p in trainable_parameters(
            qlayer)), "Trainable params must be FP32 before optimizer creation"

        # step 7.2: Recompute inputs and teacher outputs for the current block
        logger.info(f"Building inputs for block {block_index}...")
        quant_train_inps_k = build_block_inputs(
            model, fp_train_inps, block_index, dev, attention_mask_batch, position_ids)
        quant_val_inps_k = build_block_inputs(
            model, fp_val_inps, block_index, dev, attention_mask_batch, position_ids)

        # Compute FP teacher outputs for the same inputs
        fp_train_outs_k = []
        with torch.no_grad():
            layer_fp = layers[block_index].to(dev).float()
            for q_in in quant_train_inps_k:
                xin = q_in.to(dev)
                fp_out = layer_fp(
                    xin, attention_mask=attention_mask_batch, position_ids=position_ids)[0]
                fp_train_outs_k.append(fp_out.detach().cpu())

        set_quant_state(qlayer, weight_quant=True)

        if args.epochs > 0:
            # step 6.3: create optimizer and learning rate schedule
            param = []
            assert args.quant_lr > 0 or args.weight_lr > 0
            param_group_index = 0
            total_training_iteration = len(
                quant_train_inps_k) * args.epochs
            if args.quant_lr > 0:
                set_quant_parameters(qlayer, True)
                param.append({"params": list(quant_parameters(
                    qlayer)), "lr": args.quant_lr})
                empty_optimizer_1 = torch.optim.AdamW(
                    [torch.tensor(0)], lr=args.quant_lr)
                quant_scheduler = CosineAnnealingLR(
                    empty_optimizer_1, T_max=total_training_iteration, eta_min=args.quant_lr/args.min_lr_factor)
                quant_index = param_group_index
                param_group_index += 1
            else:
                set_quant_parameters(qlayer, False)

            if args.weight_lr > 0:
                set_weight_parameters(qlayer, True)
                if args.adaptive_lr_scaling and args.layer_ordering == 'sensitivity':
                    min_s, max_s = torch.min(
                        sensitivity_scores), torch.max(sensitivity_scores)
                    normalized_sensitivity = 0.5 + \
                        (sensitivity_scores[block_index] -
                         min_s) / (max_s - min_s + 1e-8)
                    lr_scale_factor = torch.clamp(
                        normalized_sensitivity, 0.5, 2.0)
                    scaled_weight_lr = args.weight_lr * lr_scale_factor
                    logger.info(
                        f"Block {block_index} sensitivity: {sensitivity_scores[block_index]:.4f}, LR scale: {lr_scale_factor:.2f}, Scaled LR: {scaled_weight_lr:.2e}")
                else:
                    scaled_weight_lr = args.weight_lr
                param.append({"params": list(weight_parameters(
                    qlayer)), "lr": scaled_weight_lr})
                empty_optimizer_2 = torch.optim.AdamW(
                    [torch.tensor(0)], lr=scaled_weight_lr)
                weight_scheduler = CosineAnnealingLR(
                    empty_optimizer_2, T_max=total_training_iteration, eta_min=scaled_weight_lr/args.min_lr_factor)
                weight_index = param_group_index
                param_group_index += 1
            else:
                set_weight_parameters(qlayer, False)

            optimizer = torch.optim.AdamW(param, weight_decay=args.wd)
            trainable_number = trainable_parameters_num(qlayer)
            logger.info(
                f"Trainable parameter number: {trainable_number/1e6:.2f}M")

            best_val_loss = 1e6
            early_stop_flag = 0
            for epoch in range(args.epochs):
                loss_list = []
                norm_list = []
                start_time = time.time()
                for index, (q_in_cpu, fp_out_cpu) in enumerate(zip(quant_train_inps_k, fp_train_outs_k)):
                    optimizer.zero_grad(set_to_none=True)
                    q_in = q_in_cpu.to(dev)
                    fp_out = fp_out_cpu.to(dev)

                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        q_out = qlayer(
                            q_in, attention_mask=attention_mask_batch, position_ids=position_ids)[0]
                        loss = F.mse_loss(q_out, fp_out)

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        # Add debug dump
                        torch.save({
                            "block_index": block_index,
                            "q_in": q_in_cpu,
                            "fp_out": fp_out_cpu,
                            "qlayer_state": qlayer.state_dict()
                        }, f"debug_block_{block_index}_nan.pth")
                        raise RuntimeError(
                            "NaN encountered during training — see debug dump")

                    loss_list.append(loss.detach().cpu())
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters(qlayer), args.clip_grad)
                    norm_list.append(norm.cpu())
                    optimizer.step()

                    if args.quant_lr > 0:
                        quant_scheduler.step()
                        optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[
                            0]
                    if args.weight_lr > 0:
                        weight_scheduler.step()
                        optimizer.param_groups[weight_index]['lr'] = weight_scheduler.get_lr()[
                            0]

                # Validation loop
                val_loss_list = []
                with torch.no_grad():
                    # Compute FP teacher outputs for validation set
                    fp_val_outs_k = []
                    for q_in in quant_val_inps_k:
                        xin = q_in.to(dev)
                        fp_out = layer_fp(
                            xin, attention_mask=attention_mask_batch, position_ids=position_ids)[0]
                        fp_val_outs_k.append(fp_out.detach().cpu())

                    for index, (q_in_cpu, fp_out_cpu) in enumerate(zip(quant_val_inps_k, fp_val_outs_k)):
                        q_in = q_in_cpu.to(dev)
                        fp_out = fp_out_cpu.to(dev)
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            q_out = qlayer(
                                q_in, attention_mask=attention_mask_batch, position_ids=position_ids)[0]
                            reconstruction_loss = F.mse_loss(q_out, fp_out)
                        val_loss_list.append(reconstruction_loss.cpu())

                loss_mean = torch.stack(loss_list).mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(
                    f"Block {block_index} | Epoch {epoch} | recon_loss: {loss_mean:.4f} | val_loss: {val_loss_mean:.4f} | quant_lr: {quant_scheduler.get_lr()[0]:.2e} | norm: {norm_mean:.4f} | time: {time.time()-start_time:.1f}s")

                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                    early_stop_flag = 0
                else:
                    early_stop_flag += 1
                    if args.early_stop > 0 and early_stop_flag >= args.early_stop:
                        logger.info(
                            f"Early stopping at epoch {epoch} for block {block_index}")
                        break
            optimizer.zero_grad()
            del optimizer

        # step 6.6: directly replace the weight with fake quantization
        qlayer.half()
        quant_inplace(qlayer)
        set_quant_state(qlayer, weight_quant=False)

        # Update the layer in the main model with the quantized version
        layers[block_index] = qlayer.to("cpu")
        # step 7: pack quantized weights into low-bits format, note that this process is slow on poor CPU or busy CPU
        if args.real_quant:
            named_linears = get_named_linears(
                qlayer, int_linear_fake.QuantLinear)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scale.clamp(
                    1e-4, 1e4).detach()
                zeros = module.weight_quantizer.zero_point.detach().cuda().round().cpu()
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0, -1).transpose(0, 1).contiguous()
                zeros = zeros.view(dim0, -1).transpose(0, 1).contiguous()
                q_linear = int_linear_real.QuantLinear(
                    args.wbits, group_size, module.in_features, module.out_features, not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(),
                              zeros.float().cpu())
                set_op_by_name(qlayer, name, q_linear)
                logger.info(f"pack quantized {name} finished")
                del module
        del layer
        torch.cuda.empty_cache()

    # delete cached dataset
    if args.off_load_to_disk:
        for path in [fp_train_cache_path, fp_val_cache_path, quant_train_cache_path, quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)

    torch.cuda.empty_cache()
    gc.collect()
    model.config.use_cache = use_cache
    return model
