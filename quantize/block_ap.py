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


def update_dataset(layer, dataset, dev, attention_mask, position_ids):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for index, inps in enumerate(dataset):
                inps = inps.to(dev)
                if len(inps.shape) == 2:
                    inps = inps.unsqueeze(0)
                new_data = layer(inps, attention_mask=attention_mask,
                                 position_ids=position_ids)[0].to('cpu')
                dataset.update_data(index, new_data)


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

    # step 6: Pre-calculate all full-precision ground truths
    fp_inps_list = [fp_train_inps, fp_val_inps]
    # Pre-allocate the lists with placeholders
    fp_outs_list = [[None for _ in range(len(layers))], [
        None for _ in range(len(layers))]]

    with torch.no_grad():
        # Fast-track validation: only pre-calculate up to the debug block
        loop_range = range(len(layers)) if args.debug_block is None else range(
            args.debug_block + 1)
        for i in loop_range:
            logger.info(f"Pre-calculating ground truth for block {i}")
            layer = layers[i].to(dev)

            current_fp_train_inps = fp_inps_list[0]
            current_fp_val_inps = fp_inps_list[1]

            # Store the INPUTS of the current layer as the GROUND TRUTH for the PREVIOUS layer's training
            if i > 0:
                fp_outs_list[0][i-1] = current_fp_train_inps
                fp_outs_list[1][i-1] = current_fp_val_inps

            # Calculate the OUTPUTS of the current layer to be the INPUTS for the NEXT layer
            next_fp_train_inps = BlockTrainDataset(
                args.train_size, args.training_seqlen, model.config.hidden_size, args.batch_size, dtype, off_load_to_disk=args.off_load_to_disk)
            next_fp_val_inps = BlockTrainDataset(
                args.val_size, args.training_seqlen, model.config.hidden_size, args.batch_size, dtype, off_load_to_disk=args.off_load_to_disk)

            update_dataset(layer, next_fp_train_inps,
                           dev, attention_mask, position_ids)
            update_dataset(layer, next_fp_val_inps, dev,
                           attention_mask, position_ids)

            # The new input is the output of the current layer
            fp_inps_list = [next_fp_train_inps, next_fp_val_inps]

            # Store the final layer's output as its own ground truth
            if i == len(layers) - 1:
                fp_outs_list[0][i] = fp_inps_list[0]
                fp_outs_list[1][i] = fp_inps_list[1]

            layer.cpu()
            torch.cuda.empty_cache()

    # step 7: start training
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
        layer = layers[block_index].to(dev)
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

        qlayer.to(dev)

        # step 7.2: use the pre-calculated FP outputs as ground truth
        fp_train_outs = fp_outs_list[0][block_index]
        fp_val_outs = fp_outs_list[1][block_index]

        set_quant_state(qlayer, weight_quant=True)  # activate quantization

        if args.epochs > 0:
            # step 6.3: create optimizer and learning rate schedule
            param = []
            assert args.quant_lr > 0 or args.weight_lr > 0
            param_group_index = 0
            total_training_iteration = args.epochs * args.train_size / args.batch_size
            if args.quant_lr > 0:
                set_quant_parameters(qlayer, True)
                param.append({"params": quant_parameters(
                    qlayer), "lr": args.quant_lr})
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
                # Optional: Adaptive learning rate scaling
                if args.adaptive_lr_scaling and args.layer_ordering == 'sensitivity':
                    # Normalize sensitivity scores (e.g., to [0.5, 1.5])
                    min_s, max_s = torch.min(
                        sensitivity_scores), torch.max(sensitivity_scores)
                    normalized_sensitivity = 0.5 + \
                        (sensitivity_scores[block_index] -
                         min_s) / (max_s - min_s + 1e-8)
                    scaled_weight_lr = args.weight_lr * normalized_sensitivity
                    logger.info(
                        f"Block {block_index} sensitivity: {sensitivity_scores[block_index]:.4f}, LR scale: {normalized_sensitivity:.2f}, Scaled LR: {scaled_weight_lr:.2e}")
                else:
                    scaled_weight_lr = args.weight_lr

                param.append({"params": weight_parameters(
                    qlayer), "lr": scaled_weight_lr})
                empty_optimizer_2 = torch.optim.AdamW(
                    [torch.tensor(0)], lr=scaled_weight_lr)
                weight_scheduler = CosineAnnealingLR(
                    empty_optimizer_2, T_max=total_training_iteration, eta_min=scaled_weight_lr/args.min_lr_factor)
                weight_index = param_group_index
                param_group_index += 1
            else:
                set_weight_parameters(qlayer, False)
            optimizer = torch.optim.AdamW(param, weight_decay=args.wd)
            # Re-initialize the scaler for each block to prevent state carry-over
            loss_scaler = utils.NativeScalerWithGradNormCount()
            trainable_number = trainable_parameters_num(qlayer)
            print(f"trainable parameter number: {trainable_number/1e6}M")

            best_val_loss = 1e6
            early_stop_flag = 0
            for epoch in range(args.epochs):
                # step: 6.4 training
                loss_list = []
                norm_list = []
                start_time = time.time()
                for index, (quant_inps, fp_outs) in enumerate(zip(quant_train_inps, fp_train_outs)):
                    # obtain output of quantization model
                    # with torch.cuda.amp.autocast():  # DISABLED FOR STABILITY
                    input = quant_inps.to(dev).half()
                    if len(input.shape) == 2:
                        input = input.unsqueeze(0)
                    label = fp_outs.to(dev).half()
                    quant_out = qlayer(
                        input, attention_mask=attention_mask_batch, position_ids=position_ids)[0]
                    reconstruction_loss = loss_func(label, quant_out)
                    loss = reconstruction_loss

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                    loss_list.append(reconstruction_loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(
                        loss, optimizer, parameters=trainable_parameters(qlayer), clip_grad=args.clip_grad).cpu()
                    norm_list.append(norm.data)

                    # adjust lr
                    if args.quant_lr > 0:
                        quant_scheduler.step()
                        optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[
                            0]
                    if args.weight_lr > 0:
                        weight_scheduler.step()
                        optimizer.param_groups[weight_index]['lr'] = weight_scheduler.get_lr()[
                            0]

                # step 6.5: calculate validation loss
                val_loss_list = []
                for index, (quant_inps, fp_outs) in enumerate(zip(quant_val_inps, fp_val_outs)):
                    # obtain output of quantization model
                    with torch.no_grad():
                        # with torch.cuda.amp.autocast():  # DISABLED FOR STABILITY
                        input = quant_inps.to(dev).half()
                        if len(input.shape) == 2:
                            input = input.unsqueeze(0)
                        label = fp_outs.to(dev).half()
                        quant_out = qlayer(
                            input, attention_mask=attention_mask_batch, position_ids=position_ids)[0]
                        reconstruction_loss = loss_func(label, quant_out)
                    val_loss_list.append(reconstruction_loss.cpu())

                # calculate the average training loss of last train_mean_num samples
                train_mean_num = min(len(loss_list), 64)
                loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(
                    f"blocks {block_index} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} quant_lr:{quant_scheduler.get_lr()[0]} norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} time {time.time()-start_time} ")
                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                else:
                    early_stop_flag += 1
                    if args.early_stop > 0 and early_stop_flag >= args.early_stop:
                        break
            optimizer.zero_grad()
            del optimizer

        # step 6.6: directly replace the weight with fake quantization
        qlayer.half()
        quant_inplace(qlayer)
        # weight has been quantized inplace
        set_quant_state(qlayer, weight_quant=False)

        # step 7.7: update inputs of quantization model
        if args.epochs > 0:
            update_dataset(qlayer, quant_train_inps, dev,
                           attention_mask, position_ids)
            update_dataset(qlayer, quant_val_inps, dev,
                           attention_mask, position_ids)
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
