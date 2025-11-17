try:
    import bitsandbytes as bnb  # optional
    _BNB_AVAILABLE = True
except Exception:
    bnb = None
    _BNB_AVAILABLE = False
import lightning as L
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import MeanSquaredError, MeanAbsoluteError
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from arch.drct import DRCT
import os
import json
import numpy as np

class WindSRDRCT(L.LightningModule):
    """风场超分辨率重建模型，基于DRCT架构并支持条件输入
    
    Args:
        args (dict): 配置参数字典，包含以下键值:
            # 基础配置
            single_channel (bool): 是否使用单通道模式，默认False
                - True: 输入输出均为单通道
                - False: 输入输出为双通道(u,v分量)
            time_window (int): 时间窗口大小，决定每个样本包含的时间步数，默认为0
            
            # 模型架构参数
            depths (List[int]): 每个DRCT块的深度，默认[6,6,6,6,6,6]
            num_heads (List[int]): 每个DRCT块的注意力头数，默认[6,6,6,6,6,6]
            embed_dim (int): 每个DRCT块的嵌入维度，默认128

            # 条件特征相关
            ref_chans (int): 参考通道数，用于条件输入，默认0
            condition_size (int): 条件特征图的大小，默认128
            cross_mode (bool): 条件输入的方式，默认True
                - True: 使用cross attention方式
                - False: 使用通道连接方式
            use_gating (bool): 是否使用门控机制，默认False
            
            # 训练相关参数
            learning_rate (float): 主干网络的学习率
            condition_lr (float): 条件特征的学习率，默认与learning_rate相同
            lr_scheduler (bool): 是否使用学习率调度器，默认False
            input_dropout (float): 输入dropout率，默认0.1
            weight_decay (float): 权重衰减率，默认0.01
            
            # 验证相关参数
            val_inference_times (int): 验证时进行推理的次数，默认为1
                - 当大于1时，将多次推理结果取平均作为最终输出
            val_use_dropout (bool): 验证时是否使用dropout，默认为False
                - 当val_inference_times>1时，设为True可增加多次推理的多样性
            save_val_results (bool): 是否保存验证结果，默认False
            
            # 模型加载
            ckpt_path (str, optional): 预训练模型路径，用于继续训练
    """
    def __init__(self, args):
        super().__init__()
        # Ensure time_window is in args
        args["time_window"] = args.get("time_window", 0)
        # 添加condition_lr参数，默认值与主学习率相同
        args["condition_lr"] = args.get("condition_lr", args.get("learning_rate"))
        # 添加weight_decay参数，默认值为0.01
        args["weight_decay"] = args.get("weight_decay", 0.01)
        # 添加新的参数控制condition输入方式: True为直接输入，False为通道连接
        args["cross_mode"] = args.get("cross_mode", True)
        # 添加use_gating参数，默认为False
        args["use_gating"] = args.get("use_gating", False)
        # 添加多次推理参数，默认为1（不进行多次推理）
        args["val_inference_times"] = args.get("val_inference_times", 1)
        # 添加是否在验证时使用dropout的参数，默认为False
        args["val_use_dropout"] = args.get("val_use_dropout", False)
        # 添加是否保存验证结果的选项
        args["save_val_results"] = args.get("save_val_results", False)
        args["save_val_dir"] = args.get("save_val_dir", os.path.join("outputs", "preds"))
        args["save_val_filename"] = args.get("save_val_filename", None)
        # 归一化保存模式：zero_centered([-0.5,0.5]) 或 0_1([0,1])；physical为物理量
        args["save_val_norm"] = args.get("save_val_norm", "zero_centered")
        args["save_val_mode"] = args.get("save_val_mode", "raw")  # raw|4ch|00only
        args["save_val_target_steps"] = args.get("save_val_target_steps", None)  # e.g., 728
        # memory Dropout2d 概率（作用于DRCT中的memory特征图）
        args["mem_dropout2d"] = args.get("mem_dropout2d", 0.0)
        # 为condition（memory参数）单独设置L2权重衰减，默认与weight_decay一致
        args["condition_weight_decay"] = args.get("condition_weight_decay", args["weight_decay"])
        # memory 激活L2正则（对本batch参与计算的mem张量做L2）；默认关闭
        args["mem_activation_l2"] = args.get("mem_activation_l2", 0.0)
        # 验证时是否对memory应用Dropout2d（多样性）；默认False
        args["val_mem_dropout2d"] = args.get("val_mem_dropout2d", False)
        self.save_hyperparameters(args)
        self.args = args
        
        # Add single_channel parameter with default False
        self.single_channel = args.get("single_channel", False)

        # Calculate input and output channels based on single_channel mode
        in_chans_per_time = 1 if self.single_channel else 2
        num_out_ch = 1 if self.single_channel else 2

        # 添加dropout层
        self.input_dropout = nn.Dropout(p=args.get("input_dropout", 0.1))

        # Load scheduler and models
        self.model = DRCT(
            upscale=4,
            in_chans=in_chans_per_time * (args.get("time_window", 0) * 2 + 1),
            mem_chans=args["ref_chans"],
            img_size=64,
            window_size=16,
            overlap_ratio=0.5,
            img_range=1.0,
            depths=args.get("depths", [6, 6, 6, 6, 6, 6]),
            embed_dim=180,
            num_heads=args.get("num_heads", [6, 6, 6, 6, 6, 6]),
            mlp_ratio=2,
            upsampler="pixelshuffle",
            resi_connection="1conv",
            num_out_ch=num_out_ch,
            cross_mode=args["cross_mode"],
            use_gating=args["use_gating"],  # 添加use_gating参数
            mem_dropout2d=args["mem_dropout2d"],
        )

        # 计算每个RDG中需要的mem通道数
        blocks_per_rdg = 5  # 每个RDG中有5个swin block
        num_rdg_layers = len(args.get("depths", [6, 6, 6, 6, 6, 6]))  # RDG层数
        
        # 计算所需的总通道数
        total_mem_channels = args["ref_chans"] * blocks_per_rdg * num_rdg_layers

        if args["ref_chans"] > 0:
            condition_size = args.get("condition_size", 128)  # Default value is 128
            if args["cross_mode"]:
                self.condition = nn.Parameter(
                    torch.rand(1, total_mem_channels, condition_size, condition_size) - 0.5
                )
            else:
                # 使用原始模式时，保持原有的通道数计算方式
                self.condition = nn.Parameter(
                    torch.rand(1, args["ref_chans"], condition_size, condition_size) - 0.5
                )
        # 2. Configure the train
        # Configure tf32
        torch.backends.cuda.matmul.allow_tf32 = True

        # 3. Add metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        # 可配置的单通道幅值动态范围与还原比例（默认与 WeatherLMDBDataset 一致）
        self.single_scale_sr_gt = float(args.get("single_scale_sr_gt", 31.347172))
        self.single_scale_lq = float(args.get("single_scale_lq", 26.298004))
        self.single_data_range = float(args.get("single_data_range", self.single_scale_sr_gt))
        self.psnr_single = PeakSignalNoiseRatio(data_range=self.single_data_range)
        self.ssim_single = StructuralSimilarityIndexMeasure(data_range=self.single_data_range)
        self.mse_single = MeanSquaredError()
        self.mae_single = MeanAbsoluteError()

        # buffers for saving validation predictions
        self._val_mm = None
        self._val_out_path = None
        self._val_offset = 0
        self._val_base_shape = None
        self._val_meta_path = None

        # 3. Resume from checkpoint
        if "ckpt_path" in args and args["ckpt_path"] is not None:
            ck = torch.load(args["ckpt_path"])  # allow both raw and lightning ckpt
            state_dict = ck.get("state_dict", ck)
            # 创建一个新的state_dict，只包含形状匹配的权重
            model_dict = self.model.state_dict()
            new_state_dict = {}
            
            # 修改这部分，去掉model.前缀
            for k, v in state_dict.items():
                # 去掉model.前缀
                new_k = k[6:] if k.startswith('model.') else k
                if new_k in model_dict and v.shape == model_dict[new_k].shape:
                    new_state_dict[new_k] = v
                else:
                    print(f"跳过权重 {k}，形状不匹配: 检查点形状 {v.shape} vs 模型形状 {model_dict[new_k].shape if new_k in model_dict else '不存在'}")
            
            # 只加载形状匹配的权重
            self.model.load_state_dict(new_state_dict, strict=False)
            print(f"成功加载了 {len(new_state_dict)}/{len(state_dict)} 个权重")

            # 尝试加载可学习的condition（若存在）
            cond_key = None
            for k in ("condition", "model.condition"):
                if k in state_dict:
                    cond_key = k
                    break
            if cond_key is not None and hasattr(self, "condition") and isinstance(state_dict[cond_key], torch.Tensor):
                if self.condition.shape == state_dict[cond_key].shape:
                    with torch.no_grad():
                        self.condition.copy_(state_dict[cond_key])
                    print(f"已从ckpt加载condition参数: {cond_key} -> {tuple(self.condition.shape)}")
                else:
                    print(
                        f"跳过condition加载：ckpt形状{tuple(state_dict[cond_key].shape)}与当前模型{tuple(self.condition.shape)}不匹配"
                    )

    def training_step(self, batch, batch_idx):
        # 1. Prepare Input
        gt = batch["gt"] - 0.5
        lq = batch["lq"] - 0.5
        top, left, crop_size = batch["top"], batch["left"], batch["crop_size"]
        
        def _cond_slice_resize(crop_h: int, crop_w: int):
            # 如果 condition 与输入尺寸不一致，则双线性插值到目标尺寸
            if self.condition.shape[-2:] != (crop_h, crop_w):
                return F.interpolate(self.condition, size=(crop_h, crop_w), mode='bilinear', align_corners=False)
            return self.condition

        if self.args["ref_chans"] > 0:
            # 获取当前batch的condition
            condition_batch = torch.cat([
                _cond_slice_resize(int(crop_size[i]), int(crop_size[i]))[
                    :,
                    :,
                    int(top[i]) : int(top[i]) + int(crop_size[i]),
                    int(left[i]) : int(left[i]) + int(crop_size[i]),
                ]
                for i in range(gt.shape[0])
            ], dim=0)
            
            if self.args["cross_mode"]:
                # 使用cross attention方式，直接将condition作为mem输入
                ref = self.input_dropout(condition_batch)
                sr = self.model(lq, ref)
            else:
                # 使用原有方式，将condition与输入在通道维度上连接
                ref = None
                condition_input = self.input_dropout(condition_batch)
                lq = torch.cat([lq, condition_input], dim=1)
                sr = self.model(lq, ref)
        else:
            sr = self.model(lq, None)

        # 确保输出形状匹配
        assert sr.shape == gt.shape, f"Model output shape {sr.shape} does not match gt shape {gt.shape}"

        # 计算损失
        loss = F.l1_loss(sr, gt, reduction="mean")
        
        # 对mem激活做L2惩罚（仅当设置了系数且存在ref_chans时）
        mem_l2_coef = float(self.args.get("mem_activation_l2", 0.0))
        if mem_l2_coef > 0 and self.args.get("ref_chans", 0) > 0:
            if self.args["cross_mode"]:
                mem_for_l2 = ref  # (B, C_mem_total, h, w)
            else:
                mem_for_l2 = condition_input  # (B, C_mem, h, w)
            mem_l2 = (mem_for_l2.pow(2).mean()) * mem_l2_coef
            loss = loss + mem_l2
            self.log("mem_l2", mem_l2, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # 记录损失
        self.log(
            "mae_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # 1. Prepare Input
        gt = batch["gt"] - 0.5
        lq = batch["lq"] - 0.5
        top, left, crop_size = batch["top"], batch["left"], batch["crop_size"]
        
        # 获取样本名称，如果不存在则使用索引
        sample_names = batch.get("name", [f"sample_{batch_idx}_{i}" for i in range(gt.shape[0])])

        # 多次推理取平均
        inference_times = self.args.get("val_inference_times", 1)
        sr_outputs = []
        
        def _cond_slice_resize(crop_h: int, crop_w: int):
            if self.condition.shape[-2:] != (crop_h, crop_w):
                return F.interpolate(self.condition, size=(crop_h, crop_w), mode='bilinear', align_corners=False)
            return self.condition

        for _ in range(inference_times):
            if self.args["ref_chans"] > 0:
                # 获取当前batch的condition
                condition_batch = torch.cat([
                    _cond_slice_resize(int(crop_size[i]), int(crop_size[i]))[
                        :,
                        :,
                        int(top[i]) : int(top[i]) + int(crop_size[i]),
                        int(left[i]) : int(left[i]) + int(crop_size[i]),
                    ]
                    for i in range(gt.shape[0])
                ], dim=0)
                
                if self.args["cross_mode"]:
                    # 使用cross attention方式，直接将condition作为mem输入
                    if self.args.get("val_use_dropout", False) and inference_times > 1:
                        ref = self.input_dropout(condition_batch)  # 验证时输入dropout
                    else:
                        ref = condition_batch  # 验证时不使用输入dropout
                    # 可选：在验证多采样时对memory再做Dropout2d以增加多样性
                    if (
                        self.args.get("val_mem_dropout2d", False)
                        and self.args.get("mem_dropout2d", 0.0) > 0
                        and inference_times > 1
                    ):
                        ref = F.dropout2d(ref, p=float(self.args["mem_dropout2d"]), training=True)
                    sr = self.model(lq, ref)
                else:
                    # 使用原有方式，将condition与输入在通道维度上连接
                    ref = None
                    if self.args.get("val_use_dropout", False) and inference_times > 1:
                        condition_input = self.input_dropout(condition_batch)
                    else:
                        condition_input = condition_batch
                    # 可选：在验证多采样时对拼接的memory分支做Dropout2d
                    if (
                        self.args.get("val_mem_dropout2d", False)
                        and self.args.get("mem_dropout2d", 0.0) > 0
                        and inference_times > 1
                    ):
                        condition_input = F.dropout2d(condition_input, p=float(self.args["mem_dropout2d"]), training=True)
                    lq_with_condition = torch.cat([lq, condition_input], dim=1)
                    sr = self.model(lq_with_condition, ref)
            else:
                sr = self.model(lq, None)
                
            sr_outputs.append(sr)
        
        # 计算多次推理的平均结果
        if inference_times > 1:
            sr = torch.mean(torch.stack(sr_outputs), dim=0)
        else:
            sr = sr_outputs[0]

        # 保存（按需，且仅在主进程）: 先保留归一化的sr
        if self.args.get("save_val_results", False) and self.trainer.is_global_zero:
            sr_for_save = sr.detach()

        # 3. Compute metrics
        if not self.single_channel:
            # Only compute vector metrics for dual channel mode
            psnr = self.psnr(sr, gt)
            ssim = self.ssim(sr, gt)

        # Scale back to original range
        if self.single_channel:
            # 与数据归一化保持一致（可通过 args 覆盖）
            sr = (sr + 0.5) * self.single_scale_sr_gt
            gt = (gt + 0.5) * self.single_scale_sr_gt
            lq = (lq + 0.5) * self.single_scale_lq  # 便于可视/统计
        else:
            SCALE = 30.0
            sr = (sr + 0.5) * SCALE
            gt = (gt + 0.5) * SCALE
            lq = (lq + 0.5) * SCALE  # 同样将lq缩放回原始范围

        if self.single_channel:
            # For single channel, use the metrics directly
            sr_single = sr
            gt_single = gt
            lq_single = lq
        else:
            # For dual channel, compute magnitude
            sr_single = torch.sqrt(sr[:, 0:1, :, :] ** 2 + sr[:, 1:2, :, :] ** 2)
            gt_single = torch.sqrt(gt[:, 0:1, :, :] ** 2 + gt[:, 1:2, :, :] ** 2)
            lq_single = torch.sqrt(lq[:, 0:1, :, :] ** 2 + lq[:, 1:2, :, :] ** 2)
            
        # Compute single channel metrics（严格要求尺寸一致）
        psnr_single = self.psnr_single(sr_single, gt_single)
        ssim_single = self.ssim_single(sr_single, gt_single)
        mse_single = self.mse_single(sr_single, gt_single)
        mae_single = self.mae_single(sr_single, gt_single)

        # 4. Log the metrics
        if not self.single_channel:
            self.log("val/psnr", psnr, on_epoch=True, sync_dist=True)
            self.log("val/ssim", ssim, on_epoch=True, sync_dist=True)
        self.log("val/psnr_single", psnr_single, on_epoch=True, sync_dist=True)
        self.log("val/ssim_single", ssim_single, on_epoch=True, sync_dist=True)
        self.log("val/mse_single", mse_single, on_epoch=True, sync_dist=True)
        self.log("val/mae_single", mae_single, on_epoch=True, sync_dist=True)
        
        # 如果使用了多次推理，记录这个信息
        if inference_times > 1:
            self.log("val/inference_times", inference_times, on_epoch=True, sync_dist=True)

        # 将预测保存到 memmap（按需）
        if self.args.get("save_val_results", False) and self.trainer.is_global_zero:
            # 根据保存模式决定保存内容
            norm_mode = str(self.args.get("save_val_norm", "zero_centered")).lower()
            # 支持多种写法：0_1/0-1/[0,1]/01/1
            is_01 = norm_mode in {"0_1", "0-1", "[0,1]", "01", "1", "0to1"}
            if is_01:
                to_store = torch.clamp(sr_for_save + 0.5, 0.0, 1.0)
            elif norm_mode == "zero_centered":
                to_store = sr_for_save
            elif norm_mode == "physical":
                to_store = sr_single
            else:
                to_store = sr_for_save
            if to_store.ndim == 3:
                to_store = to_store.unsqueeze(1)

            # 初始化 memmap
            if self._val_mm is None:
                total_n = len(self.trainer.datamodule.val_datasets[dataloader_idx])
                C, H, W = int(to_store.shape[1]), int(to_store.shape[2]), int(to_store.shape[3])
                os.makedirs(self.args.get("save_val_dir"), exist_ok=True)
                # 确定输出文件名
                fname = self.args.get("save_val_filename")
                if not fname:
                    fname = "preds_val.npy"
                out_path = os.path.join(self.args.get("save_val_dir"), fname)
                self._val_out_path = out_path
                self._val_base_shape = (total_n, C, H, W)
                self._val_mm = np.memmap(out_path, dtype=np.float32, mode="w+", shape=self._val_base_shape)
                self._val_meta_path = os.path.splitext(out_path)[0] + "_meta.json"

            b = to_store.shape[0]
            self._val_mm[self._val_offset:self._val_offset + b] = to_store.detach().cpu().numpy().astype(np.float32)
            self._val_offset += b

    def on_validation_epoch_end(self):
        # 关闭memmap并执行可选的后处理（4ch或00only分组）
        if self.args.get("save_val_results", False) and self.trainer.is_global_zero and self._val_mm is not None:
            # flush
            del self._val_mm
            self._val_mm = None

            out_path = self._val_out_path
            base_shape = self._val_base_shape
            mode = self.args.get("save_val_mode", "raw")
            target_steps = self.args.get("save_val_target_steps", None)

            final_path = out_path
            final_shape = base_shape
            try:
                if mode != "raw" and target_steps is not None:
                    arr = np.memmap(out_path, dtype=np.float32, mode='r', shape=base_shape)
                    N, C, H, W = base_shape
                    need = int(target_steps) * 4
                    N_eff = min(N, need)
                    N_eff -= (N_eff % 4)
                    T = N_eff // 4
                    arr4 = np.asarray(arr[:N_eff]).reshape(T, 4, C, H, W)
                    if mode == "4ch":
                        out = arr4[:, :, 0, :, :]
                        final_path = os.path.splitext(out_path)[0] + "_4x.npy"
                    elif mode == "00only":
                        out = arr4[:, 0, 0, :, :]
                        final_path = os.path.splitext(out_path)[0] + "_00.npy"
                    else:
                        out = None
                    if out is not None:
                        np.save(final_path, out.astype(np.float32))
                        final_shape = out.shape
            except Exception:
                # 安全回退：保持原始 out_path
                final_path = out_path
                final_shape = base_shape

            # 归一化模式解析（与 validation_step 保持一致）
            norm_mode_arg = str(self.args.get("save_val_norm", "zero_centered")).lower()
            is_01 = norm_mode_arg in {"0_1", "0-1", "[0,1]", "01", "1", "0to1"}

            # 写元数据
            meta = {
                "out_path": final_path,
                "base_path": out_path,
                "base_shape": list(map(int, base_shape)) if base_shape else None,
                "final_shape": list(map(int, final_shape)) if final_shape else None,
                "norm_mode": ("0_1" if is_01 else str(self.args.get("save_val_norm", "zero_centered"))),
                "single_channel": bool(self.single_channel),
                "denorm_max_single": float(getattr(self, 'single_scale_sr_gt', 31.347172)),
                "denorm_max_lq": float(getattr(self, 'single_scale_lq', 26.298004)),
                "mode": mode,
                "target_steps": target_steps,
            }
            try:
                with open(self._val_meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            # 可选：同时输出基础memmap为标准npy（便于直接np.load读取）
            try:
                if bool(self.args.get("save_val_write_base_npy", False)):
                    arr = np.memmap(out_path, dtype=np.float32, mode='r', shape=base_shape)
                    std_path = os.path.splitext(out_path)[0] + "_std.npy"
                    np.save(std_path, np.asarray(arr))
            except Exception:
                pass

    def configure_optimizers(self):
        # 将参数分为两组：condition参数和其他参数
        condition_params = [self.condition] if hasattr(self, 'condition') else []
        other_params = [p for n, p in self.named_parameters() if 'condition' not in n]

        # 创建两个优化器组，添加weight_decay
        param_groups = [
            {
                'params': other_params, 
                'lr': self.args["learning_rate"],
                'weight_decay': self.args["weight_decay"]
            },
        ]
        
        if condition_params:
            param_groups.append({
                'params': condition_params,
                'lr': self.args["condition_lr"],
                'weight_decay': self.args["condition_weight_decay"]
            })

        # optimizer_class = bnb.optim.AdamW8bit
        optimizer_class = torch.optim.AdamW
        
        # 使用参数组创建优化器
        optimizer = optimizer_class(param_groups)

        if self.args.get("lr_scheduler", False) is True:
            print("Applying lr_lambda lr scheduler")

            def lr_lambda(epoch):
                if epoch < 100:
                    return 1.0
                elif 100 <= epoch < 150:
                    return 0.5
                else:
                    return 0.25

            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

            return [optimizer], [scheduler]
        else:
            return [optimizer]


# Usage example:
# args = parse_args()  # Assuming you have a function to parse your script arguments
# model = ControlNetLightningModule(args)
# trainer = pl.Trainer(max_epochs=args.num_train_epochs)
# trainer.fit(model)
