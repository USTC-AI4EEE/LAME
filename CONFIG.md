# WindSR 配置说明

`configs/32.yaml`

## 配置结构
```yaml
seed_everything: true
trainer: ...          # Lightning Trainer 相关参数
model:
  args: ...           # 传给 WindSRDRCT 的模型超参
data:
  train_opt: ...      # 训练数据 LMDB 与数据增强配置
  val_opts: [...]     # 一个或多个验证数据 LMDB 配置
  batch_size: 4
  num_workers: 8
```

### trainer
- `max_epochs`、`check_val_every_n_epoch` 控制训练与验证频率。
- `accelerator/devices` 负责选择 GPU 设备。
- `strategy.class_path` 固定为 `strategy.drct.MyStrategy`。
- `callbacks` 默认只保留 `ModelCheckpoint` 和 `arch.callbacks.ema.EMA`，如不需要 EMA 可删除对应条目。

### model.args
- `single_channel`、`time_window`：输入/输出形态。
- `ref_chans`、`condition_size`、`cross_mode`、`use_gating`：DRCT 的条件记忆模块。
- `depths`、`num_heads`、`embed_dim`：DRCT 主体结构。
- `learning_rate`、`weight_decay`、`lr_scheduler`：优化超参。

### data
- `train_opt`/`val_opts` 指向 LMDB 数据路径，并控制裁剪、缩放、时间窗口、单通道模式等。
- 修改 `offset`、`len` 可以调节训练/验证样本范围。

## 使用方式
1. 复制默认 YAML 并按需调整路径或超参：
   ```bash
   cp configs/32.yaml configs/my_experiment.yaml
   ```
2. 运行训练或验证：
   ```bash
   python train.py fit --config configs/.../my_experiment.yaml
   ./validate.sh configs/.../my_experiment.yaml
   ```
3. 若需要额外记录，可在本文件中补充实验备注并将 YAML 纳入版本控制。
