# AutoDL 存储与缓存路径

训练数据、实验输出、下载缓存和临时文件放在数据盘 `/root/autodl-tmp/`。代码与安装环境可以保留在现有位置。

| 内容 | 位置或环境变量 |
|---|---|
| 训练输入 | `/root/autodl-tmp/MyModel/Code/examples/data/train/` |
| 实验输出、日志与检查点 | `/root/autodl-tmp/MyModel/Code/examples/exp/` |
| 通用缓存 | `XDG_CACHE_HOME=/root/autodl-tmp/.cache` |
| pip 下载缓存 | `PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip` |
| PyTorch 下载资源 | `TORCH_HOME=/root/autodl-tmp/.cache/torch` |
| PyTorch 扩展编译缓存 | `TORCH_EXTENSIONS_DIR=/root/autodl-tmp/.cache/torch_extensions` |
| TorchInductor 编译缓存 | `TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/.cache/torchinductor` |
| Triton 编译缓存 | `TRITON_CACHE_DIR=/root/autodl-tmp/.cache/triton` |
| CUDA 编译缓存 | `CUDA_CACHE_PATH=/root/autodl-tmp/.cache/cuda` |
| Python/pip 等程序的新临时文件 | `TMPDIR=/root/autodl-tmp/tmp` |

配置文件是 `/root/.config/autodl-cache-paths.sh`。`/root/.bashrc`、`/root/.profile` 和 `myModelEnv` 的 Conda 激活钩子会加载它。pip 用户配置也单独指定了数据盘缓存路径。

新终端或重新激活环境后自动生效。已经打开的终端执行一次：

```bash
source /root/.config/autodl-cache-paths.sh
```

然后可继续用绝对路径的 Python 启动训练。已经运行的进程保留其启动时的环境变量；部分程序硬编码的 `/tmp` 路径也不受 `TMPDIR` 控制。

已验证新的登录 shell、`myModelEnv` 激活、pip 缓存查询、PyTorch hub 路径和 Python 临时文件写入均指向数据盘。此配置负责后续写入路径，不会自动删除已有文件。

Conda 已完成 bash 初始化，VS Code 新终端可正常激活 `myModelEnv`。该环境还保存了 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`，覆盖机器原先无效的 `0` 值，以避免启动时的 OpenMP 警告。正在运行的进程沿用启动时的设置，新配置在下一次激活环境后生效。
