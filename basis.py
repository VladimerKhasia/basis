#@title BASIS: Balanced Activation Sketching with Invariant Scalars for "Ghost Backpropagation"

import os
import random
import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state
import optax
import torch
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
import tiktoken
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import functools

print(f"Running Rank-R Fubini-Sketch Benchmark on JAX Device: {jax.devices()[0]}")

# ==========================================
# 1. CONFIGURATION SYSTEM
# ==========================================
@dataclass
class ModelConfig:
    vocab_size: int = 50257
    block_size: int = 64
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    num_classes: int = 10
    resnet_channels: List[int] = field(default_factory=lambda:[64, 128, 128, 128, 256, 512, 512, 512])
    ae_dims: List[int] = field(default_factory=lambda:[256, 64, 16])

@dataclass
class DataConfig:
    data_dir: str = "./data"
    fw_train_docs: int = 50_000
    fw_val_docs: int = 500
    fw_skip_docs: int = 0
    vision_train_split: Optional[int] = None

@dataclass
class MethodConfig:
    name: str
    learning_rate: float
    momentum: float = 0.9

    # Rank-R Sketched Specific parameters
    sketch_rank: int = 16    # The magic hyperparameter 'R' (e.g., 16 or 32)
    fubini_s: float = 1.0
    fubini_lam: float = 0.01 # Magnitude penalty

@dataclass
class BenchmarkConfig:
    seed: int = 42
    batch_size: int = 16
    train_steps: int = 2000
    val_interval: int = 200
    val_steps: int = 20
    models_to_run: List[str] = field(default_factory=lambda:["GPT"])
    methods_to_run: List[MethodConfig] = field(default_factory=lambda:[
        MethodConfig(name="baseline", learning_rate=0.01),
        MethodConfig(name="rank_r_sketch", learning_rate=0.01, sketch_rank=16)
    ])
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    data_cfg: DataConfig = field(default_factory=DataConfig)

def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return jax.random.PRNGKey(seed)


# ==========================================
# 2. CORE MATH
# ==========================================

@functools.lru_cache(maxsize=None)
def get_sketched_dot_primitive(R: int, lam: float):

    @jax.custom_vjp
    def _dot(x, w, h_batch, s_batch, s_scale):
        return jnp.dot(x, w)

    def fwd(x, w, h_batch, s_batch, s_scale):
        y = jnp.dot(x, w)
        s_batch_cast = s_batch.astype(x.dtype)

        # 1. Binned Sketching
        x_signed = x * s_batch_cast[:, None]
        x_binned = jax.ops.segment_sum(x_signed, h_batch, num_segments=R)

        # 2. EXACT NORM PRESERVATION (Local Scalar Fix)
        # We compute the exact scalar energy of the real X and force the sketch to match it.
        # This completely stabilizes Count-Sketch magnitude without biasing direction!
        exact_norm_x = jnp.linalg.norm(x)
        sketch_norm_x = jnp.linalg.norm(x_binned) + 1e-8
        x_binned = x_binned * (exact_norm_x / sketch_norm_x)

        return y, (x_binned, w, h_batch, s_batch_cast, s_scale)

    def bwd(res, dy):
        x_binned, w, h_batch, s_batch_cast, s_scale = res

        # 1. Exact dx backpropagation (Pristine error signal for lower layers)
        dx = jnp.dot(dy, w.T)

        # 2. Sketch the incoming dy gradient into the same R bins
        dy_signed = dy * s_batch_cast[:, None]
        dy_binned = jax.ops.segment_sum(dy_signed, h_batch, num_segments=R)

        # 3. EXACT NORM PRESERVATION FOR dY
        exact_norm_dy = jnp.linalg.norm(dy)
        sketch_norm_dy = jnp.linalg.norm(dy_binned) + 1e-8
        dy_binned = dy_binned * (exact_norm_dy / sketch_norm_dy)

        # 4. Safe Latent L2 Shrinkage (The Corrected Fubini Penalty)
        # Acts as a mild regularizer on the binned latents if requested.
        if lam > 0.0:
            x_binned = x_binned * (1.0 - lam)
            dy_binned = dy_binned * (1.0 - lam)

        # 5. Dense Tensor-Core Matmul: (N, R) @ (R, M) -> (N, M)
        dw_final = s_scale * jnp.dot(x_binned.T, dy_binned)

        return dx, dw_final, None, None, None

    _dot.defvjp(fwd, bwd)
    return _dot

class SketchedDense(nn.Module):
    features: int
    cfg: MethodConfig
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs):
        in_features = inputs.shape[-1]
        out_features = self.features
        s, lam = self.cfg.fubini_s, self.cfg.fubini_lam

        kernel = self.param('kernel', nn.initializers.lecun_normal(), (in_features, out_features))
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros_init(), (out_features,))

        is_flat = inputs.ndim == 2
        x_flat = inputs if is_flat else inputs.reshape(-1, in_features)
        batch_size = x_flat.shape[0]

        # DYNAMIC R SAFEGUARD: Prevent memory expansion
        R_target = getattr(self.cfg, 'sketch_rank', 16)
        R = min(R_target, batch_size)

        rng = self.make_rng('sketch')
        rng_h, rng_s = jax.random.split(rng)

        # ==========================================
        # BALANCED HASHING (STRATIFIED SKETCHING)
        # ==========================================
        # Instead of randint which causes uneven bins and massive variance,
        # we strictly guarantee bins have an equal number of elements.
        # If R == batch_size, this completely eliminates all variance!
        h_balanced = jnp.arange(batch_size) % R
        h_batch = jax.random.permutation(rng_h, h_balanced)

        # Native Rademacher for speed
        s_batch = jax.random.rademacher(rng_s, (batch_size,), dtype=jnp.float32)

        dot_fn = get_sketched_dot_primitive(R, lam)
        y = dot_fn(x_flat, kernel, h_batch, s_batch, s)

        if not is_flat:
            y = y.reshape(*inputs.shape[:-1], out_features)

        if self.use_bias:
            y = y + bias

        return y

# ==========================================
# 3. DYNAMIC MODELS
# ==========================================
def get_dense_layer(features: int, m_cfg: MethodConfig, use_bias=True):
    if m_cfg.name == "baseline": return nn.Dense(features, use_bias=use_bias)
    return SketchedDense(features, m_cfg, use_bias=use_bias)

# TODO: convnet requires testing of design choices and we ommit it for now
# -----------------------------------------
class DeepAutoencoder(nn.Module):
    model_cfg: ModelConfig
    method_cfg: MethodConfig
    @nn.compact
    def __call__(self, x, training=False):
        x = x.reshape((x.shape[0], -1))
        for dim in self.model_cfg.ae_dims: x = nn.tanh(get_dense_layer(dim, self.method_cfg)(x))
        for dim in reversed(self.model_cfg.ae_dims[:-1]): x = nn.tanh(get_dense_layer(dim, self.method_cfg)(x))
        return nn.sigmoid(get_dense_layer(784, self.method_cfg)(x))

class ConvBlock(nn.Module):
    out_c: int; pool: bool = False
    @nn.compact
    def __call__(self, x, training):
        x = nn.Conv(self.out_c, (3, 3), padding=1, use_bias=False)(x)
        x = nn.relu(nn.BatchNorm(use_running_average=not training, momentum=0.9)(x))
        return nn.max_pool(x, (2, 2), (2, 2)) if self.pool else x

class ResNet9(nn.Module):
    model_cfg: ModelConfig
    method_cfg: MethodConfig
    @nn.compact
    def __call__(self, x, training=False):
        ch = self.model_cfg.resnet_channels
        pools =[False, True, False, False, True, True, False, False]
        for c, p in zip(ch, pools): x = ConvBlock(c, pool=p)(x, training)
        x = nn.max_pool(x, (4, 4), (4, 4)).reshape((x.shape[0], -1))
        return get_dense_layer(self.model_cfg.num_classes, self.method_cfg)(x)
# -----------------------------------------
# -----------------------------------------

class NanoGPT(nn.Module):
    model_cfg: ModelConfig
    method_cfg: MethodConfig
    @nn.compact
    def __call__(self, idx, training=False):
        mc = self.model_cfg
        B, T = idx.shape
        x = nn.Embed(mc.vocab_size, mc.n_embd)(idx) + nn.Embed(mc.block_size, mc.n_embd)(jnp.arange(T)[None, :])
        mask = nn.make_causal_mask(idx, dtype=jnp.bool_)

        for _ in range(mc.n_layer):
            x_n = nn.LayerNorm()(x)
            x = x + nn.MultiHeadDotProductAttention(mc.n_head)(x_n, x_n, mask=mask, deterministic=not training)
            ffwd = nn.gelu(get_dense_layer(4 * mc.n_embd, self.method_cfg)(nn.LayerNorm()(x)))
            x = x + get_dense_layer(mc.n_embd, self.method_cfg)(ffwd)

        logits = get_dense_layer(mc.vocab_size, self.method_cfg, use_bias=False)(nn.LayerNorm()(x))
        return logits


# ==========================================
# 4. DATA ENGINEERING
# ==========================================
def get_vision_loaders(task: str, cfg: BenchmarkConfig):
    os.makedirs(cfg.data_cfg.data_dir, exist_ok=True)
    g = torch.Generator().manual_seed(cfg.seed)
    kwargs = {"batch_size": cfg.batch_size, "generator": g, "num_workers": 0}

    if task == "AUTOENCODER":
        tf = transforms.Compose([transforms.ToTensor()])
        ds_t = torchvision.datasets.FashionMNIST(cfg.data_cfg.data_dir, train=True, download=True, transform=tf)
        ds_v = torchvision.datasets.FashionMNIST(cfg.data_cfg.data_dir, train=False, download=True, transform=tf)
    else:
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        tf_t = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(*stats)])
        tf_v = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
        ds_t = torchvision.datasets.CIFAR10(cfg.data_cfg.data_dir, train=True, download=True, transform=tf_t)
        ds_v = torchvision.datasets.CIFAR10(cfg.data_cfg.data_dir, train=False, download=True, transform=tf_v)

    if cfg.data_cfg.vision_train_split:
        ds_t = torch.utils.data.Subset(ds_t, range(cfg.data_cfg.vision_train_split))

    def jax_collate(batch):
        xs, ys = zip(*batch)
        return jnp.array(np.stack(xs)).transpose(0,2,3,1) if task=="RESNET" else jnp.array(np.stack(xs)), jnp.array(np.stack(ys))

    return (torch.utils.data.DataLoader(ds_t, shuffle=True, collate_fn=jax_collate, **kwargs),
            torch.utils.data.DataLoader(ds_v, shuffle=False, collate_fn=jax_collate, **kwargs))

def stream_fineweb(cfg: BenchmarkConfig, is_train: bool):
    enc = tiktoken.get_encoding("gpt2")
    split, skip, take = ("train", cfg.data_cfg.fw_skip_docs, cfg.data_cfg.fw_train_docs) if is_train else ("train", 0, cfg.data_cfg.fw_val_docs)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split=split, streaming=True).skip(skip).take(take)

    buffer =[]
    req_len = (cfg.model_cfg.block_size + 1) * cfg.batch_size
    for row in ds:
        buffer.extend(enc.encode(row["text"]) + [enc.eot_token])
        while len(buffer) >= req_len:
            chunk = np.array(buffer[:req_len], dtype=np.int32).reshape(cfg.batch_size, -1)
            buffer = buffer[req_len:]
            yield jnp.array(chunk[:, :-1]), jnp.array(chunk[:, 1:])


# ==========================================
# 5. TRAINING ENGINE
# ==========================================
class AdvancedTrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict[str, Any] = flax.core.FrozenDict()
    sketch_state: flax.core.FrozenDict[str, Any] = flax.core.FrozenDict()

def create_train_state(rng, model, method_cfg, dummy_x):
    rngs = {'params': rng, 'sketch': rng}
    variables = model.init(rngs, dummy_x, training=True)

    params = variables.get('params', flax.core.FrozenDict())
    batch_stats = variables.get('batch_stats', flax.core.FrozenDict())
    sketch_state = variables.get('sketch_state', flax.core.FrozenDict())

    tx = optax.sgd(learning_rate=method_cfg.learning_rate, momentum=method_cfg.momentum)

    return AdvancedTrainState.create(
        apply_fn=model.apply, params=params, tx=tx,
        batch_stats=batch_stats, sketch_state=sketch_state
    )

def train_engine(task: str, method_cfg: MethodConfig, cfg: BenchmarkConfig, prng_key):
    if task == "AUTOENCODER": model = DeepAutoencoder(cfg.model_cfg, method_cfg)
    elif task == "RESNET": model = ResNet9(cfg.model_cfg, method_cfg)
    else: model = NanoGPT(cfg.model_cfg, method_cfg)

    if task in["AUTOENCODER", "RESNET"]:
        train_loader, val_loader = get_vision_loaders(task, cfg)
        train_iter, val_iter = iter(train_loader), iter(val_loader)
        dummy_x = next(train_iter)[0]
    else:
        train_iter = stream_fineweb(cfg, is_train=True)
        val_iter = stream_fineweb(cfg, is_train=False)
        dummy_x = next(train_iter)[0]

    state = create_train_state(prng_key, model, method_cfg, dummy_x)

    @jax.jit
    def train_step(state, x, y, step_rng):
        def loss_fn(params, batch_stats, sketch_state, x, y):
            vars_dict = {'params': params, 'batch_stats': batch_stats, 'sketch_state': sketch_state}

            # Forward pass propagating the step_rng
            if task in["AUTOENCODER", "RESNET"]:
                logits, updates = model.apply(vars_dict, x, training=True, mutable=['batch_stats'], rngs={'sketch': step_rng})
                if task == "AUTOENCODER": loss = jnp.mean((logits - x.reshape(x.shape[0], -1)) ** 2)
                else: loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            else:
                logits = model.apply(vars_dict, x, training=True, rngs={'sketch': step_rng})
                loss = optax.softmax_cross_entropy_with_integer_labels(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).mean()
                updates = {'batch_stats': batch_stats}
            return loss, updates

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, updates), grads = grad_fn(state.params, state.batch_stats, state.sketch_state, x, y)
        state = state.apply_gradients(grads=grads)
        return state.replace(batch_stats=updates['batch_stats']), loss

    @jax.jit
    def eval_step(state, x, y):
        vars_dict = {'params': state.params, 'batch_stats': state.batch_stats, 'sketch_state': state.sketch_state}
        eval_rng = {'sketch': jax.random.PRNGKey(0)}

        if task == "AUTOENCODER":
            preds = model.apply(vars_dict, x, training=False, rngs=eval_rng)
            return jnp.mean((preds - x.reshape(x.shape[0], -1)) ** 2)
        elif task == "RESNET":
            logits = model.apply(vars_dict, x, training=False, rngs=eval_rng)
            acc = jnp.mean(jnp.argmax(logits, -1) == y)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean(), acc
        else:
            logits = model.apply(vars_dict, x, training=False, rngs=eval_rng)
            return optax.softmax_cross_entropy_with_integer_labels(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).mean()

    history = {'train_loss':[], 'val_loss':[], 'val_acc':[], 'step':[]}

    # Init main training loop key
    loop_rng = prng_key

    for step in tqdm(range(1, cfg.train_steps + 1), desc=f"{task} [{method_cfg.name}]"):
        # Split key to draw fresh randomness every step
        loop_rng, step_rng = jax.random.split(loop_rng)

        try: x, y = next(train_iter)
        except StopIteration:
            if task == "GPT":
                train_iter = stream_fineweb(cfg, is_train=True)
            else:
                train_iter = iter(train_loader)
            x, y = next(train_iter)

        state, t_loss = train_step(state, x, y, step_rng)

        if step % cfg.val_interval == 0:
            v_losses, v_accs = [],[]
            for _ in range(cfg.val_steps):
                try: vx, vy = next(val_iter)
                except StopIteration:
                    if task == "GPT":
                        val_iter = stream_fineweb(cfg, is_train=False)
                    else:
                        val_iter = iter(val_loader)
                    vx, vy = next(val_iter)

                if task == "RESNET":
                    vl, va = eval_step(state, vx, vy)
                    v_losses.append(vl); v_accs.append(va)
                else:
                    v_losses.append(eval_step(state, vx, vy))

            history['step'].append(step)
            history['train_loss'].append(float(t_loss))
            history['val_loss'].append(float(np.mean(v_losses)))
            if task == "RESNET": history['val_acc'].append(float(np.mean(v_accs)))

    return history


# ==========================================
# 6. RUNNER & VISUALIZATION
# ==========================================
def run_benchmarks(cfg: BenchmarkConfig):
    prng_key = seed_everything(cfg.seed)
    results = {}

    for task in cfg.models_to_run:
        results[task] = {}
        for method in cfg.methods_to_run:
            prng_key, subkey = jax.random.split(prng_key)
            print(f"\n--- Starting Task: {task} | Method: {method.name} ---")
            hist = train_engine(task, method, cfg, subkey)
            results[task][method.name] = hist

    fig, axes = plt.subplots(1, len(cfg.models_to_run), figsize=(6 * len(cfg.models_to_run), 5))
    if len(cfg.models_to_run) == 1: axes =[axes]

    summary_data =[]

    for ax, task in zip(axes, cfg.models_to_run):
        for method_name, hist in results[task].items():
            if hist['val_loss']:
                ax.plot(hist['step'], hist['val_loss'], marker='o', label=f"{method_name} Val Loss")
            summary_data.append({
                "Task": task, "Method": method_name,
                "Final Train Loss": hist['train_loss'][-1] if hist['train_loss'] else "N/A",
                "Final Val Loss": hist['val_loss'][-1] if hist['val_loss'] else "N/A",
                "Final Val Acc": hist['val_acc'][-1] if task == "RESNET" and hist['val_acc'] else "N/A"
            })

        ax.set_title(f"{task} Validation Loss")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    print("\nPlot saved to 'benchmark_results.png'.\n")

    print("=== BENCHMARK SUMMARY ===")
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))

if __name__ == "__main__":
    custom_cfg = BenchmarkConfig(
        seed=42,
        batch_size=1,  #1, #16, # 32, # 64,
        train_steps=50_000,
        val_interval=1000,
        val_steps=30,
        models_to_run=["GPT"], # Not fully implemented: "AUTOENCODER" and "RESNET" from ["GPT", "AUTOENCODER", "RESNET"]
        methods_to_run=[
            MethodConfig(name="baseline", learning_rate=0.01),
            MethodConfig(name="rank_1", learning_rate=0.01, sketch_rank=1),
            MethodConfig(name="rank_8", learning_rate=0.01, sketch_rank=8),
            MethodConfig(name="rank_16", learning_rate=0.01, sketch_rank=16),
            MethodConfig(name="rank_32", learning_rate=0.01, sketch_rank=32),
            # MethodConfig(name="rank_64", learning_rate=0.01, sketch_rank=64),
        ],
        model_cfg=ModelConfig(
            vocab_size=50257, block_size=64, n_embd=64, n_head=2, n_layer=2
        )
    )

    run_benchmarks(custom_cfg)


# === BENCHMARK SUMMARY ===
# Task   Method  Final Train Loss  Final Val Loss
#  GPT baseline          6.718065        6.616913
#  GPT   rank_1          7.247934        7.221930
#  GPT   rank_8          6.798278        6.762290
#  GPT  rank_16          6.731913        6.675457
#  GPT  rank_32          6.735201        6.575524
#  GPT  rank_64          6.679784        6.572975