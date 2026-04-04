# HUGSIM Setup

1. `pixi.lock` is pinned to `https://mirrors.zju.edu.cn`, which may be slow or unreachable outside the authors' network.
2. `pixi` may inherit cache paths from your shell, and those paths are often not writable on shared systems.
3. Several dependencies are compiled from source against the exact PyTorch/CUDA environment.

Use the steps below from the repository root.

## 1. Use user-writable cache directories

Pick cache directories under your home directory or another location you own:

```bash
export HUGSIM_CACHE_ROOT="${HOME}/.cache/hugsim"
export PIXI_HOME="${HUGSIM_CACHE_ROOT}/pixi"
export PIP_CACHE_DIR="${HUGSIM_CACHE_ROOT}/pip"
export UV_CACHE_DIR="${HUGSIM_CACHE_ROOT}/uv"
mkdir -p "${PIXI_HOME}" "${PIP_CACHE_DIR}" "${UV_CACHE_DIR}"
```

If your shell startup files already export `PIXI_HOME`, `PIP_CACHE_DIR`, or `UV_CACHE_DIR`, override them before every install, or update your shell config to point at writable paths.

## 2. Check GPU toolchain before building CUDA packages

The training and reconstruction code depends on CUDA-enabled PyTorch packages:

```bash
nvidia-smi
nvcc --version
```

Recommended minimum checks:

- `nvidia-smi` should work and show a GPU.
- `nvcc` should exist if you need to build CUDA extensions such as `simple-knn`, `tinycudann`, and `apex`.

If `nvidia-smi` fails, fix the server GPU/driver setup first. HUGSIM will not install cleanly without a working CUDA environment.

## 3. Start from a clean pixi environment if a previous attempt was interrupted

If you already have a partial `.pixi` environment from a failed install, remove only the repo-local environment directory:

```bash
rm -rf .pixi
```

Do not remove your global cache directories unless you want to force all packages to be downloaded again.

## 4. Work around the checked-in lockfile if the ZJU mirror is unreachable

The repository lockfile currently references the ZJU mirror. On a US campus server, the simplest fix is usually:

```bash
mv pixi.lock pixi.lock.zju
```

That forces `pixi` to resolve packages from the manifest instead of reusing the pinned mirror URLs. If your network can reach the ZJU mirror reliably, you can keep the lockfile.

## 5. Install in two passes

The authors intentionally split installation into a binary-first pass and a source-build pass.

### Pass 1: install only non-source dependencies

Edit `pixi.toml` and temporarily comment out the entries under `# install from source code`:

- `hugsim-env`
- `simple-knn`
- `gsplat`
- `flow-vis-torch`
- `unidepth`
- `trajdata`
- `tinycudann`
- `kitti360Scripts`
- `simple-waymo-open-dataset-reader`
- `pytorch3d`
- `nuscenes-devkit`

Keep `moviepy` enabled. It is not a source-build package.

Then install:

```bash
pixi install
```

### Pass 2: install the source-built packages

Uncomment the source-build dependencies again, then run:

```bash
pixi install
```

## 6. Install Apex

InverseForm depends on NVIDIA Apex:

```bash
pixi run install-apex
```

This clones `apex` into `data/InverseForm/apex` and builds CUDA/C++ extensions against the active pixi environment.

## 7. Verify the environment

Run a small import check:

```bash
pixi run python -c "import torch, open3d, roma, gymnasium; print(torch.__version__)"
LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib/python3.11/site-packages/torch/lib:$PWD/.pixi/envs/default/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
pixi run python -c "from simple_knn._C import distCUDA2; import tinycudann as tcnn; from gsplat.rendering import rasterization; print('cuda extensions ok')"
```

If the second command fails, the environment usually has one of these problems:

- PyTorch and CUDA toolkit versions do not match
- `nvcc` is missing
- the install was attempted before `nvidia-smi` worked
- the source-built packages were installed before the base torch environment was ready
- `LD_LIBRARY_PATH` does not include PyTorch's shared libraries on your server runtime

For convenience, you can source [scripts/activate_hugsim.sh](/data/guest_rui/ztrs_workspace/HUGSIM/scripts/activate_hugsim.sh), which sets writable cache paths and the required `LD_LIBRARY_PATH` for this repo-local environment.

## 8. Dataset-specific extras

Data preparation also needs:

- InverseForm checkpoints placed where the `data/InverseForm/infer_*.sh` scripts expect them
- dataset downloads for KITTI-360, Waymo, nuScenes, or PandaSet

See [data/README.md](/data/guest_rui/ztrs_workspace/HUGSIM/data/README.md) for the data pipeline details.

## 9. Closed-loop simulation extras

Closed-loop simulation is not self-contained in this repo. Before running `closed_loop.py`, install one of the external AD clients the README mentions:

- UniAD_SIM
- VAD_SIM
- NAVSIM

Those client environments can be separate from HUGSIM's pixi environment.

## Typical failure modes

`pixi` tries to write to a lab-owned cache path:

```bash
export PIXI_HOME="${HOME}/.cache/hugsim/pixi"
export PIP_CACHE_DIR="${HOME}/.cache/hugsim/pip"
export UV_CACHE_DIR="${HOME}/.cache/hugsim/uv"
mkdir -p "${PIXI_HOME}" "${PIP_CACHE_DIR}" "${UV_CACHE_DIR}"
```

`pixi install` tries to fetch from `mirrors.zju.edu.cn` and hangs or fails:

```bash
mv pixi.lock pixi.lock.zju
```

CUDA extension build fails:

```bash
nvidia-smi
nvcc --version
pixi run python -c "import torch; print(torch.version.cuda)"
```

`apex` fails to build:

- confirm the pixi environment already imports `torch`
- confirm `nvcc` is from a CUDA toolkit compatible with the installed PyTorch build
- rerun `pixi run install-apex` after the source-build pass succeeds
