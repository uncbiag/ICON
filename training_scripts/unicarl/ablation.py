"""uniGradICON -> yeetle ablation, scored on Gold Atlas MR -> CT bladder Dice (raw setting, IO-50).

A cumulative chain: each arm changes one thing from the previous one, all at yeetle's budget
(15k steps, batch 4 per GPU on 4 GPUs = 16 pairs per step). Arm 1 is as close a replication of
uniGradICON as the two dataset zips allow; arm 8 is yeetle (utp.make_net, public data).

    1_ugi              uniGradICON: its UNet pyramid, LNCC sigma 5, lr 5e-5, its flip/permute + affine-noise
                       augmentation, its two-phase schedule (final full-res step added for the last 20%),
                       on its anatomy from the private zip (COPDGene insp/exp pairs, OAI knee, HCP brain,
                       AbdomenCT-1K)
    2_attn             + the attention coarse stage in place of the coarsest UNet (only this can find the
                       ~200 mm raw-setting offset; everything before it is expected to score ~0)
    3_public_unpaired  yeetle's public datasets, unpaired ones only
    4_public_all       + the same-patient paired datasets (cross-modal MR/CT, PET/CT, synthetic views): yeetle's data
    5_augmentify       yeetle's augmentation (random rotation + shear around the network) instead of uniGradICON's
    6_sqlncc           SquaredLNCC sigma 4
    7_lr               lr 2e-5
    8_yeetle           yeetle's network: CARL rotation fields, blurs between steps, its step layout, final step
                       trained from the start (single phase)
    9_yeetle_noblur    off the chain: yeetle without the Gaussian blur of each step's displacement field (the
                       reflected extrapolation the Blur wrapper also does is kept), to isolate the blurring

Not ablatable from the zips: both store volumes already foreground-cropped, p1-p99 clipped and resized
to 160^3, so every arm uses yeetle's preprocessing and 160^3 (uniGradICON: whole volume, 175^3).

usage (unicarl_public_train/ and results/unicarl_private/ unzipped in training_scripts/unicarl):
    torchrun --standalone --nproc_per_node=4 ablation.py train ARM      -> results/ablation_ARM/
    python ablation.py eval ARM results/ablation_ARM/network_weights_final
                                           -> results/ablation_ARM_eval/  (one GPU; needs GOLD_ATLAS_DATA)
    python ablation.py eval_release        -> results/ablation_0_ugi_release_eval/  (reference row: the released
                                              uniGradICON 1.0.6 weights, its own preprocessing; needs pip unigradicon)
    python ablation.py summary             -> results/ablation_summary/
    bash run_ablation.sh                   # all arms in order, train then eval, then summary

Every command is a footsteps run (results/<name>/info.txt: command, host, git hash, uncommitted diff,
a copy of this script; package_versions.txt). Names default to the ones above; FOOTSTEPS_NAME overrides.
"""

import contextlib
import copy
import functools
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PRIVATE_CACHE = os.environ.get("UNICARL_PRIVATE_CACHE", os.path.join(HERE, "results", "unicarl_private"))
GOLD_ATLAS_DATA = os.environ.get("GOLD_ATLAS_DATA", "/home/tgreer/prostate_bench/data")
PATIENTS = ["1_01_P", "1_02_P", "1_03_P", "1_04_P", "1_05_P", "1_06_P", "1_07_P", "1_08_P",
            "2_03_P", "2_04_P", "2_05_P", "2_10_P", "2_11_P", "3_01_P", "3_02_P", "3_03_P", "3_04_P"]

INPUT_SHAPE = [1, 1, 160, 160, 160]
STEPS = 15000
FINAL_STEP_FRACTION = 0.2  # uniGradICON: 801 epochs without the final full-res step, then 201 with it
BATCH_SIZE = 4             # per GPU
LMBDA = 1.5
IO_STEPS = 50
CHECKPOINT_EVERY = 5000

_UGI = dict(data="ugi4", arch="ugi", aug="ugi", loss="lncc5", lr=5e-5)
ARMS = {}
ARMS["1_ugi"] = dict(_UGI)
ARMS["2_attn"] = dict(ARMS["1_ugi"], arch="ugi_attn")
ARMS["3_public_unpaired"] = dict(ARMS["2_attn"], data="public_unpaired")
ARMS["4_public_all"] = dict(ARMS["3_public_unpaired"], data="public_all")
ARMS["5_augmentify"] = dict(ARMS["4_public_all"], aug="augmentify")
ARMS["6_sqlncc"] = dict(ARMS["5_augmentify"], loss="sqlncc4")
ARMS["7_lr"] = dict(ARMS["6_sqlncc"], lr=2e-5)
ARMS["8_yeetle"] = dict(ARMS["7_lr"], arch="yeetle")
ARMS["9_yeetle_noblur"] = dict(ARMS["8_yeetle"], arch="yeetle_noblur")


# ---------------------------------------------------------------- networks

def make_net(cfg, final_step):
    """The training network (loss and, for aug=augmentify, the augmentation wrapper included)."""
    import icon_registration as icon
    import icon_registration.networks as networks
    import icon_registration.unicarl.fixed_point_carl as fpc
    import icon_registration.unicarl.unigradicon_train_parallel as utp

    if cfg["arch"] in ("yeetle", "yeetle_noblur"):
        # yeetle's make_net builds its own loss (SquaredLNCC sigma 4, lmbda 1.5) and augmentify wrapper,
        # and always has the final full-res step
        assert cfg["loss"] == "sqlncc4" and cfg["aug"] == "augmentify"
        net = utp.make_net(3, INPUT_SHAPE, False)
        if cfg["arch"] == "yeetle_noblur":
            for m in net.modules():
                if type(m) is fpc.Blur:
                    m.__class__ = _no_blur_class()  # no parameters, so checkpoints are interchangeable
        return net

    # unigradicon.make_network: a 40 -> 80 -> 160 pyramid of tallUNet2 displacement fields, plus a
    # second full-res step in the last training phase
    if cfg["arch"] == "ugi":
        ts = icon.FunctionFromVectorField(networks.tallUNet2(dimension=3))
    elif cfg["arch"] == "ugi_attn":
        # yeetle's coarse stage (conv features + coordinate attention), at 40^3 like yeetle's
        ts = fpc.AttentionFeaturizer(fpc.SomeDownsampleNoDilationNet(dimension=3), dimension=3)
    for _ in range(2):
        ts = icon.TwoStepRegistration(
            icon.DownsampleRegistration(ts, dimension=3),
            icon.FunctionFromVectorField(networks.tallUNet2(dimension=3)),
        )
    if final_step:
        ts = icon.TwoStepRegistration(ts, icon.FunctionFromVectorField(networks.tallUNet2(dimension=3)))
    return wrap(ts, cfg)


@functools.lru_cache(maxsize=None)
def _no_blur_class():
    import icon_registration.unicarl.fixed_point_carl as fpc

    class NoBlur(fpc.Blur):
        """fpc.Blur without the Gaussian blur: the same reflected extrapolation of the field."""
        def forward(self, A, B):
            self.identity_map.isIdentity = True
            phi = self.net(A, B)(self.identity_map)
            field = self.as_function(phi - self.identity_map)

            def transform(coords):
                if hasattr(coords, "isIdentity") and coords.shape[1:] == phi.shape[1:]:
                    return phi
                coords_reflected = coords - 2 * coords * (coords < 0) - 2 * (coords - 1) * (coords > 1)
                return coords + 2 * field(coords) - field(coords_reflected)
            return transform
    return NoBlur


def wrap(ts, cfg):
    import icon_registration as icon
    import icon_registration.unicarl.unigradicon_train_parallel as utp
    similarity = {"lncc5": lambda: icon.LNCC(sigma=5), "sqlncc4": lambda: utp.SquaredLNCC(sigma=4)}[cfg["loss"]]()
    net = icon.losses.GradientICONSparse(ts, similarity, lmbda=LMBDA)
    net.assign_identity_map(INPUT_SHAPE)
    if cfg["aug"] == "augmentify":
        net = utp.augmentify(net)
        net.assign_identity_map(INPUT_SHAPE)
    return net


def inner(net, cfg):
    """The registration network without the augmentify wrapper."""
    return net.regis_net.netPsi.netPsi if cfg["aug"] == "augmentify" else net.regis_net


def add_final_step(net, cfg):
    """uniGradICON's second training phase: the trained network plus a fresh full-res step."""
    import icon_registration as icon
    import icon_registration.networks as networks
    ts = icon.TwoStepRegistration(inner(net, cfg), icon.FunctionFromVectorField(networks.tallUNet2(dimension=3)))
    return wrap(ts, cfg)


def ugi_augment(image_A, image_B):
    """uniGradICON training/train.py `augment`, verbatim: a random axis permutation with random flips
    (shared by A and B) plus independent 5% affine noise for each image."""
    device = image_A.device
    identity_list = []
    for i in range(image_A.shape[0]):
        identity = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], device=device)
        idxs = set((0, 1, 2))
        for j in range(3):
            k = random.choice(list(idxs))
            idxs.remove(k)
            identity[0, j, k] = 1
        identity = identity * (torch.randint_like(identity, 0, 2, device=device) * 2 - 1)
        identity_list.append(identity)
    identity = torch.cat(identity_list)

    out = []
    for image in (image_A, image_B):
        noise = torch.randn((image_A.shape[0], 3, 4), device=device)
        forward = identity + .05 * noise
        grid_shape = list(image.shape)
        grid_shape[1] = 3
        forward_grid = F.affine_grid(forward, grid_shape)
        out.append(F.grid_sample(image, forward_grid, padding_mode='border'))
    return out


# ---------------------------------------------------------------- data

@contextlib.contextmanager
def _unsharded_caches():
    # the private zip's caches predate sharding and carry no world_size/world_rank; load them as one
    # unsharded copy per rank (the 4 uniGradICON datasets are ~13 GB)
    real_load = torch.load

    def load(*args, **kwargs):
        d = real_load(*args, **kwargs)
        if isinstance(d, dict) and "store" in d:
            d.setdefault("world_size", 1)
            d.setdefault("world_rank", 0)
        return d
    torch.load = load
    try:
        yield
    finally:
        torch.load = real_load


def load_datasets(kind):
    import icon_registration.unicarl.dataset as dataset
    if kind == "ugi4":
        kw = dict(maximum_images=1000, cache_filename=PRIVATE_CACHE)
        with _unsharded_caches():
            return [
                # COPDGene inspiration/expiration pairs of the same patient (uniGradICON: COPD, lung ROI only;
                # the HU-clamped version is the closest available)
                dataset.PairedDataset(INPUT_SHAPE, "dirlab_clamped", "/playpen-raid2/Data/Lung_Registration_clamp_normal_transposed/*/*_img.nii.gz",
                                      match_regex=r'transposed/([a-zA-Z0-9]+)/', **kw),
                dataset.Dataset(INPUT_SHAPE, "OAI", "/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_rescaled_LEFT/*_image.nii.gz", **kw),
                dataset.Dataset(INPUT_SHAPE, "HCP_t1_stripped", "/playpen-raid2/Data/HCP/HCP_1200/*/T1w/T1w_acpc_dc_restore_brain.nii.gz", **kw),
                # uniGradICON: Learn2Reg AbdomenCTCT, not in the zips
                dataset.Dataset(INPUT_SHAPE, "abdomen1k", "/playpen-raid2/Data/AbdomenCT-1K/AbdomenCT-1K-ImagePart*/Case_*", **kw),
            ]
    import datasets  # yeetle's public split (training_scripts/unicarl/datasets.py), sharded per rank
    if kind == "public_all":
        return list(datasets.datasets_)
    if kind == "public_unpaired":
        return [d for d in datasets.datasets_ if not isinstance(d, dataset.PairedDataset)]
    raise ValueError(kind)


def make_batch_fn(ds_list):
    def make_batch():
        A, B = [], []
        for _ in range(BATCH_SIZE):
            pair = random.choice(ds_list).get_pair()  # each dataset equally likely, as in both recipes
            A.append(pair[0][0])
            B.append(pair[1][0])
        return torch.cat(A), torch.cat(B)
    return make_batch


# ---------------------------------------------------------------- training

def train(arm):
    cfg = ARMS[arm]
    import fast_training
    import icon_registration.config
    from torch.utils.tensorboard import SummaryWriter

    world_rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    icon_registration.config.device = device
    torch.backends.cudnn.benchmark = True
    random.seed(1000 + world_rank)
    torch.manual_seed(1000 + world_rank)

    if world_rank == 0:
        import footsteps
        out = footsteps.output_dir
        with open(os.path.join(out, "config.json"), "w") as fh:
            json.dump(dict(cfg, arm=arm, steps=STEPS, batch_per_gpu=BATCH_SIZE,
                           world_size=torch.distributed.get_world_size()), fh, indent=1)
        writer = SummaryWriter(os.path.join(out, "tensorboard"), flush_secs=30)
    next_batch = fast_training.prefetching(make_batch_fn(load_datasets(cfg["data"])))

    two_phase = cfg["arch"] not in ("yeetle", "yeetle_noblur")
    phases = [(round(STEPS * (1 - FINAL_STEP_FRACTION)), False), (round(STEPS * FINAL_STEP_FRACTION), True)] if two_phase else [(STEPS, True)]
    net = None
    step = 0
    for n_steps, final_step in phases:
        # a new network (phase 2: the phase-1 network plus a fresh final step) gets a fresh optimizer,
        # as in uniGradICON's train_two_stage
        net = make_net(cfg, final_step) if net is None else add_final_step(net, cfg)
        net.to(device).train()
        ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank,
                                                        find_unused_parameters=True)
        optimizer = torch.optim.Adam(ddp.parameters(), lr=cfg["lr"])
        for _ in range(n_steps):
            A, B = next_batch()
            A, B = A.to(device, non_blocking=True), B.to(device, non_blocking=True)
            if cfg["aug"] == "ugi":
                with torch.no_grad():
                    A, B = ugi_augment(A, B)
            optimizer.zero_grad()
            loss = ddp(A, B)
            loss.all_loss.mean().backward()
            optimizer.step()
            step += 1
            if world_rank == 0:
                for k, v in loss._asdict().items():
                    writer.add_scalar(k, float(v.mean()), step)
                if step % 100 == 0:
                    print(f"{arm} step {step}/{STEPS} " + " ".join(f"{k}={float(v.mean()):.4f}" for k, v in loss._asdict().items()), flush=True)
                if step % CHECKPOINT_EVERY == 0 or step == STEPS:
                    torch.save(net.regis_net.state_dict(), os.path.join(out, f"network_weights_{step}"))
        del ddp, optimizer
    if world_rank == 0:
        torch.save(net.regis_net.state_dict(), os.path.join(out, "network_weights_final"))
        print(f"{arm} done", flush=True)
    torch.distributed.destroy_process_group()


# ---------------------------------------------------------------- evaluation

def eval_net(arm, weights):
    """The final trained network, augmentation wrapper stripped (as register.get_model / bench.py)."""
    cfg = ARMS[arm]
    net = make_net(cfg, final_step=True)
    net.regis_net.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    net.regis_net = inner(net, cfg)
    net.assign_identity_map(INPUT_SHAPE)
    return net.cuda().eval()


def evaluate(arm, weights):
    """prostate_bench/bench.py --setting raw --io 50: T2 MR -> native planning CT, uniCARL preprocessing
    on both, 50 IO steps per pair from the trained weights, bladder Dice on the CT grid."""
    import icon_registration.config
    import icon_registration.unicarl.register as register
    import footsteps
    icon_registration.config.device = torch.device("cuda")
    weights = os.path.abspath(weights)
    with open(os.path.join(footsteps.output_dir, "config.json"), "w") as fh:
        json.dump(dict(ARMS[arm], arm=arm, weights=weights, setting="raw", io_steps=IO_STEPS,
                       gold_atlas_data=GOLD_ATLAS_DATA, patients=PATIENTS), fh, indent=1)
    score(arm, eval_net(arm, weights), register.preprocess, register.preprocess)


def evaluate_release():
    """The reference row: the released uniGradICON 1.0.6 weights through the same protocol, with
    uniGradICON's own preprocessing (MR: min..p99, CT: -1000..1000 HU), as prostate_bench/bench.py does."""
    import icon_registration.config
    import footsteps
    import unigradicon
    icon_registration.config.device = torch.device("cuda")
    with open(os.path.join(footsteps.output_dir, "config.json"), "w") as fh:
        json.dump(dict(arm="0_ugi_release", weights=f"pip unigradicon {getattr(unigradicon, '__version__', '?')} get_unigradicon()",
                       setting="raw", io_steps=IO_STEPS, gold_atlas_data=GOLD_ATLAS_DATA, patients=PATIENTS), fh, indent=1)
    net = unigradicon.get_unigradicon()
    score("0_ugi_release", net, lambda im: unigradicon.preprocess(im, "mri"), lambda im: unigradicon.preprocess(im, "ct"))


def score(arm, net, preprocess_moving, preprocess_fixed):
    import itk
    import numpy as np
    import footsteps
    import icon_registration.itk_wrapper
    out = footsteps.output_dir
    initial = copy.deepcopy(net.state_dict())
    rows = []
    for pid in PATIENTS:
        mr = itk.imread(f"{GOLD_ATLAS_DATA}/{pid}/t2.nii.gz")
        ct = itk.imread(f"{GOLD_ATLAS_DATA}/{pid}/ct_raw.nii.gz")
        t = time.time()
        net.load_state_dict(initial)  # IO must not carry over between patients
        with torch.enable_grad():
            phi, _ = icon_registration.itk_wrapper.register_pair(
                net, preprocess_moving(mr), preprocess_fixed(ct), finetune_steps=IO_STEPS)
        lab_m = itk.imread(f"{GOLD_ATLAS_DATA}/{pid}/t2_bladder.nii.gz")
        lab_f = itk.imread(f"{GOLD_ATLAS_DATA}/{pid}/ct_raw_bladder.nii.gz")
        warped = itk.resample_image_filter(lab_m, transform=phi, interpolator=itk.NearestNeighborInterpolateImageFunction.New(lab_m),
                                           use_reference_image=True, reference_image=lab_f)
        a, b = itk.array_view_from_image(warped) > 0, itk.array_view_from_image(lab_f) > 0
        d = float(2 * (a & b).sum() / (a.sum() + b.sum()))
        rows.append((pid, d))
        print(f"{arm} {pid} dice={d:.3f} {time.time() - t:.0f}s", flush=True)
    path = os.path.join(out, "gold_atlas_raw_io50.csv")
    with open(path, "w") as fh:
        fh.write("patient,dice\n" + "".join(f"{p},{d:.4f}\n" for p, d in rows))
    v = np.array([d for _, d in rows])
    print(f"{arm}: mean {v.mean():.3f} median {np.median(v):.3f} min {v.min():.3f} -> {path}", flush=True)


def latest_run(name, required_file):
    """The newest footsteps run results/<name>[-N]/ that contains required_file, or None."""
    import glob
    import re
    runs = [d for d in glob.glob(os.path.join(HERE, "results", name)) + glob.glob(os.path.join(HERE, "results", name + "-*"))
            if re.fullmatch(re.escape(name) + r"(-\d+)?", os.path.basename(d)) and os.path.exists(os.path.join(d, required_file))]
    return max(runs, key=os.path.getmtime) if runs else None


def summary():
    import footsteps
    import numpy as np
    lines = [f"{'arm':20s} {'mean':>6s} {'median':>6s} {'min':>6s}  eval run (Gold Atlas raw, IO-50, bladder Dice, 17 patients)"]
    for arm in ["0_ugi_release"] + list(ARMS):
        run = latest_run(f"ablation_{arm}_eval", "gold_atlas_raw_io50.csv")
        if run is None:
            lines.append(f"{arm:20s}   (not evaluated)")
            continue
        v = np.array([float(l.split(",")[1]) for l in open(os.path.join(run, "gold_atlas_raw_io50.csv")).read().splitlines()[1:]])
        lines.append(f"{arm:20s} {v.mean():6.3f} {np.median(v):6.3f} {v.min():6.3f}  {os.path.relpath(run, HERE)}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(footsteps.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    os.chdir(HERE)  # datasets.py finds unicarl_public_train/ relative to the working directory
    sys.path.insert(0, HERE)
    command, *rest = sys.argv[1:] or ["help"]
    if command == "train":
        os.environ.setdefault("FOOTSTEPS_NAME", f"ablation_{rest[0]}")
        train(rest[0])
    elif command == "eval":
        os.environ.setdefault("FOOTSTEPS_NAME", f"ablation_{rest[0]}_eval")
        evaluate(rest[0], rest[1])
    elif command == "eval_release":
        os.environ.setdefault("FOOTSTEPS_NAME", "ablation_0_ugi_release_eval")
        evaluate_release()
    elif command == "summary":
        os.environ.setdefault("FOOTSTEPS_NAME", "ablation_summary")
        summary()
    elif command == "latest":  # for run_ablation.sh: latest_run NAME FILE
        print(latest_run(rest[0], rest[1]) or "")
    else:
        print(__doc__)
