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
    10_yeetle_nopad    off the chain: yeetle with augmentify's random affines fitted inside the field of view, so
                       the network never sees border padding (whose streaks give the affine away); compare with 8
    11_yeetle_cube     off the chain: yeetle with each training pair first put through a random one of the 48
                       symmetries of the cube (axis permutation and flips, the same for both images, as
                       uniGradICON's augmentation without its affine noise), then augmentify as usual; compare with 8

Augmentation-scale sweep (AUG_SWEEP, run by run_aug_sweep.sh): yeetle (arm 8) with augmentify's RandomMatrix
rotation_scale = shear_stretch_scale = s, for s from 0 (identity matrices: the wrapper is a no-op) to 0.125
(yeetle trains at 0.1). Arms aug0.000 ... aug0.125. aug0.100 repeats 8_yeetle (same seeds, but cuDNN's
nondeterministic kernels make it a different run), so the pair also estimates run-to-run noise.

Not ablatable from the zips: both store volumes already foreground-cropped, p1-p99 clipped and resized
to 160^3, so every arm uses yeetle's preprocessing and 160^3 (uniGradICON: whole volume, 175^3).

usage (unicarl_public_train/ and results/unicarl_private/ unzipped in training_scripts/unicarl):
    torchrun --standalone --nproc_per_node=4 ablation.py train ARM      -> results/ablation_ARM/
    python ablation.py eval ARM results/ablation_ARM/network_weights_final
                                           -> results/ablation_ARM_eval/  (one GPU; needs GOLD_ATLAS_DATA)
    python ablation.py eval_release        -> results/ablation_0_ugi_release_eval/  (reference row: the released
                                              uniGradICON 1.0.6 weights, its own preprocessing; needs pip unigradicon)
    python ablation.py eval_anatomix ARM results/ablation_ARM/network_weights_final
                                           -> results/ablation_ARM_anatomix/  (one GPU; needs ANATOMIX_EVAL_PACK)
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
ANATOMIX_EVAL_PACK = os.environ.get("ANATOMIX_EVAL_PACK", os.path.join(HERE, "anatomix_eval_pack.trch"))
PATIENTS = ["1_01_P", "1_02_P", "1_03_P", "1_04_P", "1_05_P", "1_06_P", "1_07_P", "1_08_P",
            "2_03_P", "2_04_P", "2_05_P", "2_10_P", "2_11_P", "3_01_P", "3_02_P", "3_03_P", "3_04_P"]

INPUT_SHAPE = [1, 1, 160, 160, 160]
STEPS = 15000
FINAL_STEP_FRACTION = 0.2  # uniGradICON: 801 epochs without the final full-res step, then 201 with it
BATCH_SIZE = 4             # per GPU
LMBDA = 1.5
IO_STEPS = 50
CHECKPOINT_EVERY = 5000
IMAGES_EVERY = 300  # tensorboard image diagnostics, as yeetle's training loop

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
ARMS["10_yeetle_nopad"] = dict(ARMS["8_yeetle"], aug="augmentify_nopad")
ARMS["11_yeetle_cube"] = dict(ARMS["8_yeetle"], cube=True)
AUG_SCALES = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125]
AUG_SWEEP = {f"aug{a:.3f}": dict(ARMS["8_yeetle"], aug_scale=a) for a in AUG_SCALES}
ALL_ARMS = {**ARMS, **AUG_SWEEP}

NOPAD_MIN_ZOOM = 0.9  # nopad: each affine is shrunk to fit the field of view, then by a further U(0.9, 1)


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
        assert cfg["loss"] == "sqlncc4" and cfg["aug"] in ("augmentify", "augmentify_nopad")
        # aug_scale: RandomMatrix(rotation_scale, shear_stretch_scale), yeetle's default 0.1, 0.1
        a = cfg.get("aug_scale", 0.1)
        net = utp.make_net(3, INPUT_SHAPE, False, rm=utp.RandomMatrix(rotation_scale=a, shear_stretch_scale=a))
        if cfg["arch"] == "yeetle_noblur":
            for m in net.modules():
                if type(m) is fpc.Blur:
                    m.__class__ = _no_blur_class()  # no parameters, so checkpoints are interchangeable
        if cfg["aug"] == "augmentify_nopad":
            # augmentify = TwoStepRegistration(FunctionFromMatrix(rm), PostStep(FunctionsFromMatrix(rm), inner)):
            # the inner network sees A sampled at M1 x and B sampled at M2^-1 x, both drawn from one shared rm.
            # Replace the two draws so that M1 and M2^-1 each map the unit cube into itself (no parameters, so
            # checkpoints are interchangeable)
            NoPadMatrix = _nopad_matrix_class()
            net.regis_net.netPhi.net = NoPadMatrix(inverse=False)
            net.regis_net.netPsi.netPhi.net = NoPadMatrix(inverse=True)
            net.assign_identity_map(INPUT_SHAPE)
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


def fit_in_fov(M, min_zoom=NOPAD_MIN_ZOOM):
    """Shrink a batch of centred affines (B x 4 x 4, [0,1] coordinates, as utp.RandomMatrix returns) about the
    centre until each maps the unit cube into itself, times a further zoom u ~ U(min_zoom, 1), then translate by a
    uniform draw from the slack left on each axis. Fitting instead of rejecting keeps the rotation/shear tails."""
    L = M[:, :3, :3]
    rows = L.abs().sum(2)  # corner c -> .5 + L (c - .5) stays in [0, 1] iff each row's sum |L_ij| <= 1
    u = min_zoom + (1 - min_zoom) * torch.rand(M.shape[0], 1, device=M.device)
    s = u / rows.amax(1, keepdim=True)
    slack = 0.5 * (1 - s * rows)
    t = (2 * torch.rand_like(slack) - 1) * slack
    out = torch.eye(4, device=M.device).repeat(M.shape[0], 1, 1)
    out[:, :3, :3] = s[:, :, None] * L
    out[:, :3, 3] = 0.5 - 0.5 * out[:, :3, :3].sum(2) + t  # x -> s L (x - .5) + .5 + t
    return out


@functools.lru_cache(maxsize=None)
def _nopad_matrix_class():
    import icon_registration as icon
    import icon_registration.unicarl.unigradicon_train_parallel as utp

    class NoPadMatrix(icon.RegistrationModule):
        """utp.RandomMatrix's draw, fitted inside the field of view. inverse=True returns the inverse of a fitted
        matrix, for carl.FunctionsFromMatrix, which samples its image at the inverse."""
        def __init__(self, inverse):
            super().__init__()
            self.inverse = inverse
            self.rm = utp.RandomMatrix()

        def forward(self, image_A, image_B):
            M = fit_in_fov(self.rm(image_A, image_B))
            return torch.linalg.inv(M) if self.inverse else M
    return NoPadMatrix


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
    return net.regis_net.netPsi.netPsi if cfg["aug"].startswith("augmentify") else net.regis_net


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


def cube_augment(image_A, image_B):
    """One of the 48 symmetries of the cube per pair (uniGradICON's shared axis permutation and flips, without
    its independent affine noise), applied to A and B alike. Exact on a cubic grid: no interpolation, no padding."""
    out_A, out_B = [], []
    for a, b in zip(image_A, image_B):  # each C, D, H, W
        perm = [0] + [1 + k for k in torch.randperm(3).tolist()]
        flips = [1 + k for k in range(3) if random.random() < 0.5]
        out_A.append(a.permute(perm).flip(flips))
        out_B.append(b.permute(perm).flip(flips))
    return torch.stack(out_A), torch.stack(out_B)


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

def log_images(writer, net, cfg, A, B, step):
    """yeetle's (utp.train_batchfunction) and uniGradICON's tensorboard diagnostics: middle slices of
    moving, fixed, warped moving and (warped - fixed + 0.5), for one unaugmented batch in eval mode.
    With augmentify, also the same for the inner network alone (stripped_*), as evaluation runs it: the
    wrapped network can read its random affine off the border padding, so its images can look perfect
    while the inner network on its own is far off."""
    def render(im):
        im = im[:, :1, :, :, im.shape[4] // 2].detach().float().cpu()
        im = im - im.amin(dim=(1, 2, 3), keepdim=True)
        im = im / im.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        return im[:, [0, 0, 0]]

    def log(prefix, warped):
        writer.add_images(prefix + "warped_moving_image", render(warped), step, dataformats="NCHW")
        difference = torch.clip(warped - B + 0.5, 0, 1)[:, :, :, :, warped.shape[4] // 2].cpu()
        writer.add_images(prefix + "difference", difference[:, [0, 0, 0]], step, dataformats="NCHW")
    net.eval()
    with torch.no_grad():
        net(A, B)
        log("", net.warped_image_A[:, :1])
        if cfg["aug"].startswith("augmentify"):
            wrapped = net.regis_net
            net.regis_net = inner(net, cfg)
            try:
                net(A, B)
                log("stripped_", net.warped_image_A[:, :1])
            finally:
                net.regis_net = wrapped
    net.train()
    writer.add_images("moving_image", render(A), step, dataformats="NCHW")
    writer.add_images("fixed_image", render(B), step, dataformats="NCHW")


def train(arm):
    cfg = ALL_ARMS[arm]
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
            if cfg.get("cube"):
                with torch.no_grad():
                    A, B = cube_augment(A, B)
            optimizer.zero_grad()
            loss = ddp(A, B)
            loss.all_loss.mean().backward()
            optimizer.step()
            step += 1
            if world_rank == 0:
                for k, v in loss._asdict().items():
                    writer.add_scalar(k, float(v.mean()), step)
                if step % IMAGES_EVERY == 0 or step == 1:
                    vis_A, vis_B = next_batch()
                    log_images(writer, net, cfg, vis_A.to(device), vis_B.to(device), step)
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
    cfg = ALL_ARMS[arm]
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
        json.dump(dict(ALL_ARMS[arm], arm=arm, weights=weights, setting="raw", io_steps=IO_STEPS,
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


ANATOMIX_CONDS = ["affine_border", "zoom_identity", "zoom_affine_border"]


def anatomix_epe(net, pack):
    """Endpoint error of net (inner network, no IO) on anatomix synthetic pairs with exact ground truth.

    Each pack subject has view1/view2 of one label map, each preprocessed as in training (own foreground crop,
    p1-p99, 160^3), and the crop map (scale, offset) sending view2-grid points to the same anatomy on the view1 grid.
    fixed = view2 sampled at M y, moving = view1 (sampled at Z z for the zoom conditions); both resampled with
    border padding, as augmentify does. Ground truth phi_AB = Z^-1 C M (with M -> M Z for zoom conditions).
        affine_border       M = a RandomMatrix draw: ~25% of the fixed image is border padding (the cue)
        zoom_identity       M = identity, both images the central 70% (Z): the zoom alone, a control
        zoom_affine_border  M = the same draw inside the central 70%: ~5% padding, so the affine is cue-free
    EPE is in voxels of the 160^3 grid, inside the fixed image's foreground labels (zero-padded)."""
    def warp(x, phi, mode, padding="border"):
        grid = (2 * phi - 1).permute(0, 2, 3, 4, 1).flip(-1)
        return F.grid_sample(x, grid, mode=mode, padding_mode=padding, align_corners=True)

    def apply(M, y):
        return (M[:3, :3] @ y.flatten(2) + M[:3, 3:]).view_as(y)
    idm = net.identity_map
    Z = pack["zoom"].cuda()
    rows = []
    for subj in pack["subjects"]:
        x1, x2 = ((t.cuda().float() - t.min()) / (t.max() - t.min()) for t in (subj["view1"], subj["view2"]))
        l2 = subj["label2"].cuda().float()
        scale, offset = (t.cuda().float().view(1, 3, 1, 1, 1) for t in (subj["scale"], subj["offset"]))
        for cond in ANATOMIX_CONDS:
            M = torch.eye(4, device="cuda") if cond == "zoom_identity" else subj["affine"].cuda()
            Ma = Z if cond.startswith("zoom") else torch.eye(4, device="cuda")
            M = M @ Ma if cond.startswith("zoom") else M
            My = apply(M, idm)
            truth = apply(torch.linalg.inv(Ma), scale * My + offset)
            fixed = warp(x2, My, "bilinear")
            moving = warp(x1, apply(Ma, idm), "bilinear") if cond.startswith("zoom") else x1
            mask = warp(l2, My, "nearest", "zeros")[0, 0] > 0  # foreground only: none from the padding
            with torch.no_grad():
                net(moving, fixed)
            epe = ((net.phi_AB_vectorfield - truth)[0].norm(dim=0)[mask] * (idm.shape[2] - 1)).mean().item()
            epe0 = ((idm - truth)[0].norm(dim=0)[mask] * (idm.shape[2] - 1)).mean().item()
            rows.append((subj["split"], cond, subj["id"], epe, epe0))
    return rows


def evaluate_anatomix(arm, weights):
    """Cue-free capture range: anatomix_epe on the pack (15 subjects the public data trained on, 15 held out;
    the pack stores the preprocessed views and the affine draws, so every arm sees the same pairs)."""
    import numpy as np
    import icon_registration.config
    import footsteps
    icon_registration.config.device = torch.device("cuda")
    weights = os.path.abspath(weights)
    with open(os.path.join(footsteps.output_dir, "config.json"), "w") as fh:
        json.dump(dict(ALL_ARMS[arm], arm=arm, weights=weights, pack=os.path.abspath(ANATOMIX_EVAL_PACK), conds=ANATOMIX_CONDS), fh, indent=1)
    rows = anatomix_epe(eval_net(arm, weights), torch.load(ANATOMIX_EVAL_PACK, weights_only=False))
    path = os.path.join(footsteps.output_dir, "anatomix_epe.csv")
    with open(path, "w") as fh:
        fh.write("split,cond,subject,epe,epe_before\n" + "".join(f"{sp},{c},{i},{e:.3f},{e0:.3f}\n" for sp, c, i, e, e0 in rows))
    for cond in ANATOMIX_CONDS:
        for split in ["trained", "held-out"]:
            v = np.array([r[3:] for r in rows if r[0] == split and r[1] == cond])
            print(f"{arm} {split:8s} {cond:18s} EPE {v[:, 0].mean():6.2f} (median {np.median(v[:, 0]):5.2f}) vox, before {v[:, 1].mean():5.1f}", flush=True)
    print(f"-> {path}", flush=True)


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
    for arm in ["0_ugi_release"] + list(ALL_ARMS):
        run = latest_run(f"ablation_{arm}_eval", "gold_atlas_raw_io50.csv")
        if run is None:
            lines.append(f"{arm:20s}   (not evaluated)")
            continue
        v = np.array([float(l.split(",")[1]) for l in open(os.path.join(run, "gold_atlas_raw_io50.csv")).read().splitlines()[1:]])
        lines.append(f"{arm:20s} {v.mean():6.3f} {np.median(v):6.3f} {v.min():6.3f}  {os.path.relpath(run, HERE)}")
    cols = [(c, sp) for c in ANATOMIX_CONDS for sp in ("trained", "held-out")]
    lines += ["", "anatomix mean EPE (voxels, no IO; zoom_affine_border is the cue-free affine)",
              f"{'arm':20s} " + " ".join(f"{c[:18] + ' ' + sp[:4]:>23s}" for c, sp in cols) + "  eval run"]
    for arm in ALL_ARMS:
        run = latest_run(f"ablation_{arm}_anatomix", "anatomix_epe.csv")
        if run is None:
            continue
        rows = [l.split(",") for l in open(os.path.join(run, "anatomix_epe.csv")).read().splitlines()[1:]]
        means = [np.mean([float(r[3]) for r in rows if r[1] == c and r[0] == sp]) for c, sp in cols]
        lines.append(f"{arm:20s} " + " ".join(f"{m:23.2f}" for m in means) + f"  {os.path.relpath(run, HERE)}")
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
    elif command == "eval_anatomix":
        os.environ.setdefault("FOOTSTEPS_NAME", f"ablation_{rest[0]}_anatomix")
        evaluate_anatomix(rest[0], rest[1])
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
