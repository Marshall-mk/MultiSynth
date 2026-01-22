#!/usr/bin/env python3
"""
HD-BET masks + cleaning over a directory tree.

Your HD-BET build outputs (confirmed from logs):
  -o <X>.nii.gz            => mask (binary)
  <X>_bet.nii.gz           => brain-extracted image

So:
  mask_path  = out_file_nii_gz
  brain_path = out_file_nii_gz with "_bet" inserted before .nii.gz

make-masks:
  Runs HD-BET on HR_groundtruth.nii.gz (or ref_name) and saves:
    mask_root/<SUBJ>/brain_mask.nii.gz
  Optionally:
    mask_root/<SUBJ>/HR_brain_stripped.nii.gz

clean:
  Applies saved brain_mask to selected files (exact names or patterns) and writes
  to out_root preserving structure.

Dependencies:
  pip install nibabel scipy numpy
HD-BET CLI:
  hd-bet must exist on PATH.

Offline weights:
  pass --weights_zip /path/to/release_v1.5.0.zip
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
import zipfile

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi


ZENODO_URL = "https://zenodo.org/records/14445620/files/release_v1.5.0.zip?download=1"


# --------------------------
# I/O helpers
# --------------------------
def load_nii(path: Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data, img.affine, img.header


def save_nii(path: Path, data: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header, dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(data, affine, header)
    out.set_data_dtype(dtype)
    nib.save(out, str(path))


def list_subject_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


# --------------------------
# HD-BET weights install
# --------------------------
def install_hdbet_weights_from_zip(weights_zip: Path) -> Path:
    """
    Install HD-BET weights into the directory HD_BET expects.
    Returns the folder where weights were installed.
    """
    weights_zip = Path(weights_zip).expanduser().resolve()
    if not weights_zip.exists():
        raise FileNotFoundError(f"weights zip not found: {weights_zip}")

    import HD_BET.checkpoint_download as cd  # type: ignore

    target_dir = Path(cd.folder_with_parameter_files).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(weights_zip, "r") as zf:
        zf.extractall(target_dir)

    if len(list(target_dir.rglob("*"))) < 5:
        raise RuntimeError(
            f"Extracted weights but {target_dir} looks empty.\n"
            f"Zip: {weights_zip}"
        )
    return target_dir


# --------------------------
# HD-BET runner (your build: output file is mask; *_bet is brain image)
# --------------------------
def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def bet_path_from_out(out_file_nii_gz: Path) -> Path:
    s = str(out_file_nii_gz)
    if not s.endswith(".nii.gz"):
        raise ValueError(f"Expected .nii.gz: {out_file_nii_gz}")
    return Path(s.replace(".nii.gz", "_bet.nii.gz"))


def run_hdbet(
    ref_path: Path,
    out_file_nii_gz: Path,   # MUST end with .nii.gz; for YOUR build this becomes the MASK
    mode: str = "fast",
    device: str | None = None,
    weights_zip: Path | None = None,
) -> tuple[Path, Path]:
    """
    Returns (brain_img_path, brain_mask_path)

    Your HD-BET:
      out_file_nii_gz          = mask
      out_file_nii_gz_bet      = brain image
    """
    exe = shutil.which("hd-bet")
    if exe is None:
        raise RuntimeError("Could not find `hd-bet` on PATH.")

    out_file_nii_gz = Path(out_file_nii_gz)
    if not str(out_file_nii_gz).endswith(".nii.gz"):
        raise ValueError(f"Output file must end with .nii.gz, got: {out_file_nii_gz}")

    base_cmd = [exe, "-i", str(ref_path), "-o", str(out_file_nii_gz), "--save_bet_mask"]
    if mode == "fast":
        base_cmd.append("--disable_tta")

    # Try both device flag styles
    device_variants: list[list[str]] = [[]]
    if device is not None:
        device_variants = [["-device", str(device)], ["--device", str(device)]]

    last_err = ""
    for dev_args in device_variants:
        cmd = base_cmd + dev_args
        try:
            _run_cmd(cmd)

            mask_path = out_file_nii_gz
            brain_path = bet_path_from_out(out_file_nii_gz)

            if not mask_path.exists():
                raise RuntimeError(f"HD-BET ran but mask not found: {mask_path}")
            if not brain_path.exists():
                raise RuntimeError(f"HD-BET ran but brain image not found: {brain_path}")

            return brain_path, mask_path

        except subprocess.CalledProcessError as e:
            err = e.stderr or ""
            last_err = err

            # Zenodo blocked -> install weights then retry once
            if ("zenodo.org" in err) and ("403" in err or "Forbidden" in err):
                if weights_zip is None or not Path(weights_zip).exists():
                    raise RuntimeError(
                        "HD-BET tried to download weights from Zenodo and got HTTP 403.\n\n"
                        f"Download zip:\n  {ZENODO_URL}\n"
                        "Copy to cluster and pass:\n  --weights_zip /path/to/release_v1.5.0.zip\n\n"
                        "Original stderr:\n" + err
                    )

                installed_to = install_hdbet_weights_from_zip(Path(weights_zip))
                try:
                    _run_cmd(cmd)

                    mask_path = out_file_nii_gz
                    brain_path = bet_path_from_out(out_file_nii_gz)

                    if not mask_path.exists() or not brain_path.exists():
                        written = sorted([p.name for p in out_file_nii_gz.parent.glob("*.nii.gz")])
                        raise RuntimeError(
                            "HD-BET retry ran but expected outputs missing.\n"
                            f"Weights installed to: {installed_to}\n"
                            f"Expected mask: {mask_path.name}\n"
                            f"Expected brain: {brain_path.name}\n"
                            f"Written: {written}"
                        )

                    return brain_path, mask_path
                except subprocess.CalledProcessError as e2:
                    raise RuntimeError(
                        f"HD-BET failed even after installing weights.\nCommand: {' '.join(cmd)}\nStderr:\n{e2.stderr}"
                    )

            # If device flag unrecognized, try next variant
            if "unrecognized arguments" in err and ("--device" in err or "-device" in err):
                continue

            raise RuntimeError(f"HD-BET failed.\nCommand: {' '.join(cmd)}\nStderr:\n{err}")

    raise RuntimeError("HD-BET failed for all device flag variants.\nLast stderr:\n" + last_err)


# --------------------------
# Mask post-processing
# --------------------------
def postprocess_mask(mask: np.ndarray, close_iters: int = 0, dilate_iters: int = 0, keep_largest: bool = True) -> np.ndarray:
    m = (mask > 0.5) & np.isfinite(mask)
    structure = np.ones((3, 3, 3), dtype=bool)

    if keep_largest:
        lbl, n = ndi.label(m, structure=structure)
        if n > 0:
            sizes = ndi.sum(m, lbl, index=np.arange(1, n + 1))
            m = (lbl == (int(np.argmax(sizes)) + 1))

    if close_iters > 0:
        m = ndi.binary_closing(m, structure=structure, iterations=int(close_iters))

    if dilate_iters > 0:
        m = ndi.binary_dilation(m, structure=structure, iterations=int(dilate_iters))

    return m.astype(np.uint8)


# --------------------------
# Target selection
# --------------------------
def collect_targets(subject_dir: Path, files: list[str] | None, patterns: list[str] | None) -> list[Path]:
    targets: list[Path] = []
    if files:
        for name in files:
            p = subject_dir / name
            if p.exists() and p.is_file():
                targets.append(p)
    if patterns:
        for pat in patterns:
            targets.extend(sorted([p for p in subject_dir.glob(pat) if p.is_file()]))

    seen = set()
    uniq = []
    for p in targets:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


# --------------------------
# Commands
# --------------------------
def cmd_make_masks(args):
    in_root = Path(args.in_root)
    mask_root = Path(args.mask_root)

    if args.weights_zip:
        installed_to = install_hdbet_weights_from_zip(Path(args.weights_zip))
        print(f"[OK] HD-BET weights installed to: {installed_to}")

    subjects = list_subject_dirs(in_root)
    print(f"Found {len(subjects)} subject folders in {in_root}")

    for subj in subjects:
        ref = subj / args.ref_name
        if not ref.exists():
            print(f"[SKIP] {subj.name}: missing {args.ref_name}")
            continue

        out_subj = mask_root / subj.name
        out_subj.mkdir(parents=True, exist_ok=True)

        out_mask_path = out_subj / "brain_mask.nii.gz"
        out_brain_path = out_subj / "mask.nii.gz"

        if out_mask_path.exists() and not args.overwrite:
            print(f"[SKIP] {subj.name}: mask already exists")
            continue

        print(f"[HD-BET] {subj.name} on {args.ref_name}")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            tmp_mask_file = td / "hdbet_ref.nii.gz"  # (YOUR build) this is the mask output name

            brain_img, brain_mask = run_hdbet(
                ref_path=ref,
                out_file_nii_gz=tmp_mask_file,
                mode=args.mode,
                device=args.device,
                weights_zip=Path(args.weights_zip) if args.weights_zip else None,
            )

            m, aff, hdr = load_nii(brain_mask)
            m_pp = postprocess_mask(
                m,
                close_iters=args.mask_close,
                dilate_iters=args.mask_dilate,
                keep_largest=not args.no_keep_largest,
            )
            save_nii(out_mask_path, m_pp.astype(np.uint8), aff, hdr, dtype=np.uint8)

            if args.save_brain:
                b, affb, hdrb = load_nii(brain_img)
                save_nii(out_brain_path, b.astype(np.float32), affb, hdrb, dtype=np.float32)

        print(f"  -> {out_mask_path}")

    print("Done making masks.")


def cmd_clean(args):
    in_root = Path(args.in_root)
    mask_root = Path(args.mask_root)
    out_root = Path(args.out_root)

    subjects = list_subject_dirs(in_root)
    print(f"Found {len(subjects)} subject folders in {in_root}")

    for subj in subjects:
        subj_name = subj.name
        mask_path = mask_root / subj_name / "mask.nii.gz"
        if not mask_path.exists():
            print(f"[SKIP] {subj_name}: missing mask {mask_path}")
            continue

        mask, _, _ = load_nii(mask_path)
        mask_bool = (mask > 0.5)

        targets = collect_targets(subj, args.files, args.patterns)
        if not targets:
            print(f"[WARN] {subj_name}: no targets matched")
            continue

        print(f"[SUBJ] {subj_name}: cleaning {len(targets)} file(s)")
        for in_path in targets:
            vol, aff, hdr = load_nii(in_path)
            if vol.shape != mask_bool.shape:
                print(f"  [SKIP] {in_path.name}: shape mismatch {vol.shape} vs mask {mask_bool.shape}")
                continue

            out_vol = vol.copy()
            out_vol[~mask_bool] = float(args.fill)

            rel = in_path.relative_to(in_root)
            out_path = out_root / rel

            if args.suffix:
                if out_path.name.endswith(".nii.gz"):
                    base = out_path.name[:-7]
                    out_path = out_path.with_name(base + args.suffix + ".nii.gz")
                else:
                    out_path = out_path.with_name(out_path.stem + args.suffix + out_path.suffix)

            save_nii(out_path, out_vol.astype(np.float32), aff, hdr, dtype=np.float32)
            print(f"  -> {out_path}")

    print("Done cleaning.")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("make-masks")
    p1.add_argument("--in_root", required=True)
    p1.add_argument("--mask_root", required=True)
    p1.add_argument("--ref_name", default="HR_groundtruth.nii.gz")
    p1.add_argument("--mode", choices=["fast", "accurate"], default="fast")
    p1.add_argument("--device", default=None, help="Use 0/1/... or cpu.")
    p1.add_argument("--overwrite", action="store_true")
    p1.add_argument("--save_brain", action="store_true")
    p1.add_argument("--weights_zip", default=None)

    p1.add_argument("--mask_dilate", type=int, default=0)
    p1.add_argument("--mask_close", type=int, default=0)
    p1.add_argument("--no_keep_largest", action="store_true")

    p2 = sub.add_parser("clean")
    p2.add_argument("--in_root", required=True)
    p2.add_argument("--mask_root", required=True)
    p2.add_argument("--out_root", required=True)
    p2.add_argument("--fill", type=float, default=0.0)
    p2.add_argument("--suffix", default="")

    g = p2.add_mutually_exclusive_group(required=True)
    g.add_argument("--files", nargs="+")
    g.add_argument("--patterns", nargs="+")

    args = p.parse_args()

    if args.cmd == "make-masks":
        cmd_make_masks(args)
    else:
        cmd_clean(args)


if __name__ == "__main__":
    main()
