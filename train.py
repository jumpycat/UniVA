"""Train architecture fingerprints from frozen synthetic reconstructions."""

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import torch
from data import ArchitectureBatches, ImageDataset
from decoder import Decoder
from encoder import Encoder
from losses import balanced_pair_loss, supervised_contrastive_loss
from model import Classifier, FingerprintExtractor
from torch import nn
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, help="Directory of real images.")
    parser.add_argument(
        "--arch-dir", required=True, help="Numbered architecture JSONL files."
    )
    parser.add_argument(
        "--synthesis-checkpoint", required=True, help="Encoder/decoder weights."
    )
    parser.add_argument(
        "--denoiser-checkpoint", required=True, help="Corvi DnCNN weights."
    )
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument(
        "--resume", help="Resume a checkpoint produced by this trainer."
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Images per architecture."
    )
    parser.add_argument(
        "--num-classes", type=int, default=8, help="Architectures per branch."
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--end-lr", type=float, default=3e-6)
    parser.add_argument("--temperature-start", type=float, default=0.02)
    parser.add_argument("--temperature-end", type=float, default=0.1)
    parser.add_argument("--negative-weight", type=float, default=10.0)
    parser.add_argument("--negative-start", type=int, default=3000)
    parser.add_argument("--different-weight", type=float, default=1.0)
    parser.add_argument("--same-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight", type=float, default=1.0)
    parser.add_argument("--classification-weight", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pretrained", action="store_true", help="Skip ImageNet initialization."
    )
    args = parser.parse_args()
    if args.batch_size < 2 or args.num_classes < 2:
        parser.error("Use at least two images per architecture and two architectures.")
    if args.image_size < 64 or args.image_size % 8:
        parser.error("Image size must be a multiple of 8 and at least 64.")
    if (
        min(args.steps, args.feature_dim, args.log_every, args.save_every) < 1
        or args.workers < 0
    ):
        parser.error(
            "Steps, feature size and intervals must be positive; "
            "workers cannot be negative."
        )
    if not 0 < args.end_lr <= args.lr:
        parser.error("Learning rates must satisfy 0 < end-lr <= lr.")
    if min(args.temperature_start, args.temperature_end) <= 0:
        parser.error("Temperatures must be positive.")
    weights = [
        args.different_weight,
        args.same_weight,
        args.pair_weight,
        args.classification_weight,
        args.negative_weight,
    ]
    if min(weights) < 0 or args.different_weight + args.same_weight <= 0:
        parser.error(
            "Weights must be nonnegative and at least one contrastive "
            "branch must be enabled."
        )
    for name in ("synthesis_checkpoint", "denoiser_checkpoint", "resume"):
        value = getattr(args, name)
        if value and not Path(value).is_file():
            parser.error(f"Checkpoint does not exist: {value}")
    return args


def image_batches(loader, device):
    """Repeat the shuffled image loader without swallowing decoding errors."""
    while True:
        for images in loader:
            yield images.to(device, non_blocking=True)


@torch.no_grad()
def synthesize(encoder, decoder, batches, architectures, shared_content):
    """Assign one local class label to each architecture in the current branch."""
    generated, real, labels = [], [], []
    if shared_content:
        images = next(batches)
        latents = encoder(images)
    for label, architecture in enumerate(architectures):
        if not shared_content:
            images = next(batches)
            latents = encoder(images)
        generated.append(decoder(latents, arch_json=architecture))
        if not shared_content or label == 0:
            real.append(images)
        labels.append(
            torch.full((len(images),), label, dtype=torch.long, device=images.device)
        )
    return torch.cat(generated), torch.cat(real), torch.cat(labels)


def branch_loss(
    model,
    pair_head,
    class_head,
    generated,
    real,
    labels,
    temperature,
    negative_weight,
    contrastive_weight,
    args,
):
    features = model(generated)
    contrastive = supervised_contrastive_loss(
        features, labels, temperature, negative_weight
    )
    total = contrastive_weight * contrastive
    metrics = {"contrastive": contrastive.detach().item()}
    if args.pair_weight:
        pair = balanced_pair_loss(pair_head, features, labels)
        total = total + args.pair_weight * pair
        metrics["pair"] = pair.detach().item()
    if args.classification_weight:
        real_logits = class_head(model(real))
        # Shared-content classification uses the first reconstructed batch.
        fake_logits = class_head(features[: len(real)])
        classification = nn.functional.binary_cross_entropy_with_logits(
            real_logits, torch.zeros_like(real_logits)
        ) + nn.functional.binary_cross_entropy_with_logits(
            fake_logits, torch.ones_like(fake_logits)
        )
        total = total + args.classification_weight * classification
        metrics["classification"] = classification.detach().item()
    return total, metrics


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = ImageDataset(args.train_dir, args.image_size)
    if len(dataset) < args.batch_size:
        raise ValueError("The image dataset must contain at least one complete batch.")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    architectures = ArchitectureBatches(args.arch_dir, args.num_classes)
    encoder, decoder = Encoder().to(device), Decoder().to(device)
    synthesis = torch.load(
        args.synthesis_checkpoint, map_location="cpu", weights_only=True
    )
    encoder.load_state_dict(synthesis["encoder"])
    decoder.load_state_dict(synthesis["decoder"])
    encoder.requires_grad_(False).eval()
    decoder.requires_grad_(False).eval()
    del synthesis

    model = (
        FingerprintExtractor(
            args.denoiser_checkpoint,
            args.feature_dim,
            pretrained=not args.no_pretrained and not args.resume,
        )
        .to(device)
        .train()
    )
    pair_head = Classifier(2 * args.feature_dim).to(device)
    class_head = Classifier(args.feature_dim).to(device)
    modules = {
        "model": model,
        "pair_classifier": pair_head,
        "cls_classifier": class_head,
    }
    parameters = [
        p for module in modules.values() for p in module.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args.end_lr / args.lr,
        total_iters=args.steps,
    )
    start_step, architecture_index = 0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        # The schedule and branch layout must stay consistent across a resumed run.
        mutable = {
            "resume",
            "output_dir",
            "device",
            "workers",
            "log_every",
            "save_every",
            "train_dir",
            "arch_dir",
            "synthesis_checkpoint",
            "denoiser_checkpoint",
            "no_pretrained",
        }
        for key, value in checkpoint["config"].items():
            if key not in mutable and getattr(args, key) != value:
                raise ValueError(
                    f"Resume configuration differs for {key}; use {value!r}."
                )
        for name, module in modules.items():
            module.load_state_dict(checkpoint[name])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = checkpoint["step"]
        architecture_index = checkpoint["architecture_index"]
        del checkpoint
    if start_step >= args.steps:
        raise ValueError(
            "The checkpoint has already reached the requested training steps."
        )

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2) + "\n", encoding="utf-8"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "train.log")],
    )
    logging.info(
        "Images: %d | Architecture batches: %d | Device: %s",
        len(dataset),
        len(architectures.paths),
        device,
    )
    batches = image_batches(loader, device)
    for step in range(start_step + 1, args.steps + 1):
        temperature = (
            args.temperature_start
            + (args.temperature_end - args.temperature_start) * step / args.steps
        )
        negative_weight = args.negative_weight if step >= args.negative_start else 0.0
        optimizer.zero_grad(set_to_none=True)
        total_value, metrics = 0.0, {}
        for name, shared_content, weight in (
            ("different", False, args.different_weight),
            ("same", True, args.same_weight),
        ):
            if weight == 0:
                continue
            generated, real, labels = synthesize(
                encoder,
                decoder,
                batches,
                architectures[architecture_index],
                shared_content,
            )
            architecture_index += 1
            loss, branch_metrics = branch_loss(
                model,
                pair_head,
                class_head,
                generated,
                real,
                labels,
                temperature,
                negative_weight,
                weight,
                args,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at step {step} in {name} branch."
                )
            # Accumulate both branches and free each graph before the optimizer step.
            loss.backward()
            total_value += loss.detach().item()
            metrics.update(
                {f"{name}/{key}": value for key, value in branch_metrics.items()}
            )
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            detail = " | ".join(f"{key}={value:.5f}" for key, value in metrics.items())
            logging.info(
                "Step %d/%d | loss=%.5f | %s", step, args.steps, total_value, detail
            )
        if step % args.save_every == 0 or step == args.steps:
            checkpoint = {name: module.state_dict() for name, module in modules.items()}
            checkpoint.update(
                step=step,
                architecture_index=architecture_index,
                config=vars(args),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
            )
            path = run_dir / f"step-{step:07d}.pth"
            torch.save(checkpoint, path)
            logging.info("Saved %s", path)


if __name__ == "__main__":
    main(parse_args())
