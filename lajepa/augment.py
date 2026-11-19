"""Observation-only augmentation.

The 0.3 bypass leaves the whole window untouched. Otherwise photometric,
corruption, and rotation gates are independent, and the sampled parameters
are shared by every frame. Only the primary camera receives corruption and
rotation. Future targets are never passed through this module.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class ObservationAugmentor:
    def __init__(
        self,
        bypass_prob: float = 0.3,
        photometric_prob: float = 0.6,
        corruption_prob: float = 0.5,
        rotation_prob: float = 0.5,
        vertical_fov_deg: float = 45.0,
        primary_camera: int = 0,
    ):
        self.bypass_prob = bypass_prob
        self.photometric_prob = photometric_prob
        self.corruption_prob = corruption_prob
        self.rotation_prob = rotation_prob
        self.vertical_fov_deg = vertical_fov_deg
        self.primary_camera = primary_camera

    def __call__(self, images: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """``images`` is ``[cameras, time, 3, H, W]`` in ``[0, 1]``."""
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"expected [cameras, time, 3, H, W], got {tuple(images.shape)}")
        if _uniform(generator, images.device) < self.bypass_prob:
            return images
        output = images.clone()
        if _uniform(generator, images.device) < self.photometric_prob:
            output = _photometric(output, generator)
        if _uniform(generator, images.device) < self.corruption_prob:
            output = _corrupt_primary(output, self.primary_camera, generator)
        if _uniform(generator, images.device) < self.rotation_prob:
            output = _rotate_primary(output, self.primary_camera, self.vertical_fov_deg, generator)
        return output.clamp(0.0, 1.0)


def _uniform(generator: torch.Generator | None, device: torch.device) -> float:
    return float(torch.rand((), generator=generator).item())


def _rand(generator, *shape) -> torch.Tensor:
    # A missing shape is a scalar draw. torch.rand requires an explicit size.
    size = shape if shape else ((),)
    return torch.rand(*size, generator=generator)


def _randint(generator, low: int, high: int) -> int:
    return int(torch.randint(low, high, (), generator=generator).item())


def _photometric(images: torch.Tensor, generator) -> torch.Tensor:
    brightness = 1.0 + (_rand(generator).item() * 0.6 - 0.3)
    contrast = 1.0 + (_rand(generator).item() * 0.6 - 0.3)
    saturation = 1.0 + (_rand(generator).item() * 0.6 - 0.3)
    hue = _rand(generator).item() * 0.06 - 0.03
    gamma = 0.8 + _rand(generator).item() * 0.45
    x = images * brightness
    mean = x.mean(dim=2, keepdim=True)
    x = (x - mean) * contrast + mean
    weights = x.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 1, 3, 1, 1)
    gray = (x * weights).sum(dim=2, keepdim=True)
    x = gray + saturation * (x - gray)
    x = _shift_hue(x.clamp(0.0, 1.0), hue)
    x = x.clamp(0.0, 1.0).pow(gamma)
    return x


def _shift_hue(images: torch.Tensor, hue: float) -> torch.Tensor:
    if hue == 0.0:
        return images
    red, green, blue = images[:, :, 0], images[:, :, 1], images[:, :, 2]
    maxc, _ = images.max(dim=2)
    minc, _ = images.min(dim=2)
    delta = (maxc - minc).clamp_min(1e-8)
    hue_channel = torch.zeros_like(maxc)
    mask = maxc == red
    hue_channel = torch.where(mask, ((green - blue) / delta) % 6, hue_channel)
    mask = maxc == green
    hue_channel = torch.where(mask, (blue - red) / delta + 2, hue_channel)
    mask = maxc == blue
    hue_channel = torch.where(mask, (red - green) / delta + 4, hue_channel)
    hue_channel = (hue_channel / 6.0 + hue) % 1.0
    hue_channel = hue_channel * 6.0
    sector = hue_channel.floor()
    frac = hue_channel - sector
    p = minc
    q = minc + delta * (1 - frac)  # not used in the compact form below
    # Standard HSV sector interpolation.
    c = delta
    x = c * (1 - (hue_channel % 2 - 1).abs())
    zeros = torch.zeros_like(c)
    sector = sector.long() % 6
    rgb = torch.stack([zeros, zeros, zeros], dim=2)
    choices = {
        0: (c, x, zeros),
        1: (x, c, zeros),
        2: (zeros, c, x),
        3: (zeros, x, c),
        4: (x, zeros, c),
        5: (c, zeros, x),
    }
    for index, (r, g, b) in choices.items():
        selected = sector == index
        rgb[:, :, 0] = torch.where(selected, r, rgb[:, :, 0])
        rgb[:, :, 1] = torch.where(selected, g, rgb[:, :, 1])
        rgb[:, :, 2] = torch.where(selected, b, rgb[:, :, 2])
    return (rgb + minc.unsqueeze(2)).clamp(0.0, 1.0)


def _corrupt_primary(images: torch.Tensor, camera: int, generator) -> torch.Tensor:
    choice = _randint(generator, 0, 3)
    frames = images[camera]
    if choice == 0:
        sigma = 0.5 + _rand(generator).item() * 5.5
        frames = _gaussian_blur(frames, sigma)
    elif choice == 1:
        length = _randint(generator, 5, 21)
        angle = _rand(generator).item() * math.pi
        frames = _motion_blur(frames, length, angle)
    else:
        frames = _fog(frames, generator)
    images = images.clone()
    images[camera] = frames.clamp(0.0, 1.0)
    return images


def _gaussian_kernel(sigma: float, device) -> torch.Tensor:
    radius = max(1, int(math.ceil(sigma * 3)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_blur(frames: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = _gaussian_kernel(sigma, frames.device)
    return _separable_blur(frames, kernel)


def _separable_blur(frames: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    time, channels, height, width = frames.shape
    flat = frames.reshape(time * channels, 1, height, width)
    k = kernel.numel()
    horiz = kernel.view(1, 1, 1, k)
    vert = kernel.view(1, 1, k, 1)
    flat = F.conv2d(flat, horiz, padding=(0, k // 2))
    flat = F.conv2d(flat, vert, padding=(k // 2, 0))
    return flat.reshape(time, channels, height, width)


def _motion_blur(frames: torch.Tensor, length: int, angle: float) -> torch.Tensor:
    kernel = torch.zeros(length, length, device=frames.device)
    center = length // 2
    for step in range(length):
        offset = step - center
        y = int(round(center + math.sin(angle) * offset))
        x = int(round(center + math.cos(angle) * offset))
        if 0 <= y < length and 0 <= x < length:
            kernel[y, x] = 1.0
    if float(kernel.sum()) == 0.0:
        kernel[center, center] = 1.0
    kernel = kernel / kernel.sum()
    time, channels, height, width = frames.shape
    flat = frames.reshape(time * channels, 1, height, width)
    weight = kernel.view(1, 1, length, length)
    flat = F.conv2d(flat, weight, padding=length // 2)
    extra_h = flat.shape[-2] - height
    extra_w = flat.shape[-1] - width
    if extra_h or extra_w:
        top = extra_h // 2
        left = extra_w // 2
        flat = flat[..., top : top + height, left : left + width]
    return flat.reshape(time, channels, height, width)


def _fog(frames: torch.Tensor, generator) -> torch.Tensor:
    grid = _randint(generator, 4, 9)
    opacity = 0.2 + _rand(generator).item() * 0.4
    contrast = 0.5 + _rand(generator).item() * 0.4
    gray = 0.5 + _rand(generator).item() * 0.5
    _, _, height, width = frames.shape
    noise = torch.randn(1, 1, grid, grid, generator=generator)
    noise = F.interpolate(noise, size=(height, width), mode="bilinear", align_corners=False)
    noise = noise - noise.min()
    noise = noise / noise.max().clamp_min(1e-6)
    x = (frames - 0.5) * contrast + 0.5
    mask = opacity * noise
    return x * (1 - mask) + gray * mask


def _rotate_primary(images: torch.Tensor, camera: int, vfov_deg: float, generator) -> torch.Tensor:
    yaw = math.radians(_rand(generator).item() * 8 - 4)
    pitch = math.radians(_rand(generator).item() * 8 - 4)
    roll = math.radians(_rand(generator).item() * 6 - 3)
    frames = images[camera]
    _, _, height, width = frames.shape
    homography = _rotation_homography(height, width, vfov_deg, yaw, pitch, roll)
    warped = torch.stack([_warp(frame, homography) for frame in frames], dim=0)
    images = images.clone()
    images[camera] = warped
    return images


def _rotation_homography(height, width, vfov_deg, yaw, pitch, roll) -> torch.Tensor:
    fy = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    fx = fy
    intrinsics = torch.tensor(
        [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    rotation = _euler(yaw, pitch, roll)
    homography = intrinsics @ rotation @ torch.linalg.inv(intrinsics)
    return torch.linalg.inv(homography)


def _euler(yaw: float, pitch: float, roll: float) -> torch.Tensor:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rz = torch.tensor([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _warp(frame: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
    channels, height, width = frame.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    pixels = torch.stack([xs, ys, ones], dim=0).reshape(3, -1)
    source = homography @ pixels
    source = source[:2] / source[2:3].clamp_min(1e-6)
    grid_x = source[0] / max(width - 1, 1) * 2 - 1
    grid_y = source[1] / max(height - 1, 1) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, height, width, 2)
    warped = F.grid_sample(frame.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True)
    return warped[0]
