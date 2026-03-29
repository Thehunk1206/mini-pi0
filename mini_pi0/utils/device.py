import torch


def _has_mps():
    """Check whether torch Metal (MPS) backend is available."""

    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_auto_device():
    """Resolve best available torch device in priority order.

    Returns:
        ``cuda`` if available, else ``mps`` if available, else ``cpu``.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _has_mps():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device="auto"):
    """Resolve and validate user-requested torch device.

    Args:
        device: Device request string or ``torch.device`` instance.

    Returns:
        Validated ``torch.device``.

    Raises:
        RuntimeError: If requested accelerator backend is unavailable.
    """

    if isinstance(device, torch.device):
        requested = device
    else:
        device = "auto" if device is None else str(device).strip().lower()
        requested = get_auto_device() if device == "auto" else torch.device(device)

    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine.")
    if requested.type == "mps" and not _has_mps():
        raise RuntimeError("MPS requested but not available. Use --device auto or --device cpu.")
    return requested
