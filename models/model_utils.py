# --------------------------------------------------------------------------
# model size configurations
SIZE_DICT = {
    "tiny": {"width": 192, "layers": 3, "heads": 4},
    "small": {"width": 512, "layers": 8, "heads": 8},
    "base": {"width": 768, "layers": 12, "heads": 12},
    "large": {"width": 1024, "layers": 24, "heads": 16},
    "xl": {"width": 1152, "layers": 28, "heads": 16},
    "huge": {"width": 1280, "layers": 32, "heads": 16},
}
