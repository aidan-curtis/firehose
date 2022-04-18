TRAINING_ENABLED: bool = False


def training_enabled() -> bool:
    return TRAINING_ENABLED


def set_training_enabled(mode: bool):
    global TRAINING_ENABLED
    TRAINING_ENABLED = mode
