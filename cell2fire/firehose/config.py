TRAINING_ENABLED: bool = False
DEBUG_MODE: bool = False


def training_enabled() -> bool:
    return TRAINING_ENABLED


def set_training_enabled(mode: bool):
    global TRAINING_ENABLED
    TRAINING_ENABLED = mode
    print("Training mode set to:", training_enabled())


def debug_mode() -> bool:
    return DEBUG_MODE


def set_debug_mode(mode: bool):
    global DEBUG_MODE
    DEBUG_MODE = mode
    print("Debug mode set to:", debug_mode())
