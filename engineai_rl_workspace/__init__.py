import os
from engineai_rl_lib.text import Color

ENGINEAI_WORKSPACE_ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)
ENGINEAI_WORKSPACE_PACKAGE_DIR = os.path.join(
    ENGINEAI_WORKSPACE_ROOT_DIR, "engineai_rl_workspace"
)

# Redis configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
LOCK_KEY = "resume_lock"
LOCK_TIMEOUT = 180  # seconds
LOCK_MESSAGE = (
    "Previous run is resuming, waiting...\n"
    "If last run is complete and this message is still showing, it means the last run was interrupted.\n"
    "This may cause git checked out another commit, cfg files are overwritten by resume cfg files or stashed files have not been unstashed.\n"
    'Checkout original commit and unstash files, if needed, then run "redis-cli del resume_lock" to release lock.'
)
PROGRAM_START_MESSAGE = Color.apply(
    "Program is initializing! Please don't change code or config before initialization completed!",
    Color.MAGENTA,
)
INITIALIZATION_COMPLETE_MESSAGE = Color.apply("Initialization completed!", Color.GREEN)
FAIL_TO_LOAD_JSON_MESSAGE = Color.apply(
    "Fail to generate resume files from json, exit.", Color.RED
)
