import logging
from config import LOG_FILE

logger = logging.getLogger("netflix_app")
logger.setLevel(logging.INFO)

if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)


def log_action(user_id, action, result=""):
    logger.info(f"user_id={user_id} | action={action} | result={result}")