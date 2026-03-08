import logging
import sys

def setup_logger():
    """Настраивает корневой логгер для вывода в консоль и файл."""
    logger = logging.getLogger()  # корневой логгер
    logger.setLevel(logging.INFO)

    # Проверяем, не добавлены ли уже обработчики (чтобы избежать дублирования)
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler("qa_interview.log")
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger