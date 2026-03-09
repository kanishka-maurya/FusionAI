import sys
import logging

from backend.core.logging import logging
from backend.core.exceptions import CustomException


def test_function():
    try:
        logging.info("Starting test function")

        # Intentional error
        result = 10 / 0

    except Exception as e:
        logging.error("An error occurred")
        raise CustomException(e, sys)


if __name__ == "__main__":
    test_function()

