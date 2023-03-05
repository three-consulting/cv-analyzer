from loguru import logger


def main():
    logger.add("../app.log", format="{time} {level} {message}", level="INFO")


if __name__ == "__main__":
    main()
