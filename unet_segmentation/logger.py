import logging


class Logger:  # pragma: no cover
    def __init__(self, name="unet_model_building", log_level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        # Create a console handler and set the level
        ch = logging.StreamHandler()
        ch.setLevel(log_level)

        # Create a formatter that includes custom fields in the log message
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s -"
            + "%(message)s - %(custom_fields)s"
        )

        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    # get_logger is used to get the logger object to
    # pass to the logging middleware
    def get_logger(self):
        return self.logger

    def log(self, level, message, **kwargs):
        """
        Log a message with the specified level and custom fields.

        :param level: Log level (e.g., logging.INFO, logging.ERROR)
        :param message: Log message string
        :param kwargs: Additional fields to log
        """
        if self.logger.isEnabledFor(level):
            # Inject custom fields into the LogRecord's extra parameter
            self.logger._log(level, message, (), extra={"custom_fields": kwargs})

    def debug(self, message, **kwargs):
        self.log(logging.DEBUG, message, **kwargs)

    def info(self, message, **kwargs):
        self.log(logging.INFO, message, **kwargs)

    def warning(self, message, **kwargs):
        self.log(logging.WARNING, message, **kwargs)

    def error(self, message, **kwargs):
        self.log(logging.ERROR, message, **kwargs)

    def exception(self, message, **kwargs):
        self.error(message, **kwargs)

    def critical(self, message, **kwargs):
        self.log(logging.CRITICAL, message, **kwargs)
