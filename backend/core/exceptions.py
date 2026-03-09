import sys

class CustomException(Exception):

    def __init__(self, error, error_details: sys):
        self.error_message = self._error_message_details(error, error_details)
        super().__init__(self.error_message)

    def _error_message_details(self, error, error_details: sys):
        _, _, exc_tb = error_details.exc_info()

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return (
            f"Error occurred in python file: {file_name}, "
            f"line number: {line_number}, error message: {str(error)}"
        )

    def __str__(self):
        return self.error_message

