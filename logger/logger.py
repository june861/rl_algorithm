import datetime  
import inspect  
import os  
import sys  
  
class Logger:  
    # ANSI color codes  
    RED = '\033[91m'  
    ENDC = '\033[0m'  # Reset to default color  
  
    def __init__(self, log_file=None, std_out_console = False):  
        self.log_file = log_file
        self.console = std_out_console
        if self.log_file:  
            self.file_handler = open(self.log_file, 'a')  
  
    def __del__(self):  
        if hasattr(self, 'file_handler') and self.file_handler:  
            self.file_handler.close()  
  
    def _format_message(self, level, message, frame):  
        now = datetime.datetime.now()  
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')  
        frame_info = inspect.getframeinfo(frame)  
        filename = os.path.basename(frame_info.filename)  
        lineno = frame_info.lineno  
        log_message = f"{timestamp} | {level} | {filename}:{lineno} | {message}"  
        return log_message  
  
    def _print_to_console(self, message, color=None):  
        if color:  
            print(f"{color}{message}{self.ENDC}", file=sys.stdout)  
        else:  
            print(message, file=sys.stdout)  
  
    def info(self, message):  
        log_message = self._format_message('INFO', message, inspect.currentframe().f_back)
        if self.console:  
            self._print_to_console(log_message)  
        if self.log_file:  
            self.file_handler.write(log_message + '\n')  
            self.file_handler.flush()  
  
    def error(self, message):  
        log_message = self._format_message('ERROR', message, inspect.currentframe().f_back)
        if self.console:
            self._print_to_console(log_message, color=self.RED)  
        if self.log_file:  
            self.file_handler.write(log_message + '\n')  
            self.file_handler.flush()  
  
# example  
# if __name__ == "__main__":  
#     logger = Logger('app.log')  
#     logger.info("This is an info log message.")  
#     logger.error("This is an error log message.")  