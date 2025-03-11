import logging
from datetime import datetime
import os

Log_folder = os.path.join(os.getcwd(), "logs")
os.makedirs(Log_folder, exist_ok=True)

log_file = f"{datetime.now().strftime('%d-%m-%Y %H_%M_%S')}.log"
log_file_path = os.path.join(Log_folder, log_file)

logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s - %(message)s",
    level=logging.DEBUG,
    filemode="a"
)

## Testing Logger file
logging.info("Testing Logger file")
