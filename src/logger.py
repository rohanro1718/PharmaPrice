import logging
import os
from datetime import datetime

# Create log filename using current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Get logs directory path
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)

# Create logs directory if it doesn't exist
os.makedirs(logs_path,exist_ok=True)

# Create complete log file path
LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# if __name__=="__main__":
#     logging.info("Logging has started")