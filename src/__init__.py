import logging
import click
from rich.logging import RichHandler
from dotenv import load_dotenv


#### load env variables

load_dotenv()

#### Define logger

LOGGER_NAME = "document-qa"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, tracebacks_suppress=[click])],
)
logger = logging.getLogger(LOGGER_NAME)
logging.getLogger("numexpr").setLevel(logging.ERROR)
