import os
from dotenv import load_dotenv

load_dotenv()

# APIs
HDX_HAPI_BASE_URL  = "https://hapi.humdata.org/api/v1"
FEWSNET_BASE_URL   = "https://fdw.fews.net/api"
FEWSNET_TOKEN      = os.getenv("FEWSNET_TOKEN", "")
IPC_API_KEY        = os.getenv("IPC_API_KEY", "")
HDX_APP_IDENTIFIER = os.getenv("HDX_APP_IDENTIFIER", "")

# Geography
COUNTRY_CODE = "MDG"
COUNTRY_NAME = "Madagascar"

# IPC phases
IPC_PHASES = {
    1: "Minimal",
    2: "Stressed",
    3: "Crisis",
    4: "Emergency",
    5: "Catastrophe"
}

# Paths
DATA_RAW       = "data/raw"
DATA_PROCESSED = "data/processed"
DATA_EXTERNAL  = "data/external"
MODELS_DIR     = "models"
