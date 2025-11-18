# app/routes/__init__.py
# make routes package discoverable
from .predictions import router as predictions_router
from .data_upload import router as data_upload_router
