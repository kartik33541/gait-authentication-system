import sys
import os

# add production folder to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROD_PATH = os.path.join(BASE_DIR, "production")

sys.path.append(PROD_PATH)

# import flask app from your server
from app.flask_server import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)