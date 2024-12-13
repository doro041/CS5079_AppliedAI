
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# ...existing code...

if __name__ == '__main__':
    app.run(port=5000)