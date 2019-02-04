from app import app
from flask import jsonify


@app.route('/', methods=["GET"])
def hello():
    return jsonify({
        'ena': 'dyo'
    })
