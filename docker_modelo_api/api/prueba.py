import json
from os import environ
from flask import Flask, request, jsonify
app=Flask(__name__)

records = [{"name" : "DASP", "email" : "trabajosdedanielalejandro@gmail.com"}]

@app.route('/')
def index():
    return jsonify({'message':"Hello world :)"})
@app.route('/user/all', methods=['GET'])
def all_records():
    return jsonify(records)
@app.route('/user', methods=['GET'])
def query_records():
    name=request.args.get['name']
    for record in records:
        if record['name'] == name:
            return jsonify(record)
    return jsonify({'error':'data not found'})


@app.route('/user', methods=['PUT'])
def create_record():
    record=json.loads(request.data)
    records.append(record)
    return jsonify(record)