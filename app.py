import flask
import requests as req
from flask import redirect, url_for, request, jsonify, render_template
import json
from model import transliterate as model_predict
# from deeptranslit import DeepTranslit

app = flask.Flask(__name__)
app.config["DEBUG"] = True

api = "https://www.cfilt.iitb.ac.in/indicnlpweb/indicnlpws/transliterate_bulk/{}/{}/{}/rule"
# to_marathi = DeepTranslit("marathi")

@app.route('/')
def home():
   return render_template("index.html")

@app.route('/transliterate', methods=['POST'])
def transliterate():
   print("Transliterating...")
   post_data = request.form
   return jsonify({"input": post_data['input'], 'transliteration': model_predict(post_data['input'])})

app.run(debug = True)