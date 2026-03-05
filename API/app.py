from pathlib import Path
import logging
import os
import secrets

from flask import Flask, jsonify, request, session, render_template
from flask_cors import CORS
from datetime import timedelta

from Classes.Base import Config
from Routes.Upload.UploadRoute import upload_api
from Routes.Case.CaseRoute import case_api
from Routes.Case.SyncS3Route import syncs3_api
from Routes.Case.ViewDataRoute import viewdata_api
from Routes.DataFile.DataFileRoute import datafile_api

from Routes.Macro.MacroRoute import macro_api

# -------------------------
# FIX: Make template/static paths independent of cwd
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
WEBAPP_PATH = BASE_DIR / "WebAPP"

template_dir = str(WEBAPP_PATH)
static_dir = str(WEBAPP_PATH)

app = Flask(__name__, static_url_path='', static_folder=static_dir, template_folder=template_dir)

app.permanent_session_lifetime = timedelta(days=5)
secret_key = os.environ.get("MUIOGO_SECRET_KEY", "").strip()
if not secret_key:
    secret_key = secrets.token_hex(32)
    logging.warning(
        "MUIOGO_SECRET_KEY is not configured. Using a temporary in-memory key. "
        "Run setup to create a persistent secret in .env."
    )
app.config['SECRET_KEY'] = secret_key
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config["MAX_CONTENT_LENGTH"] = None

# Registering Blueprints
app.register_blueprint(upload_api)
app.register_blueprint(case_api)
app.register_blueprint(viewdata_api)
app.register_blueprint(datafile_api)
app.register_blueprint(syncs3_api)

# Registering Track 1 Blueprint
app.register_blueprint(macro_api)

CORS(app)

@app.after_request
def add_headers(response):
    if Config.HEROKU_DEPLOY == 0: 
        # localhost
        response.headers.add('Access-Control-Allow-Origin', 'http://127.0.0.1')
    else:
        # HEROKU
        response.headers.add('Access-Control-Allow-Origin', 'https://osemosys.herokuapp.com/')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    return response

# Entry point to frontend
@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')


@app.route("/getSession", methods=['GET'])
def getSession():
    try:
        ses = session.get('osycase', None) or None
        response = {
            "session": ses
        }
        return jsonify(response), 200
    except KeyError:
        return jsonify('No selected parameters!'), 404

@app.route("/setSession", methods=['POST'])
def setSession():
    try:
        cs = request.json['case']
        if cs is None:
            session.pop('osycase', None)
            return jsonify({"osycase": None}), 200

        from pathlib import Path
        if not Path(Config.DATA_STORAGE, cs).is_dir():
            return jsonify({'message': 'Case not found.', 'status_code': 'error'}), 404
        session['osycase'] = cs
        response = {"osycase": session['osycase']}
        return jsonify(response), 200
    except KeyError:
        return jsonify('No selected parameters!'), 404


if __name__ == '__main__':
    import mimetypes
    mimetypes.add_type('application/javascript', '.js')
    port = int(os.environ.get("PORT", 5002))

    def print_startup_info(host, current_port, server_name):
        mode = 'local' if Config.HEROKU_DEPLOY == 0 else 'heroku'
        access_host = '127.0.0.1' if host == '0.0.0.0' else host
        print("MUIOGO API starting...")
        print(f"Server: {server_name}")
        print(f"Mode: {mode}")
        print(f"Host: {host}")
        print(f"Port: {current_port}")
        print(f"Open: http://{access_host}:{current_port}")

    if Config.HEROKU_DEPLOY == 0: 
        from waitress import serve
        host = '127.0.0.1'
        print_startup_info(host, port, 'waitress')
        serve(app, host=host, port=port)
    else:
        host = '0.0.0.0'
        print_startup_info(host, port, 'flask-dev')
        app.run(host=host, port=port, debug=True)