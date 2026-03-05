from flask import Blueprint, request, jsonify
from pydantic import ValidationError
import json
from pathlib import Path

from Classes.Base import Config
from Routes.Macro.macro_schemas import MacroBaselineSchema

macro_api = Blueprint('MacroRoute', __name__)

@macro_api.route("/api/macro/upload", methods=['POST'])
def upload_macro_baseline():
    """
    Ingests, validates, and stores baseline macroeconomic data for OG-Core.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Empty or invalid JSON payload."}), 400

        casename = data.get('casename')
        if not casename:
            return jsonify({"error": "casename is required."}), 400

        # 1. Prevent Path Traversal
        base_storage = Path(Config.DATA_STORAGE).resolve()
        case_dir = (base_storage / casename).resolve()
        
        if base_storage not in case_dir.parents:
            return jsonify({"error": "Invalid casename parameter."}), 400

        if not case_dir.is_dir():
            return jsonify({"error": f"Case directory '{casename}' does not exist."}), 404

        # 2. Mathematical Validation (Pydantic)
        macro_parameters = data.get('macro_parameters', {})
        validated_data = MacroBaselineSchema(**macro_parameters)

        # 3. Secure File Write
        macro_file_path = case_dir / "ogcore_baseline.json"
        with open(macro_file_path, "w") as f:
            json.dump(validated_data.model_dump(), f, indent=4)

        return jsonify({
            "status": "Success",
            "message": "Macroeconomic baseline validated and securely stored.",
            "file_path": str(macro_file_path.name)
        }), 201

    except ValidationError as e:
        # Returns exact parameter failures to the frontend UI
        return jsonify({
            "error": "Data Validation Failed",
            "details": e.errors()
        }), 422
    except Exception as ex:
        return jsonify({"error": "Internal Server Error", "details": str(ex)}), 500