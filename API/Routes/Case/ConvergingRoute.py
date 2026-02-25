from flask import Blueprint, request, jsonify
import threading
import uuid
import logging
import traceback
from pathlib import Path

from og_clews_integration.converger import ConvergingOrchestrator
from og_clews_integration.schemas import OgCoreInputSchema

converge_api = Blueprint('converge_api', __name__)
logger = logging.getLogger(__name__)

# Simple in-memory task store for the PoC.
# In production, this would have a much better architecture (as per constraints) including a Redis cache or Database table.
TASK_STORE = {}

def background_converge_task(task_id: str, case_name: str, alpha: float, tolerance: float, max_iterations: int):
    """
    Executes the heavy orchestrator in a background thread to prevent Flask blocking.
    Updates the TASK_STORE dictionary with live status.
    """
    try:
        # Initialize the stateful orchestrator
        orchestrator = ConvergingOrchestrator(alpha=alpha, tolerance=tolerance, max_iterations=max_iterations)
        
        # In a full integration, we would load the actual base_demand from the case JSON and trigger the actual ModelRunner.run_clews().
        # For this API skeleton, we update the task store to simulate progress.
        
        TASK_STORE[task_id]['status'] = "Running"
        
        # Simulate the integration loop running --> orchestrator.run_iteration
        
        # On Success:
        TASK_STORE[task_id].update({
            "status": "Success",
            "iterations_run": 7, # Mocked for API structure
            "convergence_delta": 0.00004,
            "output_locations": {
                "clews_results": f"DATA_STORAGE/{case_name}/res/clews_converged.json",
                "ogcore_macro": f"DATA_STORAGE/{case_name}/res/ogcore_macro_table.csv"
            }
        })
        logger.info(f"Task {task_id} completed successfully.")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        logger.debug(traceback.format_exc())
        TASK_STORE[task_id].update({
            "status": "Error",
            "message": str(e)
        })

@converge_api.route("/api/run/converge", methods=['POST'])
def start_converging_run():
    """
    Initiates an asynchronous OG-CLEWS converging loop.
    
    EXPECTED REQUEST PAYLOAD:
    {
      "case_name": "SIDS_Climate_Tax_2026",
      "dampening_factor_alpha": 0.2,
      "convergence_tolerance": 0.0001,
      "max_iterations": 20
    }
    
    EXPECTED RESPONSE PAYLOAD (202 Accepted):
    {
      "status": "Running",
      "task_id": "conv_8f92a1b",
      "message": "Converging loop initiated in background."
    }
    """
    try:
        data = request.json
        case_name = data.get('case_name')
        alpha = data.get('dampening_factor_alpha', 0.2)
        tolerance = data.get('convergence_tolerance', 0.0001)
        max_iterations = data.get('max_iterations', 20)

        if not case_name:
            return jsonify({"error": "case_name is required"}), 400

        task_id = f"conv_{uuid.uuid4().hex[:8]}"
        
        TASK_STORE[task_id] = {
            "status": "Pending",
            "task_id": task_id,
            "case_name": case_name
        }

        # Spawn the background thread so we don't block Flask
        thread = threading.Thread(
            target=background_converge_task, 
            args=(task_id, case_name, alpha, tolerance, max_iterations)
        )
        thread.daemon = True # Thread dies if the main server is shut down
        thread.start()

        return jsonify({
            "status": "Running",
            "task_id": task_id,
            "message": "Converging loop initiated in background."
        }), 202

    except Exception as e:
        return jsonify({"error": "Failed to initiate run", "details": str(e)}), 500

@converge_api.route("/api/run/status/<task_id>", methods=['GET'])
def get_run_status(task_id):
    """
    Polls the current status of a background converging run.
    
    EXPECTED FINAL RESPONSE (200 OK):
    {
      "status": "Success",
      "iterations_run": 7,
      "convergence_delta": 0.00004,
      "output_locations": {
        "clews_results": "DATA_STORAGE/SIDS_Climate_Tax_2026/res/clews_converged.json",
        "ogcore_macro": "DATA_STORAGE/SIDS_Climate_Tax_2026/res/ogcore_macro_table.csv"
      }
    }
    """
    task_info = TASK_STORE.get(task_id)
    if not task_info:
        return jsonify({"error": "Task ID not found"}), 404

    return jsonify(task_info), 200