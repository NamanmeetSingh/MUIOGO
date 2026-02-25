import sys
import os
# Force Python to look in the parent directory (API/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pytest
from flask import Flask

# Import the blueprint and the in-memory store we created
from Routes.Case.ConvergingRoute import converge_api, TASK_STORE

@pytest.fixture
def client():
    """Sets up a dummy Flask application for testing the blueprint."""
    app = Flask(__name__)
    app.register_blueprint(converge_api)
    app.testing = True
    
    with app.test_client() as client:
        yield client

def test_start_converging_run_success(client):
    """Proves the endpoint accepts the payload and returns a 202 instantly."""
    payload = {
        "case_name": "Test_SIDS_Case",
        "dampening_factor_alpha": 0.3,
        "convergence_tolerance": 0.0001,
        "max_iterations": 10
    }
    
    response = client.post('/api/run/converge', json=payload)
    
    # Assert we get the non-blocking 202 Accepted status
    assert response.status_code == 202
    
    data = response.get_json()
    assert data['status'] == 'Running'
    assert 'task_id' in data
    
    # Verify it was added to our task store
    assert data['task_id'] in TASK_STORE

def test_start_converging_run_missing_data(client):
    """Proves our API enforces required inputs."""
    payload = {
        "dampening_factor_alpha": 0.3
        # Missing 'case_name'
    }
    
    response = client.post('/api/run/converge', json=payload)
    assert response.status_code == 400
    assert 'error' in response.get_json()

def test_background_thread_execution(client):
    """
    The ultimate proof: Tests that the background thread actually executes 
    and updates the state independently of the main thread.
    """
    payload = {
        "case_name": "Test_Thread_Case",
        "dampening_factor_alpha": 0.2
    }
    
    # 1. Fire the POST request (Main Thread)
    post_response = client.post('/api/run/converge', json=payload)
    task_id = post_response.get_json()['task_id']
    
    # 2. Immediately poll the GET request (Should be Pending/Running if caught fast enough, but since our mock background task is fast, we will just wait a fraction of a second).
    time.sleep(0.1) 
    
    # 3. Poll the GET route to verify the background thread updated the dictionary
    get_response = client.get(f'/api/run/status/{task_id}')
    
    assert get_response.status_code == 200
    data = get_response.get_json()
    
    # If the thread worked, the status will have transitioned to Success in the background
    assert data['status'] == 'Success'
    assert data['iterations_run'] == 7
    assert "clews_converged.json" in data['output_locations']['clews_results']