#!/usr/bin/env python
"""
API Testing Script for BiologicalOptimizationEnv

This script tests all API endpoints locally without requiring a running server.
It uses FastAPI's TestClient for direct testing.
"""

import json
import sys
from typing import Dict, Any


def test_api():
    """Test all API endpoints using TestClient"""
    print("\n" + "="*60)
    print("BiologicalOptimizationEnv - API Testing")
    print("="*60 + "\n")
    
    try:
        from fastapi.testclient import TestClient
        from server.app import app
        
        client = TestClient(app)
        
        # Test 1: Health check
        print("[TEST 1] GET / - Health Check")
        response = client.get("/")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data["status"] == "ok", "Status is not 'ok'"
        print(f"✓ Health check passed")
        print(f"  Response: {json.dumps(data, indent=2)}\n")
        
        # Test 2: Reset environment (easy task)
        print("[TEST 2] POST /reset - Reset Environment (Easy)")
        response = client.post("/reset", json={"task": "easy", "seed": 42})
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "state" in data, "Response missing 'state' key"
        assert "episode_info" in data, "Response missing 'episode_info' key"
        state = data["state"]
        
        print(f"✓ Reset successful (easy task)")
        print(f"  Temperature: {state['temperature']:.1f}°C")
        print(f"  pH: {state['ph']:.2f}")
        print(f"  Mutation: {state['mutation_level']:.2f}\n")
        
        # Test 3: Get state
        print("[TEST 3] GET /state - Get Current State")
        response = client.get("/state")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        print(f"✓ State retrieved")
        print(f"  Performance: {data['state']['performance_score']:.2f}")
        print(f"  Steps: {data['state']['steps_count']}\n")
        
        # Test 4: Step with adjust_temperature
        print("[TEST 4] POST /step - Adjust Temperature")
        response = client.post("/step", json={
            "action": {
                "action_type": "adjust_temperature",
                "value": 2.0
            }
        })
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "state" in data, "Response missing 'state'"
        assert "reward" in data, "Response missing 'reward'"
        assert "done" in data, "Response missing 'done'"
        assert "info" in data, "Response missing 'info'"
        
        print(f"✓ Step executed (adjust_temperature)")
        print(f"  New temperature: {data['state']['temperature']:.1f}°C")
        print(f"  Reward: {data['reward']:.3f}")
        print(f"  Done: {data['done']}\n")
        
        # Test 5: Step with adjust_ph
        print("[TEST 5] POST /step - Adjust pH")
        response = client.post("/step", json={
            "action": {
                "action_type": "adjust_ph",
                "value": 0.3
            }
        })
        assert response.status_code == 200
        data = response.json()
        print(f"✓ Step executed (adjust_ph)")
        print(f"  New pH: {data['state']['ph']:.2f}")
        print(f"  Reward: {data['reward']:.3f}\n")
        
        # Test 6: Step with adjust_mutation
        print("[TEST 6] POST /step - Adjust Mutation")
        response = client.post("/step", json={
            "action": {
                "action_type": "adjust_mutation",
                "value": 0.05
            }
        })
        assert response.status_code == 200
        data = response.json()
        print(f"✓ Step executed (adjust_mutation)")
        print(f"  New mutation: {data['state']['mutation_level']:.2f}")
        print(f"  Reward: {data['reward']:.3f}\n")
        
        # Test 7: Step with run_experiment
        print("[TEST 7] POST /step - Run Experiment")
        response = client.post("/step", json={
            "action": {
                "action_type": "run_experiment",
                "value": 0.0
            }
        })
        assert response.status_code == 200
        data = response.json()
        print(f"✓ Step executed (run_experiment)")
        print(f"  Performance after experiment: {data['state']['performance_score']:.3f}")
        print(f"  Reward: {data['reward']:.3f}\n")
        
        # Test 8: Reset environment (medium task, random seed)
        print("[TEST 8] POST /reset - Reset Environment (Medium, Random)")
        response = client.post("/reset", json={"task": "medium", "seed": 123})
        assert response.status_code == 200
        data = response.json()
        print(f"✓ Reset successful (medium task, seed=123)")
        state = data["state"]
        print(f"  Random temperature: {state['temperature']:.1f}°C")
        print(f"  Random pH: {state['ph']:.2f}")
        print(f"  Random mutation: {state['mutation_level']:.2f}\n")
        
        # Test 9: Reset environment (hard task)
        print("[TEST 9] POST /reset - Reset Environment (Hard)")
        response = client.post("/reset", json={"task": "hard"})
        assert response.status_code == 200
        data = response.json()
        print(f"✓ Reset successful (hard task)")
        state = data["state"]
        print(f"  Hard task temperature: {state['temperature']:.1f}°C (far from optimal)")
        print(f"  Hard task pH: {state['ph']:.2f} (far from optimal)")
        print(f"  Hard task mutation: {state['mutation_level']:.2f} (far from optimal)\n")
        
        # Test 10: Deterministic behavior
        print("[TEST 10] Deterministic Behavior Test")
        response1 = client.post("/reset", json={"task": "easy", "seed": 42})
        temps1 = [response1.json()["state"]["temperature"]]
        
        for _ in range(3):
            response = client.post("/step", json={
                "action": {"action_type": "adjust_temperature", "value": 1.0}
            })
            temps1.append(response.json()["state"]["temperature"])
        
        response2 = client.post("/reset", json={"task": "easy", "seed": 42})
        temps2 = [response2.json()["state"]["temperature"]]
        
        for _ in range(3):
            response = client.post("/step", json={
                "action": {"action_type": "adjust_temperature", "value": 1.0}
            })
            temps2.append(response.json()["state"]["temperature"])
        
        if temps1 == temps2:
            print(f"✓ Deterministic behavior confirmed")
            for i, (t1, t2) in enumerate(zip(temps1, temps2)):
                print(f"  Step {i}: {t1:.4f} == {t2:.4f}")
        else:
            print(f"✗ Non-deterministic behavior detected")
            return False
        print()
        
        # Test 11: Error handling - invalid task
        print("[TEST 11] Error Handling - Invalid Task")
        response = client.post("/reset", json={"task": "invalid"})
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        print(f"✓ Invalid task correctly rejected (status: {response.status_code})")
        print(f"  Error: {response.json()['detail']}\n")
        
        # Test 12: Error handling - state without initialization
        print("[TEST 12] Error Handling - GET State When Needed")
        # Note: Due to global state, environment is initialized from previous tests
        # A fresh server instance would need state before /state works
        response = client.get("/state")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        print(f"✓ State endpoint works after previous tests\n")
        
        # Summary
        print("="*60)
        print("✓ ALL API TESTS PASSED!")
        print("="*60)
        return True
    
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_api()
    sys.exit(0 if success else 1)
