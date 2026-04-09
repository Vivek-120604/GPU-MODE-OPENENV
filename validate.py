#!/usr/bin/env python
"""
Comprehensive validation script for BiologicalOptimizationEnv

This script validates:
1. Module imports
2. Environment initialization for all three tasks
3. Action execution and reward calculation
4. Deterministic behavior with seeds
5. API server initialization
6. Inference client initialization
"""

import sys
import os

def test_imports():
    """Test all module imports"""
    print("\n" + "="*60)
    print("TEST 1: Module Imports")
    print("="*60)
    
    try:
        from server.models import State, Action, Observation, ResetRequest, StepRequest, StateResponse
        print("✓ Models imported")
        
        from server.environment import BiologicalOptimizationEnv
        print("✓ Environment imported")
        
        from server.app import app
        print("✓ FastAPI app imported")
        
        from inference import BiologicalOptimizationAgent
        print("✓ Inference client imported")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_environment_initialization():
    """Test environment initialization for all tasks"""
    print("\n" + "="*60)
    print("TEST 2: Environment Initialization")
    print("="*60)
    
    from server.environment import BiologicalOptimizationEnv
    
    try:
        for task in ["easy", "medium", "hard"]:
            env = BiologicalOptimizationEnv(seed=42, task=task)
            state = env.reset(seed=42, task=task)
            
            # Validate state structure
            required_keys = ["temperature", "ph", "mutation_level", "performance_score", "steps_count", "stability_count"]
            if all(key in state for key in required_keys):
                print(f"✓ {task.upper()} task initialized successfully")
                print(f"  - Temperature: {state['temperature']:.1f}°C")
                print(f"  - pH: {state['ph']:.2f}")
                print(f"  - Mutation: {state['mutation_level']:.2f}")
            else:
                print(f"✗ {task.upper()} task missing required keys")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Environment initialization failed: {e}")
        return False

def test_action_execution():
    """Test all action types and reward calculation"""
    print("\n" + "="*60)
    print("TEST 3: Action Execution and Rewards")
    print("="*60)
    
    from server.environment import BiologicalOptimizationEnv
    
    try:
        env = BiologicalOptimizationEnv(seed=42, task="easy")
        env.reset(seed=42, task="easy")
        
        actions = [
            ("adjust_temperature", 2.0),
            ("adjust_ph", 0.5),
            ("adjust_mutation", 0.1),
            ("run_experiment", 0.0),
        ]
        
        for action_type, value in actions:
            state, reward, done, info = env.step({"action_type": action_type, "value": value})
            
            # Validate return types
            if not isinstance(reward, (int, float)):
                print(f"✗ Invalid reward type: {type(reward)}")
                return False
            
            if not isinstance(done, bool):
                print(f"✗ Invalid done type: {type(done)}")
                return False
            
            print(f"✓ {action_type:20s} -> reward={reward:7.3f}, done={done}")
        
        return True
    except Exception as e:
        print(f"✗ Action execution failed: {e}")
        return False

def test_determinism():
    """Test deterministic behavior with seeds"""
    print("\n" + "="*60)
    print("TEST 4: Deterministic Behavior (Seed Reproducibility)")
    print("="*60)
    
    from server.environment import BiologicalOptimizationEnv
    
    try:
        # Run twice with same seed
        temperatures_1 = []
        env1 = BiologicalOptimizationEnv(seed=777, task="medium")
        env1.reset(seed=777, task="medium")
        for _ in range(5):
            s, _, _, _ = env1.step({"action_type": "adjust_temperature", "value": 0.5})
            temperatures_1.append(s['temperature'])
        
        temperatures_2 = []
        env2 = BiologicalOptimizationEnv(seed=777, task="medium")
        env2.reset(seed=777, task="medium")
        for _ in range(5):
            s, _, _, _ = env2.step({"action_type": "adjust_temperature", "value": 0.5})
            temperatures_2.append(s['temperature'])
        
        if temperatures_1 == temperatures_2:
            print("✓ Deterministic behavior confirmed:")
            for i, (t1, t2) in enumerate(zip(temperatures_1, temperatures_2)):
                print(f"  Step {i+1}: {t1:.4f} == {t2:.4f}")
            return True
        else:
            print("✗ Non-deterministic behavior detected")
            return False
    
    except Exception as e:
        print(f"✗ Determinism test failed: {e}")
        return False

def test_termination():
    """Test episode termination conditions"""
    print("\n" + "="*60)
    print("TEST 5: Termination Conditions")
    print("="*60)
    
    from server.environment import BiologicalOptimizationEnv
    
    try:
        env = BiologicalOptimizationEnv(seed=42, task="easy")
        env.reset(seed=42, task="easy")
        
        done = False
        steps = 0
        max_episode_steps = 100
        
        while not done and steps < max_episode_steps:
            state, reward, done, info = env.step({"action_type": "adjust_temperature", "value": 0.5})
            steps += 1
        
        if done:
            print(f"✓ Episode terminated after {steps} steps")
            print(f"  - Performance: {state['performance_score']:.3f}")
            print(f"  - Success: {info.get('success', False)}")
            return True
        else:
            print(f"✗ Episode did not terminate after {max_episode_steps} steps")
            return False
    
    except Exception as e:
        print(f"✗ Termination test failed: {e}")
        return False

def test_state_bounds():
    """Test that state values remain within bounds"""
    print("\n" + "="*60)
    print("TEST 6: State Bounds Validation")
    print("="*60)
    
    from server.environment import BiologicalOptimizationEnv
    
    try:
        env = BiologicalOptimizationEnv(seed=42, task="hard")
        env.reset(seed=42, task="hard")
        
        out_of_bounds = False
        for i in range(20):
            actions = [
                {"action_type": "adjust_temperature", "value": 5.0},
                {"action_type": "adjust_ph", "value": 1.0},
                {"action_type": "adjust_mutation", "value": 0.2},
            ]
            state, _, _, _ = env.step(actions[i % 3])
            
            if not (15.0 <= state['temperature'] <= 45.0):
                print(f"✗ Temperature out of bounds: {state['temperature']}")
                out_of_bounds = True
            if not (5.0 <= state['ph'] <= 9.0):
                print(f"✗ pH out of bounds: {state['ph']}")
                out_of_bounds = True
            if not (0.0 <= state['mutation_level'] <= 1.0):
                print(f"✗ Mutation out of bounds: {state['mutation_level']}")
                out_of_bounds = True
        
        if not out_of_bounds:
            print("✓ All state values remained within bounds after aggressive actions")
            return True
        else:
            return False
    
    except Exception as e:
        print(f"✗ State bounds test failed: {e}")
        return False

def test_api_server():
    """Test FastAPI server initialization"""
    print("\n" + "="*60)
    print("TEST 7: FastAPI Server")
    print("="*60)
    
    try:
        from server.app import app
        
        print(f"✓ Server title: {app.title}")
        print(f"✓ Server version: {app.version}")
        
        # Count endpoints
        endpoints = {}
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                path = route.path
                methods = ', '.join(sorted(route.methods))
                if path not in endpoints:
                    endpoints[path] = methods
        
        print(f"✓ Endpoints configured ({len(endpoints)} total):")
        for path, methods in sorted(endpoints.items()):
            print(f"  - {methods:15s} {path}")
        
        return len(endpoints) >= 4  # At least /, /reset, /step, /state
    
    except Exception as e:
        print(f"✗ API server test failed: {e}")
        return False

def test_inference_client():
    """Test inference client initialization"""
    print("\n" + "="*60)
    print("TEST 8: Inference Client")
    print("="*60)
    
    try:
        os.environ['HF_TOKEN'] = 'test_token'
        from inference import BiologicalOptimizationAgent, ENV_BASE_URL, MODEL_NAME, MAX_STEPS
        
        agent = BiologicalOptimizationAgent()
        
        # Verify agent has required methods
        assert hasattr(agent, 'get_action'),        "Missing get_action"
        assert hasattr(agent, 'get_llm_action'),    "Missing get_llm_action"
        assert hasattr(agent, 'get_fallback_action'), "Missing get_fallback_action"
        
        print(f"✓ Agent class: BiologicalOptimizationAgent")
        print(f"✓ ENV_BASE_URL: {ENV_BASE_URL}")
        print(f"✓ MODEL_NAME:   {MODEL_NAME}")
        print(f"✓ MAX_STEPS:    {MAX_STEPS}")
        
        # Smoke-test action generation
        dummy_state = {"temperature": 30.0, "ph": 7.0, "mutation_level": 0.5,
                       "performance_score": 0.4, "steps_count": 1, "stability_count": 0}
        action = agent.get_action(dummy_state, [0.3])
        assert 'action_type' in action and 'value' in action, "Bad action dict"
        print(f"✓ get_action smoke-test passed: {action}")
        
        return True
    
    except Exception as e:
        print(f"✗ Inference client test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("BiologicalOptimizationEnv - Comprehensive Validation")
    print("="*60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Environment Initialization", test_environment_initialization),
        ("Action Execution", test_action_execution),
        ("Deterministic Behavior", test_determinism),
        ("Termination Conditions", test_termination),
        ("State Bounds", test_state_bounds),
        ("FastAPI Server", test_api_server),
        ("Inference Client", test_inference_client),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL VALIDATION TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
