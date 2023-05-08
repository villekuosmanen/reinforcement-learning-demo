# reinforcement-learning-demo
Demo for reinforcement learning

https://user-images.githubusercontent.com/25554034/236901403-c933e950-389c-4de4-b9c3-7801fca9a792.mov

# How to run
```
pipenv run python initial_training.py
pipenv run python run.py
```

# Special features

## Jitter

Use `apply_jitter()` method to add random noise to inputs: simulates random noise often seen in real world, performance still remains high.

## Lag

Commented out code `state_lag` defers updates to state by one, adding lag often seen in real world robotics. Performance remains high.

## Updating a model

Use `update_model.py` script to rerun training for an existing trained model. This can improve performance, though from my experience it does not.
