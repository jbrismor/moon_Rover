---
date: 2024-12-02
author: Kangheng Liu
---
# Reward Structure

## Principle
- Full Coverage
    - moving
        - height
        - dust
    - halt
    - gather
    - sunlight
    - crash
    - stuck

- No Repetition
    No excess rewarding/penelizing
    - delta battery
        handle most scenes. should not be penalizing **unless** drains too much/fills too much
    - gather

- Fast Convergence
    - Sensible reward values
    - Penalize slow movement

## Method
- initial reward: 0
- baseline cost: reward -1 every step
- on gather: 
    - reward for successful gather
    - penalize for unsuccessful gather (empty deposit)
- maintain battery level:
    - handles energy income/cost automatically

