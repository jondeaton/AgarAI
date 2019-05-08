# AgarAI

to run DQN training

    python -m dqn.train
        
to start tensorboard during training

    tensorboard --logdir model_outputs/dqn
      
      
# todo
- Make the environment more flexible
  - configurable:
    - arena dimensionss
    - number of food
    - number of viruses
    - number of bots of each type
- Report information from the environment about what's happening (to tensorboard)
- Refactor the code for converting game state into features (python and C++)
- Fix the experience replay buffer
- Try a very simple environment, make sure its doing expeted things
- come up with better values to represent things that are absent

