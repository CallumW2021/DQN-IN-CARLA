import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
import tensorflow.keras.backend as backend
from keras.models import load_model
from RLAgent import Environment


MODEL_PATH = 'models/Xception____-1.50max___-1.80avg___-3.00min__1722197988.model'

if __name__ == '__main__':

    # Memory fraction
       # Load the model
    model = load_model(MODEL_PATH)

    # Create environment
    env = Environment()

    # For speed measurements
    fps_counter = deque(maxlen=60)
    model.predict(np.ones((1, env.im_height, env.im_width, 3)))

    # Loop episodes
    while True:

        print('Restarting episode')

        # Reset environment
        current_state = env.reset()
        env.collision_hist = []

        done = False

        # Loop steps
        while True:

            #FPS counter
            step_start = time.time()

            #Show frame
            cv2.imshow(f'Agent - preview', current_state)
            cv2.waitKey(1)

            # Predict an action based on observation space
            qs = model.predict(np.array(current_state).reshape(-1, *current_state.shape)/255)[0]
            action = np.argmax(qs)

            # Step environment
            new_state, reward, done, _ = env.step(action)

            # Set current step 
            current_state = new_state

            # If done agent crashed
            if done:
                break

            # Measure step time, print q values and action chosen
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}, {qs[3]:>5.2f}, {qs[4]:>5.2f}] {action}')

        # Destroy an actor at end of episode
        for actor in env.actor_list:
            actor.destroy()
