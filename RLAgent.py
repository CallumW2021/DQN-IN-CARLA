#Sources

#https://stackoverflow.com/questions/59448001/how-to-get-q-values-in-rl-ddqn
#https://stackoverflow.com/questions/58711624/modifying-tensorboard-in-tensorflow-2-0
#https://stackoverflow.com/questions/63515917/attributeerror-modifiedtensorboard-object-has-no-attribute-train-dir
#A lot of code adpated from https://pythonprogramming.net self-driving carla
#https://github.com/carla-simulator/carla/issues/1466


import carla
import random
import time
import numpy as np
import cv2
import glob
import os
import sys
import math
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
from threading import Thread
from keras.callbacks import TensorBoard
from tqdm import tqdm

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
                              sys.version_info.major,
                              sys.version_info.minor,
                              'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

#Defined Parameters for the model at top of code

showPreview = False
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
secondsPerEpisode = 10
replayMemorySize = 5000
minReplayMemorySize = 1000
miniBatchSize = 16
predictionBatch = 1
trainingBatch = miniBatchSize // 4
updateEvery = 5
modelName = "Xception"

minReward = -5

#Episodes to train the model

EPISODES = 3000

#Discount factor to determine weight towards future rewards

DISCOUNT = 0.99

#Epsilon defines exploration rate

epsilon = 1
EPSILON_DECAY = 0.9993
minEpsilon = 0.001

#Write stats to tensorboard per 10 episodes

statsEvery = 10
num_of_vehicles = 10

# Enable eager execution
tf.config.experimental_run_functions_eagerly(True)

class Environment:

    #Show camera preview of the agent in a window, so we can see what is going on
    
    SHOW_CAM = showPreview
    STEER_AMT = 1.0
    im_width = IMAGE_WIDTH
    im_height = IMAGE_HEIGHT
    front_camera = None

    def __init__(self):
        #Connect to Carla, and set up environment
        self.client = carla.Client("localhost",2000)
        self.client.set_timeout(6.0)
        self.world = self.client.get_world()
        self.client.load_world('Town02')
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.vehicle_blueprints = self.blueprint_library.filter("vehicle.*")
        self.spawn_points = self.world.get_map().get_spawn_points()

        for _ in range(num_of_vehicles):
            self.vehicle_bp = random.choice(self.vehicle_blueprints)
            self.spawn_point = random.choice(self.spawn_points)
            self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.spawn_point)
            self.vehicle.set_autopilot(True)       
            print(f"Spawned {self.vehicle.type_id} at {self.spawn_point.location}")
            time.sleep(1)

        time.sleep(1)
            

    def reset(self):
        #On epsiode end, reset agent and spawn again
        spawned = False
        self.collision_hist = []
        self.lane_hist = []
        self.actor_list = []

        while(spawned == False):

            try:
                self.transform = random.choice(self.world.get_map().get_spawn_points())
                self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
                spawned = True

            except:
                print("Spawn Collision")

        self.actor_list.append(self.vehicle)

        #Append the carla RGB camera to the agent

        self.rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        #Transform camera position so it is positioned more optimally

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        #Sensor to detect lane-crossing

        lanesensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lanesensor = self.world.spawn_actor(lanesensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.lanesensor)
        self.lanesensor.listen(lambda event: self.lane_data(event))

        #Detect collisions

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    #Functions for appending sensor data to lists

    def collision_data(self, event):
        self.collision_hist.append(event)

    def lane_data(self, event):
        self.lane_hist.append(event)

    #Reshaping image to be fed into CNN

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    #Discritization of action space, the 5 possible moves the agent can take in any step.

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle =0, steer=0))
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0, steer=0))

        v = self.vehicle.get_velocity()
        #Convert velocity into Kmh for reward function
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        #Reward Function Punishments and Rewards

        if len(self.collision_hist) != 0:
            done = True
            reward = -10
        elif len(self.lane_hist) != 0:
             done = False
             reward = -0.1
        elif kmh < 50:
            done = False
            reward = -0.2
        else:
            done = False
            reward = 0.5

        if self.episode_start + secondsPerEpisode < time.time():
            done = True

        return self.front_camera, reward, done, None

class QNetwork:

    #Initialize model with weights
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=replayMemorySize)

        #Create tensorboard based on model name & timestamp, to log and save metrics.

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{modelName}-{int(time.time())}")

        self.target_update_counter = 0

        self.terminate = False
        self.last_logged_episode = 0

        self.training_initialized = False

        #Create model using pre-made CNN Xception

    def create_model(self):
        base_model = Xception(weights=None, include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        #5 Activations for the 5 different discretized actions

        predictions = Dense(5, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
        model.run_eagerly = True
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

        #Train the model

    def train(self):
        if len(self.replay_memory) < minReplayMemorySize:
            return

        minibatch = random.sample(self.replay_memory, miniBatchSize)

        #Define arrays for states, future states, q values, and future q values.

        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states, predictionBatch)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states, predictionBatch)

        X = []
        y = []

        #Formula for calculating q-values

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        #Log tensorboard metrics by step

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step

        history = self.model.fit(np.array(X) / 255, np.array(y), batch_size=trainingBatch, verbose=0, shuffle=False,
                                 callbacks=[self.tensorboard] if log_this_step else None)

        #Log accuracy and loss metrics

        if log_this_step:
            accuracy = history.history['accuracy'][0]
            loss = history.history['loss'][0]
            self.tensorboard.update_stats(accuracy=accuracy, loss=loss)
            self.target_update_counter += 1

        if self.target_update_counter > updateEvery:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 5)).astype(np.float32)

        self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)

#Custom tensorboard class to write logs

class ModifiedTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
            self.step += 1
            self.writer.flush()

    #Main function for running code

if __name__ == "__main__":
    FPS = 20
    ep_rewards = [-200]

    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    #Create folder to store saved models

    if not os.path.isdir("models"):
        os.makedirs("models")

    DQN = QNetwork()
    environment = Environment()

    #Create seperate thread for model training

    trainer_thread = Thread(target=DQN.train_in_loop, daemon=True)
    trainer_thread.start()

    while not DQN.training_initialized:
        time.sleep(0.01)

    DQN.get_qs(np.ones((environment.im_height, environment.im_width, 3)))

    #Tqdm to add progress bars and loops to training, to track the number of episodes

    for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episodes"):
        environment.collision_hist = []
        environment.lane_hist = []
        DQN.tensorboard.step = episode
        episode_reward = 0
        step = 1
        current_state = environment.reset()
        done = False
        episode_start = time.time()

        #If random value lower than epsilon, pick a random action, otherwise pick action based on Q values

        while True:
            if np.random.random() > epsilon:
                action = np.argmax(DQN.get_qs(current_state))
            else:
                action = np.random.randint(0, 5)
                time.sleep(1 / FPS)
            new_state, reward, done, _ = environment.step(action)
            episode_reward += reward
            DQN.update_replay_memory((current_state, action, reward, new_state, done))
            current_state = new_state
            step += 1

            if done:
                break

        for actor in environment.actor_list:
            actor.destroy()

        #Log rewards for agent in tensorboard

        ep_rewards.append(episode_reward)
        if not episode % statsEvery or episode == 1:
            average_reward = sum(ep_rewards[-statsEvery:]) / len(ep_rewards[-statsEvery:])
            min_reward = min(ep_rewards[-statsEvery:])
            max_reward = max(ep_rewards[-statsEvery:])
            DQN.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            #Save model everytime we get a better model than the previous

            if min_reward >= minReward:
                DQN.model.save(f'models/{modelName}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        if epsilon > minEpsilon:
            epsilon *= EPSILON_DECAY
            epsilon = max(minEpsilon, epsilon)

    #Training finished, save final model

    DQN.terminate = True
    trainer_thread.join()
    DQN.model.save(f'models/{modelName}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
