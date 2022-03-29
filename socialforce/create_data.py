"""
Simulate crowd flow by Social Force Model
Provide training data for GNS
"""

import math
import random
import json
from typing import List, Tuple, Callable
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

from absl import app
from absl import flags

import socialforce

FLAGS = flags.FLAGS

flags.DEFINE_integer('n_train_data', 1000,
                     'The number of train data', lower_bound=1)
flags.DEFINE_integer('n_valid_data', 100,
                     'The number of valid data', lower_bound=1)
flags.DEFINE_integer('n_test_data', 100,
                     'The number of test data', lower_bound=1)
flags.DEFINE_integer('simulation_length', 100,
                     'The length of each simulation', lower_bound=1)
flags.DEFINE_integer('n_agents_mean', 20,
                     'The mean of the number of agents', lower_bound=1)
flags.DEFINE_integer(
    'n_agents_std', 1, 'The standard deviation of the number of agents', lower_bound=0)

flags.DEFINE_string('base_output_path', './output/',
                    'The base path for output data')

tf.enable_eager_execution()

Position = Tuple[float, float]
Segment = Tuple[Position, Position]
State = np.ndarray
States = np.ndarray  # time-series data of State
Space = List[np.ndarray]

CreatePosition = Callable


class Simulation():
    """
    Contain all information of Social Force Model simulation
    This consist of
    - State
    - Space
    """

    def __init__(self, simulation_length: int = 100, n_agents: int = 50):
        self.simulation_length = simulation_length
        self.n_agents = n_agents
        self.state = np.array([])
        self.space = []

    def create_rectangular_random_position(
            self,
            x_lower: float = -10.0,
            x_upper: float = -3.0,
            y_lower: float = -10.0,
            y_upper: float = 10.0) -> Position:
        """Create agents randomly in the rectangular area"""
        rand_x = random.uniform(x_lower, x_upper)
        rand_y = random.uniform(y_lower, y_upper)
        return rand_x, rand_y

    def _get_xy_from_rd(self, radius: float, degree: float) -> Position:
        """Return XY coordinates from polar coodinates"""
        rad = math.radians(degree)
        x = radius * math.cos(rad)
        y = radius * math.sin(rad)
        return x, y

    def create_circular_random_position(
            self,
            r_lower: float = 5.0,
            r_upper: float = 8.0,
            theta_lower: float = 0.0,
            theta_upper: float = 360.0) -> Position:
        """Create agents ramdomly in the """
        rand_radius = random.uniform(r_lower, r_upper)
        rand_angle = random.uniform(theta_lower, theta_upper)
        return self._get_xy_from_rd(rand_radius, rand_angle)

    def add_agents(self, destination: Position = (0.0, 0.0), create_position: CreatePosition = create_rectangular_random_position) -> State:
        """Return initial agent state"""
        init_speed = (1.0, 0.0)
        if self.state.shape[0] == 0:
            self.state: np.ndarray = np.array([
                [*create_position(self), *init_speed, *destination] for _ in range(self.n_agents)])
        else:
            new_state = np.array([
                [*create_position(self), *init_speed, *destination] for _ in range(self.n_agents)])
            self.state = np.append(self.state, new_state, axis=0)
        return self.state

    def add_random_dest_agents(self) -> State:
        dest = self.create_rectangular_random_position(
            x_lower = 2.0,
            x_upper = 8.0,
            y_lower = -3.0,
            y_upper = 3.0
            )
        self.add_agents(destination=dest)
        return self.state

    def add_obstacle(self, obstacle: Segment) -> Space:
        """Add obstacle to the space"""
        length = np.sqrt((obstacle[0][0] - obstacle[1][0]) ** 2 + (obstacle[0][1] - obstacle[1][1]) ** 2)
        n_split = round(10 * length)
        if len(self.space) == 0:
            self.space = [np.linspace(*obstacle, 100)]
        else:
            self.space.extend([np.linspace(*obstacle, 100)])
        return self.space

    def add_random_obstacle(
        self,
        x_lower: float = -10.0,
        x_upper: float = 10.0,
        y_lower: float = -10.0,
        y_upper: float = 10.0) -> Space:
        obstacle_start_point = self.create_rectangular_random_position(x_lower, x_upper, y_lower, y_upper)
        obstacle_end_point = self.create_rectangular_random_position(x_lower, x_upper, y_lower, y_upper)
        obstacle: Segment = (obstacle_start_point, obstacle_end_point)
        self.add_obstacle(obstacle)
        return self.space
    
    def create_random_hole_position(self, lower: float = -5.0, upper: float = 5.0):
        hole_width = 2.0
        hole_lower = random.uniform(lower, (upper - hole_width))
        hole_upper = hole_lower + hole_width
        return hole_lower, hole_upper

    def add_hole(self, hole: Tuple[float, float], x_pos: float = 0.0) -> Space:
        """Add obstacle with a hole to the space"""
        self.add_obstacle(((x_pos, -10), (x_pos, hole[0])))
        self.add_obstacle(((x_pos, hole[1]), (x_pos, 10)))
        return self.space

    def add_random_hole(self, x_pos: float = 0.0):
        hole = self.create_random_hole_position()
        self.add_hole(hole)

    def execute(self):
        s = socialforce.Simulator(
            self.state, socialforce.PedSpacePotential(self.space))
        self.states = np.stack([s.step().state.copy()
                                for _ in range(self.simulation_length)])

    def visualize(self, output_filename: str = './output/sample.gif') -> None:
        """Visualize social force simulation"""
        with socialforce.show.animation(
                len(self.states),
                output_filename,
                writer='pillow') as context:
            ax = context['ax']
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            # TODO: determine lim dynamically according to state and space
            ax.set_xlim(-20, 20)
            ax.set_ylim(-20, 20)

            for s in self.space:
                ax.plot(s[:, 0], s[:, 1], 'o', color='black', markersize=2.5)

            actors = []
            for ped in range(self.states.shape[1]):
                speed = np.linalg.norm(self.states[0, ped, 2:4])
                radius = 0.2 + speed / 2.0 * 0.3
                p = plt.Circle(self.states[0, ped, 0:2], radius=radius,
                               facecolor='black' if self.states[0,
                                                                ped, 4] > 0 else 'white',
                               edgecolor='black')
                actors.append(p)
                ax.add_patch(p)

            def update(i):
                for ped, p in enumerate(actors):
                    # p.set_data(states[i:i+5, ped, 0], states[i:i+5, ped, 1])
                    p.center = self.states[i, ped, 0:2]
                    speed = np.linalg.norm(self.states[i, ped, 2:4])
                    p.set_radius(0.2 + speed / 2.0 * 0.3)

            context['update_function'] = update


class SimulationAdmin():
    """
    Admin class for Simulation class. This create tfrecord and metadata,
    administrate a number of simulation.
    """

    def __init__(self, n_train_data: int = 1000, n_valid_data: int = 100, n_test_data: int = 100, simulation_length: int = 100, destination: Position = (0.0, 0.0)):
        self.n_data_dict: dict = {
            'train': n_train_data,
            'valid': n_valid_data,
            'test': n_test_data
        }
        self.simulation_length = simulation_length
        self.destination = destination

    def _bytes_feature(self, value):
        """string / byte 型から byte_list を返す"""
        if isinstance(value, type(tf.constant(0))):
            # BytesList won't unpack a string from an EagerTensor.
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """float / double 型から float_list を返す"""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        """bool / enum / int / uint 型から Int64_list を返す"""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def create_metadata(self, metadata: dict, base_output_path: str) -> None:
        """Create json formatted metadata file"""
        # file_path = "../deepmind-research/learning_to_simulate/datasets/SFM/metadata.json"
        file_path = base_output_path + 'metadata.json'
        print(metadata)
        with open(file_path, 'w') as f:
            json.dump(metadata, f)

    def create_tfrecord(
            self,
            base_output_path: str,
            n_data: int,
            mode: str) -> None:
        """Create tfrecord formatted feature data file"""
        timestep_num_list = np.array([])
        agents_num_list = np.array([])
        vel_mean_list = np.empty((0, 2))
        vel_var_list = np.empty((0, 2))
        acc_mean_list = np.empty((0, 2))
        acc_var_list = np.empty((0, 2))
        # file name is each of train.tfrecord/test.tfrecord/valid.tfrecord
        # file_path = "../deepmind-research/learning_to_simulate/datasets/SFM/train.tfrecord"
        file_path = base_output_path + mode + '.tfrecord'
        with tf.python_io.TFRecordWriter(file_path) as w:
            for i in range(n_data):
                print(f"Dealing with ({i + 1}/{n_data}) simulation")
                simulation = Simulation(self.simulation_length)
                # TODO: implement as callback function
                simulation.add_random_dest_agents()
                simulation.add_random_hole()
                simulation.execute()
                if i == 0:
                    simulation.visualize()
                timestep_num_list = np.append(timestep_num_list, simulation.states.shape[0])
                agents_num_list = np.append(agents_num_list, simulation.states.shape[1])
                vel_mean_list = np.append(
                    vel_mean_list,
                    np.array([np.mean(simulation.states[:, :, 2:4], axis=(0, 1))]),
                    axis=0
                )
                vel_var_list = np.append(
                    vel_var_list,
                    np.array([np.mean(
                        (simulation.states[:, :, 2:4] - vel_mean_list[i]) ** 2,
                        axis=(0, 1))]),
                    axis=0
                )
                first_step_vel_mean = np.mean(simulation.states[0, :, 2:4], axis=0)
                final_step_vel_mean = np.mean(simulation.states[-1, :, 2:4], axis=0)
                try:
                    # acc_mean = (\bar{v_(L+1)} - \bar{v_1}) / L
                    acc_mean_list = np.append(acc_mean_list, np.array(
                        [(final_step_vel_mean - first_step_vel_mean) / (timestep_num_list[i] - 1)]), axis=0)
                except ZeroDivisionError as e:
                    print(e)
                    print("Simulation timestep should be more than 1.")
                acc_var_list = np.append(
                    acc_var_list,
                    np.array([np.mean(
                        (np.diff(simulation.states[:, :, 2:4], axis=0) -
                        acc_mean_list[i]) ** 2,
                        axis=(0, 1))]),
                    axis=0
                )
                obstacle_agents = self.create_obstacle_agents(simulation.space, self.simulation_length)
                moving_agents = self.create_moving_agents(simulation.states)
                agents = np.concatenate([obstacle_agents, moving_agents], axis=1)
                obstacle_row = [np.int64(3)] * obstacle_agents.shape[1]
                moving_row = [np.int64(8)] * moving_agents.shape[1]
                agents_row = obstacle_row + moving_row
                print(agents.shape)
                destination_x = [np.float32(self.destination[0])] * agents.shape[1]
                destination_y = [np.float32(self.destination[1])] * agents.shape[1]

                context = tf.train.Features(feature={
                    'particle_type': self._bytes_feature(np.array(agents_row).tobytes()),
                    'destination_x': self._bytes_feature(np.array(destination_x).tobytes()),
                    'destination_y': self._bytes_feature(np.array(destination_y).tobytes()),
                    'key': self._int64_feature(np.int64(0))
                })
                description_feature = [
                    self._bytes_feature((v - np.stack([destination_x, destination_y], axis=1)).tobytes()) for v in agents
                ]
                feature_lists = tf.train.FeatureLists(feature_list={
                    "position": tf.train.FeatureList(feature=description_feature)
                })

                sequence_example = tf.train.SequenceExample(context=context,
                                                            feature_lists=feature_lists)
                w.write(sequence_example.SerializeToString())
        vel_mean = np.sum(
            vel_mean_list *
            timestep_num_list.reshape((-1, 1)) * agents_num_list.reshape((-1, 1)),
            axis=0) / np.sum(timestep_num_list * agents_num_list)
        acc_mean = np.sum(
            acc_mean_list * (timestep_num_list.reshape((-1, 1)) -
                            1) * agents_num_list.reshape((-1, 1)),
            axis=0) / np.sum((timestep_num_list - 1) * agents_num_list)
        vel_var = np.sum(
            (vel_var_list + vel_mean_list ** 2) * timestep_num_list.reshape((-1, 1)) *
            agents_num_list.reshape((-1, 1)), axis=0) / np.sum(timestep_num_list * agents_num_list)
        acc_var = np.sum(
            (acc_var_list + acc_mean_list ** 2) * (timestep_num_list.reshape((-1, 1)) - 1) *
            agents_num_list.reshape((-1, 1)), axis=0) / np.sum((timestep_num_list - 1) * agents_num_list)
        vel_std = np.sqrt(vel_var)
        acc_std = np.sqrt(acc_var)
        metadata: dict = {
            'bounds': [[-15.0, 15.0], [-10.0, 10.0]],
            'sequence_length': self.simulation_length - 1,
            'default_connectivity_radius': 0.2,
            'dim': 2,
            'dt': 0.4,
            'vel_mean': vel_mean.tolist(),
            'vel_std': vel_std.tolist(),
            'acc_mean': acc_mean.tolist(),
            'acc_std': acc_std.tolist()
        }
        self.create_metadata(metadata, base_output_path)

    def create_all_data(self, n_train_data: int = 1000, n_valid_data: int = 100, n_test_data: int = 100, simulation_length: int = 100):
        self.n_data_dict['train'] = n_train_data
        self.n_data_dict['valid'] = n_valid_data
        self.n_data_dict['test'] = n_test_data
        self.simulation_length = simulation_length
        for mode, n_data in self.n_data_dict.items():
            if n_data > 0:
                self.create_tfrecord(base_output_path=FLAGS.base_output_path, n_data=n_data, mode=mode)

    def create_train_data(self, n_data: int = 1000):
        self.create_all_data(n_train_data=n_data, n_valid_data=0, n_test_data=0)

    def create_valid_data(self, n_data: int = 100):
        self.create_all_data(n_train_data=0, n_valid_data=n_data, n_test_data=0)

    def create_test_data(self, n_data: int = 100):
        self.create_all_data(n_train_data=0, n_valid_data=0, n_test_data=n_data)
    
    def create_obstacle_agents(self, space: Space, simulation_length: int) -> np.ndarray:
        """Return obstacle agents' position sequence as np.float32"""
        obstacle_agents = np.array(space).astype(np.float32).reshape((-1, 2))
        return np.array([obstacle_agents] * simulation_length)

    def create_moving_agents(self, states: States) -> np.ndarray:
        """Return agents' position sequence as np.float32"""
        return states[:, :, 0:2].astype(np.float32)


def main(_):
    """Output multiple simulation results and each animations"""
    admin = SimulationAdmin()
    admin.create_all_data(FLAGS.n_train_data, FLAGS.n_valid_data, FLAGS.n_test_data, FLAGS.simulation_length)


if __name__ == '__main__':
    app.run(main)
