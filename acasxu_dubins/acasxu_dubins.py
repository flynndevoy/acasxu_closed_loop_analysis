'''
ACASXu neural networks closed loop simulation with dubin's car dynamics

Used for falsification, where the opponent is allowed to maneuver over time
'''

from functools import lru_cache
import time
import math
import argparse

import numpy as np
from scipy import ndimage
from scipy.linalg import expm

import matplotlib.pyplot as plt
from matplotlib import patches, animation
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.lines import Line2D

import onnxruntime as ort
from numba import njit

def init_plot():
    'initialize plotting style'

    #matplotlib.use('TkAgg') # set backend

    p = 'bak_matplotlib.mlpstyle'
    plt.style.use(['bmh', p])

def load_network(last_cmd):
    '''load the one neural network as a 2-tuple (range_for_scaling, means_for_scaling)'''

    onnx_filename = f"ACASXU_run2a_{last_cmd + 1}_1_batch_2000.onnx"

    #print(f"Loading {mat_filename}...")
    #matfile = loadmat(mat_filename)
    #range_for_scaling = matfile['range_for_scaling'][0]
    #means_for_scaling = matfile['means_for_scaling'][0]
    #mat_filename = f"ACASXU_run2a_1_1_batch_2000.mat"

    means_for_scaling = [19791.091, 0.0, 0.0, 650.0, 600.0, 7.5188840201005975]
    range_for_scaling = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]

    session = ort.InferenceSession(onnx_filename)

    # warm up the network
    i = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    i.shape = (1, 1, 1, 5)
    session.run(None, {'input': i})

    return session, range_for_scaling, means_for_scaling

def load_networks():
    '''load the 5 neural networks into nn-enum's data structures and return them as a list'''

    nets = []

    for last_cmd in range(5):
        nets.append(load_network(last_cmd))

    return nets


'Lru Cache.'
def get_time_elapse_mat(command1, dt, command2=0):
    '''get the matrix exponential for the given command

    state: x, y, vx, vy, x2, y2, vx2, vy2
    '''

    y_list = [0.0, 1.5, -1.5, 3.0, -3.0]
    y1 = y_list[command1]
    y2 = y_list[command2]

    dtheta1 = (y1 / 180 * np.pi)
    dtheta2 = (y2 / 180 * np.pi)

    a_mat = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0], # x' = vx
        [0, 0, 0, 1, 0, 0, 0, 0], # y' = vy
        [0, 0, 0, -dtheta1, 0, 0, 0, 0], # vx' = -vy * dtheta1
        [0, 0, dtheta1, 0, 0, 0, 0, 0], # vy' = vx * dtheta1
    #
        [0, 0, 0, 0, 0, 0, 1, 0], # x' = vx
        [0, 0, 0, 0, 0, 0, 0, 1], # y' = vy
        [0, 0, 0, 0, 0, 0, 0, -dtheta2], # vx' = -vy * dtheta2
        [0, 0, 0, 0, 0, 0, dtheta2, 0], # vy' = vx * dtheta1
        ], dtype=float)

    return expm(a_mat * dt)

def run_network(network_tuple, x, stdout=False):
    'run the network and return the output'

    session, range_for_scaling, means_for_scaling = network_tuple

    # normalize input
    for i in range(5):
        x[i] = (x[i] - means_for_scaling[i]) / range_for_scaling[i]

    if stdout:
        print(f"input (after scaling): {x}")

    in_array = np.array(x, dtype=np.float32)
    in_array.shape = (1, 1, 1, 5)
    outputs = session.run(None, {'input': in_array})

    return outputs[0][0]

@njit(cache=True)
def state7_to_state5(state7, v_own, v_int):
    """compute rho, theta, psi from state7"""

    assert len(state7) == 7

    x1, y1, theta1, x2, y2, theta2, _ = state7

    rho = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    dy = y2 - y1
    dx = x2 - x1

    theta = np.arctan2(dy, dx)
    psi = theta2 - theta1

    theta -= theta1

    while theta < -np.pi:
        theta += 2 * np.pi

    while theta > np.pi:
        theta -= 2 * np.pi

    if psi < -np.pi:
        psi += 2 * np.pi

    while psi > np.pi:
        psi -= 2 * np.pi

    return np.array([rho, theta, psi, v_own, v_int])

@njit(cache=True)
def state7_to_state8(state7, v_own, v_int):
    """compute x,y, vx, vy, x2, y2, vx2, vy2 from state7"""

    assert len(state7) == 7

    x1 = state7[0]
    y1 = state7[1]
    vx1 = math.cos(state7[2]) * v_own
    vy1 = math.sin(state7[2]) * v_own

    x2 = state7[3]
    y2 = state7[4]
    vx2 = math.cos(state7[5]) * v_int
    vy2 = math.sin(state7[5]) * v_int

    return np.array([x1, y1, vx1, vy1, x2, y2, vx2, vy2])

@lru_cache(maxsize=None)
def get_airplane_img():
    """load airplane image form file"""

    img = plt.imread('airplane.png')

    return img

def init_time_elapse_mats(dt):
    """get value of time_elapse_mats array"""

    rv = []

    for cmd in range(5):
        rv.append([])

        for int_cmd in range(5):
            mat = get_time_elapse_mat(cmd, dt, int_cmd)
            rv[-1].append(mat)

    return rv

@njit(cache=True)
def step_state(state7, v_own, v_int, time_elapse_mat, dt):
    """perform one time step with the given commands"""

    state8_vec = state7_to_state8(state7, v_own, v_int)

    s = time_elapse_mat @ state8_vec

    # extract observation (like theta) from state
    new_time = state7[-1] + dt
    theta1 = math.atan2(s[3], s[2])
    theta2 = math.atan2(s[7], s[6])
    rv = np.array([s[0], s[1], theta1, s[4], s[5], theta2, new_time])

    return rv

class State():
    'state of execution container'

    nets = load_networks()
    plane_size = 1500

    nn_update_rate = 1.0 # todo: make this a parameter
    dt = 1.0
    min_dwell_time = 0.0
    q_hysteresis_margin = 0.0

    'Valid range" [100, 1145]'
    #v_own = 800 # ft/sec

    'Valid range: [60,1145]'
    #v_int = 500

    time_elapse_mats = init_time_elapse_mats(dt)

    def __init__(self, init_vec, v_own=800, v_int=500, save_states=False):
        assert len(init_vec) == 7, "init vec should have length 7"

        self.vec = np.array(init_vec, dtype=float) # current state
        self.next_nn_update = 0
        self.command = 0 # initial command
        self.v_own = v_own
        self.v_int = v_int
        self.last_advisory_change_time = init_vec[-1]

        # these are set when simulation() if save_states=True
        self.save_states = save_states
        self.vec_list = [] # state history
        self.commands = [] # commands history
        self.int_commands = [] # intruder command history

        # used only if plotting
        self.artists_dict = {} # set when make_artists is called
        self.img = None # assigned if plotting

        # assigned by simulate()
        self.u_list = []
        self.u_list_index = None
        self.min_dist = np.inf
        self.min_dist_time = None
        self.first_alert_time = None
        self.first_alert_range = None
        self.alert_active_time = 0.0
        self.alert_episode_count = 0
        self.in_alert_episode = False
        self.advisory_change_count = 0
        self.reversal_count = 0
        self.last_non_coc_direction = 0
        self.sim_metrics = {}

    def __str__(self):
        x1, y1, _theta1, x2, y2, _theta2, _ = self.vec
        rho = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

        return f'State(v_own: {self.v_own}, v_int: {self.v_int}, rho: {rho})'

    def artists_list(self):
        'return list of artists'

        return list(self.artists_dict.values())

    def set_plane_visible(self, vis):
        'set ownship plane visibility status'

        self.artists_dict['dot0'].set_visible(not vis)
        self.artists_dict['circle0'].set_visible(False) # circle always False
        self.artists_dict['lc0'].set_visible(True)
        self.artists_dict['plane0'].set_visible(vis)

    def update_artists(self, axes):
        '''update artists in self.artists_dict to be consistant with self.vec, returns a list of artists'''

        assert self.artists_dict
        rv = []

        x1, y1, theta1, x2, y2, theta2, _ = self.vec

        for i, x, y, theta in zip([0, 1], [x1, x2], [y1, y2], [theta1, theta2]):
            key = f'plane{i}'

            if key in self.artists_dict:
                plane = self.artists_dict[key]
                rv.append(plane)

                if plane.get_visible():
                    theta_deg = (theta - np.pi / 2) / np.pi * 180 # original image is facing up, not right
                    original_size = list(self.img.shape)
                    img_rotated = ndimage.rotate(self.img, theta_deg, order=1)
                    rotated_size = list(img_rotated.shape)
                    ratios = [r / o for r, o in zip(rotated_size, original_size)]
                    plane.set_data(img_rotated)

                    size = State.plane_size
                    width = size * ratios[0]
                    height = size * ratios[1]
                    box = Bbox.from_bounds(x - width/2, y - height/2, width, height)
                    tbox = TransformedBbox(box, axes.transData)
                    plane.bbox = tbox

            key = f'dot{i}'
            if key in self.artists_dict:
                dot = self.artists_dict[f'dot{i}']
                cir = self.artists_dict[f'circle{i}']
                rv += [dot, cir]

                dot.set_data([x], [y])
                cir.set_center((x, y))

        # line collection
        lc = self.artists_dict['lc0']
        rv.append(lc)

        int_lc = self.artists_dict['int_lc0']
        rv.append(int_lc)

        self.update_lc_artists(lc, int_lc)

        return rv

    def update_lc_artists(self, own_lc, int_lc):
        'update line collection artist based on current state'

        assert self.vec_list

        for lc_index, lc in enumerate([own_lc, int_lc]):
            paths = lc.get_paths()
            colors = []
            lws = []
            paths.clear()
            last_command = -1
            codes = []
            verts = []

            for i, vec in enumerate(self.vec_list):
                if np.linalg.norm(vec - self.vec) < 1e-6:
                    # done
                    break

                if lc_index == 0:
                    cmd = self.commands[i]
                else:
                    cmd = self.int_commands[i]

                x = 0 if lc_index == 0 else 3
                y = 1 if lc_index == 0 else 4

                # command[i] is the line from i to (i+1)
                if cmd != last_command:
                    if codes:
                        paths.append(Path(verts, codes))

                    codes = [Path.MOVETO]
                    verts = [(vec[x], vec[y])]

                    if cmd == 1: # weak left
                        lws.append(2)
                        colors.append('b')
                    elif cmd == 2: # weak right
                        lws.append(2)
                        colors.append('c')
                    elif cmd == 3: # strong left
                        lws.append(2)
                        colors.append('g')
                    elif cmd == 4: # strong right
                        lws.append(2)
                        colors.append('r')
                    else:
                        assert cmd == 0 # coc
                        lws.append(2)
                        colors.append('k')

                codes.append(Path.LINETO)

                verts.append((self.vec_list[i+1][x], self.vec_list[i+1][y]))

            # add last one
            if codes:
                paths.append(Path(verts, codes))

            lc.set_lw(lws)
            lc.set_color(colors)

    def make_artists(self, axes, show_intruder):
        'make self.artists_dict'

        assert self.vec_list
        self.img = get_airplane_img()

        posa_list = [(v[0], v[1], v[2]) for v in self.vec_list]
        posb_list = [(v[3], v[4], v[5]) for v in self.vec_list]

        pos_lists = [posa_list, posb_list]

        if show_intruder:
            pos_lists.append(posb_list)

        for i, pos_list in enumerate(pos_lists):
            x, y, theta = pos_list[0]

            l = axes.plot(*zip(*pos_list), f'c-', lw=0, zorder=1)[0]
            l.set_visible(False)
            self.artists_dict[f'line{i}'] = l

            if i == 0:
                lc = LineCollection([], lw=2, animated=True, color='k', zorder=1)
                axes.add_collection(lc)
                self.artists_dict[f'lc{i}'] = lc

                int_lc = LineCollection([], lw=2, animated=True, color='k', zorder=1)
                axes.add_collection(int_lc)
                self.artists_dict[f'int_lc{i}'] = int_lc

            # only sim_index = 0 gets intruder aircraft
            if i == 0 or (i == 1 and show_intruder):
                size = State.plane_size
                box = Bbox.from_bounds(x - size/2, y - size/2, size, size)
                tbox = TransformedBbox(box, axes.transData)
                box_image = BboxImage(tbox, zorder=2)

                theta_deg = (theta - np.pi / 2) / np.pi * 180 # original image is facing up, not right
                img_rotated = ndimage.rotate(self.img, theta_deg, order=1)

                box_image.set_data(img_rotated)
                axes.add_artist(box_image)
                self.artists_dict[f'plane{i}'] = box_image

            if i == 0:
                dot = axes.plot([x], [y], 'k.', markersize=6.0, zorder=2)[0]
                self.artists_dict[f'dot{i}'] = dot

                rad = 1500
                c = patches.Ellipse((x, y), rad, rad, color='k', lw=3.0, fill=False)
                axes.add_patch(c)
                self.artists_dict[f'circle{i}'] = c

    def step(self):
        'execute one time step and update the model'

        tol = 1e-6

        if self.next_nn_update < tol:
            assert abs(self.next_nn_update) < tol, f"time step doesn't sync with nn update time. " + \
                      f"next update: {self.next_nn_update}"

            # update command
            self.update_command()

            self.next_nn_update = State.nn_update_rate

        self.next_nn_update -= State.dt
        intruder_cmd = self.u_list[self.u_list_index]

        if self.save_states:
            self.commands.append(self.command)
            self.int_commands.append(intruder_cmd)

        rho_now = math.sqrt((self.vec[0] - self.vec[3])**2 + (self.vec[1] - self.vec[4])**2)

        if self.command != 0:
            if self.first_alert_time is None:
                self.first_alert_time = self.vec[-1]
                self.first_alert_range = rho_now

            self.alert_active_time += State.dt

            if not self.in_alert_episode:
                self.in_alert_episode = True
                self.alert_episode_count += 1
        elif self.in_alert_episode:
            self.in_alert_episode = False

        time_elapse_mat = State.time_elapse_mats[self.command][intruder_cmd] #get_time_elapse_mat(self.command, State.dt, intruder_cmd)

        self.vec = step_state(self.vec, self.v_own, self.v_int, time_elapse_mat, State.dt)

    def simulate(self, cmd_list):
        '''simulate system

        saves result in self.vec_list
        also saves self.min_dist
        '''

        self.u_list = cmd_list
        self.u_list_index = None
        self.min_dist = np.inf
        self.min_dist_time = None
        self.first_alert_time = None
        self.first_alert_range = None
        self.alert_active_time = 0.0
        self.alert_episode_count = 0
        self.in_alert_episode = False
        self.advisory_change_count = 0
        self.reversal_count = 0
        self.last_non_coc_direction = 0
        self.sim_metrics = {}

        assert isinstance(cmd_list, list)
        tmax = len(cmd_list) * State.nn_update_rate

        t = 0.0

        if self.save_states:
            rv = [self.vec.copy()]

        #self.min_dist = 0, math.sqrt((self.vec[0] - self.vec[3])**2 + (self.vec[1] - self.vec[4])**2), self.vec.copy()
        prev_dist_sq = (self.vec[0] - self.vec[3])**2 + (self.vec[1] - self.vec[4])**2
        min_dist_sq = prev_dist_sq
        min_dist_time = self.vec[-1]

        while t + 1e-6 < tmax:
            self.step()

            cur_dist_sq = (self.vec[0] - self.vec[3])**2 + (self.vec[1] - self.vec[4])**2
            if cur_dist_sq < min_dist_sq:
                min_dist_sq = cur_dist_sq
                min_dist_time = self.vec[-1]

            if self.save_states:
                rv.append(self.vec.copy())

            t += State.dt

            if cur_dist_sq > prev_dist_sq:
                #print(f"Distance was increasing at time {round(t, 2)}, stopping simulation. Min_dist: {round(prev_dist, 1)}ft")
                break

            prev_dist_sq = cur_dist_sq

        self.min_dist = math.sqrt(min_dist_sq)
        self.min_dist_time = min_dist_time
        alert_lead_time = None
        if self.first_alert_time is not None:
            alert_lead_time = self.min_dist_time - self.first_alert_time

        self.sim_metrics = {
            'min_dist': self.min_dist,
            'min_dist_time': self.min_dist_time,
            'first_alert_time': self.first_alert_time,
            'first_alert_range': self.first_alert_range,
            'alert_lead_time_to_cpa': alert_lead_time,
            'alert_active_time': self.alert_active_time,
            'alert_episode_count': self.alert_episode_count,
            'advisory_change_count': self.advisory_change_count,
            'reversal_count': self.reversal_count,
        }

        if self.save_states:
            self.vec_list = rv

        if not self.save_states:
            assert not self.vec_list
            assert not self.commands
            assert not self.int_commands

    def update_command(self):
        'update command based on current state'''

        rho, theta, psi, v_own, v_int = state7_to_state5(self.vec, self.v_own, self.v_int)

        # 0: rho, distance
        # 1: theta, angle to intruder relative to ownship heading
        # 2: psi, heading of intruder relative to ownship heading
        # 3: v_own, speed of ownship
        # 4: v_int, speed in intruder

        # min inputs: 0, -3.1415, -3.1415, 100, 0
        # max inputs: 60760, 3.1415, 3,1415, 1200, 1200

        if rho > 60760:
            new_command = 0
            hysteresis_ok = True
        else:
            last_command = self.command

            net = State.nets[last_command]

            state = [rho, theta, psi, v_own, v_int]

            res = run_network(net, state)
            new_command = np.argmin(res)

            hysteresis_ok = True
            if new_command != self.command and State.q_hysteresis_margin > 0.0:
                proposed_q = float(res[new_command])
                current_q = float(res[self.command])
                hysteresis_ok = (current_q - proposed_q) >= State.q_hysteresis_margin

        advisory_age = self.vec[-1] - self.last_advisory_change_time

        # Only issue an advisory update when all stability constraints are satisfied.
        if new_command != self.command:
            if hysteresis_ok and advisory_age + 1e-9 >= State.min_dwell_time:
                old_command = self.command
                self.command = new_command
                self.last_advisory_change_time = self.vec[-1]
                self.advisory_change_count += 1

                new_dir = State.get_advisory_direction(new_command)
                if new_dir != 0:
                    if self.last_non_coc_direction != 0 and new_dir != self.last_non_coc_direction:
                        self.reversal_count += 1

                    self.last_non_coc_direction = new_dir

                if old_command != 0 and new_command == 0 and self.in_alert_episode:
                    self.in_alert_episode = False

            #names = ['clear-of-conflict', 'weak-left', 'weak-right', 'strong-left', 'strong-right']

        if self.u_list_index is None:
            self.u_list_index = 0
        else:
            self.u_list_index += 1

            # repeat last command if no more commands
            self.u_list_index = min(self.u_list_index, len(self.u_list) - 1)

    @staticmethod
    def get_advisory_direction(cmd):
        """map command to turn direction: left=-1, none=0, right=1"""

        if cmd in (1, 3):
            return -1

        if cmd in (2, 4):
            return 1

        return 0

def summarize_performance(metrics, num_sims, total_runtime, false_alert_distance, nuisance_max_alert_time, nmac_distance):
    """print aggregate metrics for a sweep of simulations"""

    alerts = metrics['alerts']
    first_alert_ranges = np.array(metrics['first_alert_ranges'], dtype=float)
    lead_times = np.array(metrics['alert_lead_times'], dtype=float)
    min_dists = np.array(metrics['min_dists'], dtype=float)
    false_alerts = metrics['false_alerts']
    nuisance_alerts = metrics['nuisance_alerts']
    nmac_count = metrics['nmac_count']
    reversals = metrics['reversal_sims']
    advisory_changes = metrics['total_advisory_changes']

    sims_per_sec = num_sims / total_runtime if total_runtime > 0 else np.inf
    ms_per_sim = 1000.0 * total_runtime / num_sims if num_sims > 0 else np.inf

    print("\nPerformance metrics summary")
    print(f"  simulations: {num_sims}")
    if 'initial_turning_count' in metrics and 'initial_coc_count' in metrics:
        turning_count = metrics['initial_turning_count']
        coc_count = metrics['initial_coc_count']
        print(
            f"  initial advisory mix: turning={turning_count}/{num_sims} ({100.0 * turning_count / num_sims:.1f}%), "
            f"COC={coc_count}/{num_sims} ({100.0 * coc_count / num_sims:.1f}%)"
        )
    print(f"  runtime: {total_runtime:.3f}s total ({ms_per_sim:.3f} ms/sim, {sims_per_sec:.1f} sim/s)")

    print("  Detection and alerting range:")
    if alerts > 0:
        print(f"    alert rate: {alerts}/{num_sims} ({100.0 * alerts / num_sims:.2f}%)")
        print(f"    first alert range mean: {np.mean(first_alert_ranges):.1f} ft")
        print(f"    first alert range median: {np.median(first_alert_ranges):.1f} ft")
    else:
        print("    no alerts issued")

    print("  Alert timing relative to closest point of approach:")
    if lead_times.size > 0:
        print(f"    lead time mean (CPA - first alert): {np.mean(lead_times):.2f} s")
        print(f"    lead time median (CPA - first alert): {np.median(lead_times):.2f} s")
        late_count = int(np.sum(lead_times < 0))
        print(f"    late alerts (after CPA): {late_count}/{lead_times.size}")
    else:
        print("    no alert timing samples")

    print("  False alert and nuisance alert rate:")
    print(f"    false alert distance threshold: {false_alert_distance:.1f} ft")
    print(f"    false alerts: {false_alerts}/{num_sims} ({100.0 * false_alerts / num_sims:.2f}%)")
    print(f"    nuisance alert max active time: {nuisance_max_alert_time:.2f} s")
    print(f"    nuisance alerts: {nuisance_alerts}/{num_sims} ({100.0 * nuisance_alerts / num_sims:.2f}%)")

    print("  NMAC proxy metrics:")
    print(f"    NMAC distance threshold: {nmac_distance:.1f} ft")
    print(f"    NMAC proxy events: {nmac_count}/{num_sims} ({100.0 * nmac_count / num_sims:.2f}%)")
    print(f"    min distance mean: {np.mean(min_dists):.1f} ft")
    print(f"    min distance median: {np.median(min_dists):.1f} ft")

    print("  Advisory stability and reversals:")
    print(f"    advisory changes total: {advisory_changes}")
    print(f"    advisory changes mean per sim: {advisory_changes / num_sims:.3f}")
    print(f"    sims with at least one reversal: {reversals}/{num_sims} ({100.0 * reversals / num_sims:.2f}%)")

def plot(s, save_mp4):
    """plot a specific simulation"""

    s.vec = s.vec_list[0] # for printing the correct state
    print(f"plotting state {s}")

    init_plot()

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    axes.axis('equal')

    axes.set_title("ACAS Xu Simulations")
    axes.set_xlabel('X Position (ft)')
    axes.set_ylabel('Y Position (ft)')

    time_text = axes.text(0.02, 0.98, 'Time: 0', horizontalalignment='left', fontsize=14,
                          verticalalignment='top', transform=axes.transAxes)
    time_text.set_visible(True)

    custom_lines = [Line2D([0], [0], color='g', lw=2),
                    Line2D([0], [0], color='b', lw=2),
                    Line2D([0], [0], color='k', lw=2),
                    Line2D([0], [0], color='c', lw=2),
                    Line2D([0], [0], color='r', lw=2)]

    axes.legend(custom_lines, ['Strong Left', 'Weak Left', 'Clear of Conflict', 'Weak Right', 'Strong Right'], \
                fontsize=14, loc='lower left')

    s.make_artists(axes, show_intruder=True)
    states = [s]

    plt.tight_layout()

    num_steps = len(states[0].vec_list)
    interval = 20 # ms per frame
    freeze_frames = 10 if not save_mp4 else 80

    num_runs = 1 # 3
    num_frames = num_runs * num_steps + 2 * num_runs * freeze_frames

    #plt.savefig('plot.png')
    #plot_commands(states[0])

    def animate(f):
        'animate function'

        if not save_mp4:
            f *= 1 # multiplier to make animation faster

        # if (f+1) % 10 == 0:
        #     print(f"Frame: {f+1} / {num_frames}")

        run_index = f // (num_steps + 2 * freeze_frames)

        f = f - run_index * (num_steps + 2*freeze_frames)

        f -= freeze_frames

        f = max(0, f)
        f = min(f, num_steps - 1)

        num_states = len(states)

        if f == 0:
            # initiaze current run_index
            show_plane = num_states <= 10
            for s in states[:num_states]:
                s.set_plane_visible(show_plane)

            for s in states[num_states:]:
                for a in s.artists_list():
                    a.set_visible(False)

        time_text.set_text(f'Time: {f * State.dt:.1f}')

        artists = [time_text]

        for s in states[:num_states]:
            s.vec = s.vec_list[f]
            artists += s.update_artists(axes)

        for s in states[num_states:]:
            artists += s.artists_list()

        return artists

    my_anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=interval, blit=True, repeat=True)

    if save_mp4:
        writer = animation.writers['ffmpeg'](fps=50, metadata=dict(artist='Stanley Bak'), bitrate=1800)

        my_anim.save('sim.mp4', writer=writer)
    else:
        plt.show()

def make_random_input(seed, intruder_can_turn=True, num_inputs=100):
    """make a random input for the system"""

    np.random.seed(seed) # deterministic random numbers

    # state vector is: x, y, theta, x2, y2, theta2, time
    init_vec = np.zeros(7)
    init_vec[2] = np.pi / 2 # ownship moving up initially

    radius = 10000 + np.random.random() * 55000 # [10000, 65000]
    angle = np.random.random() * 2 * np.pi
    int_x = radius * np.cos(angle)
    int_y = radius * np.sin(angle)
    int_heading = np.random.random() * 2 * np.pi

    init_vec[3] = int_x
    init_vec[4] = int_y
    init_vec[5] = int_heading

    # intruder commands for every control period (0 to 4)
    if intruder_can_turn:
        cmd_list = []

        for _ in range(num_inputs):
            cmd_list.append(np.random.randint(5))
    else:
        cmd_list = [0] * num_inputs

    # generate random valid velocities
    #init_velo = [np.random.randint(100, 1146),
    #             np.random.randint(60, 1146)]
    init_velo = [np.random.randint(100, 1200),
                 np.random.randint(0, 1200)]

    return init_vec, cmd_list, init_velo

def get_initial_advisory(init_vec, v_own, v_int):
    """compute initial advisory for a candidate initial condition"""

    state5 = state7_to_state5(init_vec, v_own, v_int)

    if state5[0] > 60760:
        return 0 # rho exceeds network limit

    res = run_network(State.nets[0], state5)
    return int(np.argmin(res))

def main():
    'main entry point'

    # parse arguments
    parser = argparse.ArgumentParser(description='Run ACASXU Dublins model simulator.')
    parser.add_argument("--save-mp4", action='store_true', default=False, help="Save plotted mp4 files to disk.")
    parser.add_argument("--intruder-turn", action='store_true', default=False, help="Toggles boolean flag to allow intruder to perform \
                                                                                     commands other than flying straight.")
    parser.add_argument("--seed", type=str, default="min", help="Seed to simulate: use 'min' to search for minimum-distance seed, or provide an integer seed (e.g. 671).")
    parser.add_argument("--num-sims", type=int, default=1000, help="Number of random seeds to evaluate when --seed=min.")
    parser.add_argument("--turning-ratio", type=float, default=0.9, help="Target fraction of --seed=min simulations that start with a turning advisory (1-4).")
    parser.add_argument("--max-sampling-attempts", type=int, default=250000, help="Maximum candidate seeds examined while enforcing the turning/COC start ratio.")
    parser.add_argument("--min-dwell-time", type=float, default=0.0, help="Minimum advisory dwell time in seconds before command changes are allowed.")
    parser.add_argument("--q-hysteresis-margin", type=float, default=0.0, help="Minimum Q-value improvement required to change advisories.")
    parser.add_argument("--nmac-distance", type=float, default=500.0, help="Distance threshold in feet for NMAC proxy events.")
    parser.add_argument("--false-alert-distance", type=float, default=4000.0, help="If an alert occurs but min distance stays above this threshold, count as false alert.")
    parser.add_argument("--nuisance-max-alert-time", type=float, default=2.0, help="Maximum total alert-active time (s) to classify an alert as nuisance.")
    args = parser.parse_args()

    intruder_can_turn = args.intruder_turn
    save_mp4 = args.save_mp4
    seed_choice = args.seed.strip().lower()
    num_sims = max(1, args.num_sims)
    turning_ratio = min(1.0, max(0.0, args.turning_ratio))
    max_sampling_attempts = max(1, args.max_sampling_attempts)
    State.min_dwell_time = max(0.0, args.min_dwell_time)
    State.q_hysteresis_margin = max(0.0, args.q_hysteresis_margin)
    nmac_distance = max(0.0, args.nmac_distance)
    false_alert_distance = max(0.0, args.false_alert_distance)
    nuisance_max_alert_time = max(0.0, args.nuisance_max_alert_time)

    interesting_seed = -1
    interesting_state = None

    if seed_choice == "min":
        # Build a stratified set of starts: mostly turning-advisory starts with a COC minority.
        target_turning = int(round(num_sims * turning_ratio))
        target_coc = num_sims - target_turning
        turning_cases = []
        coc_cases = []
        candidate_seed = 0

        print(f"Selecting starts: target_turning={target_turning}, target_coc={target_coc}")

        while (len(turning_cases) < target_turning or len(coc_cases) < target_coc) and candidate_seed < max_sampling_attempts:
            init_vec, cmd_list, init_velo = make_random_input(candidate_seed, intruder_can_turn=intruder_can_turn)
            v_own = init_velo[0]
            v_int = init_velo[1]
            command = get_initial_advisory(init_vec, v_own, v_int)

            case = (candidate_seed, init_vec, cmd_list, init_velo)

            if command == 0:
                if len(coc_cases) < target_coc:
                    coc_cases.append(case)
            elif len(turning_cases) < target_turning:
                turning_cases.append(case)

            candidate_seed += 1

        if len(turning_cases) < target_turning or len(coc_cases) < target_coc:
            raise RuntimeError(
                f"Could not satisfy requested start mix within {max_sampling_attempts} attempts. "
                f"Need turning={target_turning}, coc={target_coc}; got turning={len(turning_cases)}, coc={len(coc_cases)}."
            )

        selected_cases = turning_cases + coc_cases
        # Deterministic shuffle so order is mixed but repeatable.
        np.random.default_rng(0).shuffle(selected_cases)

        print(
            f"Selected {len(selected_cases)} starts from {candidate_seed} candidates "
            f"({len(turning_cases)} turning, {len(coc_cases)} COC)"
        )

        start = time.perf_counter()
        metrics = {
            'alerts': 0,
            'first_alert_ranges': [],
            'alert_lead_times': [],
            'false_alerts': 0,
            'nuisance_alerts': 0,
            'nmac_count': 0,
            'total_advisory_changes': 0,
            'reversal_sims': 0,
            'min_dists': [],
            'initial_turning_count': len(turning_cases),
            'initial_coc_count': len(coc_cases),
        }

        for case in selected_cases:
            seed, init_vec, cmd_list, init_velo = case
            if seed % 1000 == 0:
                print(f"{(seed//1000) % 10}", end='', flush=True)
            elif seed % 100 == 0:
                print(".", end='', flush=True)

            v_own = init_velo[0]
            v_int = init_velo[1]

            # run the simulation
            s = State(init_vec, v_own, v_int, save_states=False)
            s.simulate(cmd_list)
            sim_m = s.sim_metrics
            alerted = sim_m['first_alert_time'] is not None

            if alerted:
                metrics['alerts'] += 1
                metrics['first_alert_ranges'].append(sim_m['first_alert_range'])
                metrics['alert_lead_times'].append(sim_m['alert_lead_time_to_cpa'])

            if alerted and sim_m['min_dist'] > false_alert_distance:
                metrics['false_alerts'] += 1

            if alerted and sim_m['alert_active_time'] <= nuisance_max_alert_time and sim_m['min_dist'] > nmac_distance:
                metrics['nuisance_alerts'] += 1

            if sim_m['min_dist'] < nmac_distance:
                metrics['nmac_count'] += 1

            metrics['total_advisory_changes'] += sim_m['advisory_change_count']
            metrics['min_dists'].append(sim_m['min_dist'])

            if sim_m['reversal_count'] > 0:
                metrics['reversal_sims'] += 1

            # save most interesting state based on some criteria
            if interesting_state is None or s.min_dist < interesting_state.min_dist:
                interesting_seed = seed
                interesting_state = s

        diff = time.perf_counter() - start
        summarize_performance(
            metrics=metrics,
            num_sims=num_sims,
            total_runtime=diff,
            false_alert_distance=false_alert_distance,
            nuisance_max_alert_time=nuisance_max_alert_time,
            nmac_distance=nmac_distance,
        )
    else:
        try:
            interesting_seed = int(seed_choice)
            if interesting_seed < 0:
                raise ValueError
        except ValueError:
            parser.error("--seed must be 'min' or a non-negative integer (e.g. 671).")

    # optional: do plot
    assert interesting_seed != -1

    init_vec, cmd_list, init_velo = make_random_input(interesting_seed, intruder_can_turn=intruder_can_turn)
    s = State(init_vec, init_velo[0], init_velo[1], save_states=True)
    s.simulate(cmd_list)

    d = round(s.min_dist, 2)
    print(f"\nSeed {interesting_seed} has min_dist {d}ft")
    print(
        f"Seed metrics: first_alert_time={s.sim_metrics['first_alert_time']}, "
        f"first_alert_range={None if s.sim_metrics['first_alert_range'] is None else round(s.sim_metrics['first_alert_range'], 2)}ft, "
        f"alert_lead_time_to_cpa={s.sim_metrics['alert_lead_time_to_cpa']}, "
        f"advisory_changes={s.sim_metrics['advisory_change_count']}, "
        f"reversals={s.sim_metrics['reversal_count']}, "
        f"alert_active_time={round(s.sim_metrics['alert_active_time'], 2)}s"
    )
    plot(s, save_mp4)

if __name__ == "__main__":
    main()
