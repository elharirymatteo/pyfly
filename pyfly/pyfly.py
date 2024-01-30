import numpy as np
import math
import scipy.integrate
import scipy.io
import os.path as osp
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec
from utils import Plot
from model_variables import Variable, ControlVariable, EnergyVariable, AttitudeQuaternion, Wind, Actuation


class PyFly:
    REQUIRED_VARIABLES = ["alpha", "beta", "roll", "pitch", "yaw", "omega_p", "omega_q", "omega_r", "position_n",
                          "position_e", "position_d", "velocity_u", "velocity_v", "velocity_w", "Va",
                          "elevator", "aileron", "rudder", "throttle"]

    def __init__(self,
                 config_path=osp.join(osp.dirname(__file__), "pyfly_config.json"),
                 parameter_path=osp.join(osp.dirname(__file__), "x8_param.mat"),
                 config_kw=None):
        """
        A flight simulator for fixed wing aircraft with configurable initial conditions, constraints and turbulence
        conditions

        :param config_path: (string) Path to json configuration file for simulator
        :param parameter_path: (string) Path to file containing required aircraft parameters
        """

        def set_config_attrs(parent, kws):
            for attr, val in kws.items():
                if isinstance(val, dict):
                    set_config_attrs(parent[attr], val)
                else:
                    parent[attr] = val

        _, parameter_extension = osp.splitext(parameter_path)
        if parameter_extension == ".mat":
            self.params = scipy.io.loadmat(parameter_path, squeeze_me=True)
        elif parameter_extension == ".json":
            with open(parameter_path) as param_file:
                self.params = json.load(param_file)
        else:
            raise Exception("Unsupported parameter file extension.")

        self.I = np.array([[self.params["Jx"], 0, -self.params["Jxz"]],
                           [0, self.params["Jy"], 0, ],
                           [-self.params["Jxz"], 0, self.params["Jz"]]
                           ])

        self.gammas = [self.I[0, 0] * self.I[2, 2] - self.I[0, 2] ** 2]
        self.gammas.append((np.abs(self.I[0, 2]) * (self.I[0, 0] - self.I[1, 1] + self.I[2, 2])) / self.gammas[0])
        self.gammas.append((self.I[2, 2] * (self.I[2, 2] - self.I[1, 1]) + self.I[0, 2] ** 2) / self.gammas[0])
        self.gammas.append(self.I[2, 2] / self.gammas[0])
        self.gammas.append(np.abs(self.I[0, 2]) / self.gammas[0])
        self.gammas.append((self.I[2, 2] - self.I[0, 0]) / self.I[1, 1])
        self.gammas.append(np.abs(self.I[0, 2]) / self.I[1, 1])
        self.gammas.append(((self.I[0, 0] - self.I[1, 1]) * self.I[0, 0] + self.I[0, 2] ** 2) / self.gammas[0])
        self.gammas.append(self.I[0, 0] / self.gammas[0])

        self.params["ar"] = self.params["b"] ** 2 / self.params["S_wing"]  # aspect ratio

        with open(config_path) as config_file:
            self.cfg = json.load(config_file)

        if config_kw is not None:
            set_config_attrs(self.cfg, config_kw)

        self.state = {}
        self.attitude_states = ["roll", "pitch", "yaw"]
        self.actuator_states = ["elevator", "aileron", "rudder", "throttle", "elevon_left", "elevon_right"]
        self.model_inputs = ["elevator", "aileron", "rudder", "throttle"]
        self.energy_states = []

        if not set(self.REQUIRED_VARIABLES).issubset([v["name"] for v in self.cfg["variables"]]):
            raise Exception("Missing required variable(s) in config file: {}".format(
                ",".join(list(set(self.REQUIRED_VARIABLES) - set([v["name"] for v in self.cfg["variables"]])))))

        self.dt = self.cfg["dt"]
        self.rho = self.cfg["rho"]
        self.g = self.cfg["g"]
        self.wind = Wind(mag_min=self.cfg["wind_magnitude_min"], mag_max=self.cfg["wind_magnitude_max"],
                         turbulence=self.cfg["turbulence"], turbulence_intensity=self.cfg["turbulence_intensity"],
                         sim_length=self.cfg.get("turbulence_sim_length", 300),
                         dt=self.cfg["dt"], b=self.params["b"])

        self.state["attitude"] = AttitudeQuaternion()
        self.attitude_states_with_constraints = []

        self.actuation = Actuation(model_inputs=self.model_inputs,
                                   actuator_inputs=self.cfg["actuation"]["inputs"],
                                   dynamics=self.cfg["actuation"]["dynamics"])

        for v in self.cfg["variables"]:
            if v["name"] in self.attitude_states and any([v.get(attribute, None) is not None for attribute in
                                                          ["constraint_min", "constraint_max", "value_min",
                                                           "value_max"]]):
                self.attitude_states_with_constraints.append(v["name"])
            if v["name"] in self.actuator_states:
                self.state[v["name"]] = ControlVariable(**v)
                self.actuation.add_state(self.state[v["name"]])
            elif "energy" in v["name"]:
                self.energy_states.append(v["name"])
                self.state[v["name"]] = EnergyVariable(mass=self.params["mass"], inertia_matrix=self.I, gravity=self.g, **v)
            else:
                self.state[v["name"]] = Variable(**v)

            if "wind" in v["name"]:
                self.wind.components.append(self.state[v["name"]])

        for state in self.model_inputs:
            if state not in self.state:
                self.state[state] = ControlVariable(name=state, disabled=True)
                self.actuation.add_state(self.state[state])

        self.actuation.finalize()

        for energy_state in self.energy_states:
            for req_var_name in self.state[energy_state].required_variables:
                self.state[energy_state].add_requirement(req_var_name, self.state[req_var_name])

        # TODO: check that all plotted variables are declared in cfg.variables
        self.plots = []
        for i, p in enumerate(self.cfg["plots"]):
            vars = p.pop("states")
            p["dt"] = self.dt
            p["id"] = i
            self.plots.append(Plot(**p))

            for v_name in vars:
                self.plots[-1].add_variable(self.state[v_name])

        self.cur_sim_step = None
        self.viewer = None

    def seed(self, seed=None):
        """
        Seed the random number generator of the flight simulator

        :param seed: (int) seed for random state
        """
        for i, var in enumerate(self.state.values()):
            var.seed(seed + i)

        self.wind.seed(seed)

    def reset(self, state=None, turbulence_noise=None):
        """
        Reset state of simulator. Must be called before first use.

        :param state: (dict) set initial value of states to given value.
        """
        self.cur_sim_step = 0

        for name, var in self.state.items():
            if name in ["Va", "alpha", "beta", "attitude"] or "wind" in name or "energy" in name or isinstance(var, ControlVariable):
                continue
            var_init = state[name] if state is not None and name in state else None
            var.reset(value=var_init)

        self.actuation.reset(state)

        wind_init = None
        if state is not None:
            if "wind" in state:
                wind_init = state["wind"]
            elif all([comp in state for comp in ["wind_n", "wind_e", "wind_d"]]):
                wind_init = [state["wind_n"], state["wind_e"], state["wind_d"]]
        self.wind.reset(wind_init, turbulence_noise)

        Theta = self.get_states_vector(["roll", "pitch", "yaw"])
        vel = np.array(self.get_states_vector(["velocity_u", "velocity_v", "velocity_w"]))

        Va, alpha, beta = self._calculate_airspeed_factors(Theta, vel)
        self.state["Va"].reset(Va)
        self.state["alpha"].reset(alpha)
        self.state["beta"].reset(beta)

        self.state["attitude"].reset(Theta)

        for energy_state in self.energy_states:
            self.state[energy_state].reset(self.state[energy_state].calculate_value())

    def render(self, mode="plot", close=False, viewer=None, targets=None, block=False):
        """
        Visualize history of simulator states.

        :param mode: (str) render mode, one of plot for graph representation and animation for 3D animation with blender
        :param close: (bool) close figure after showing
        :param viewer: (dict) viewer object with figure and gridspec that pyfly will attach plots to
        :param targets: (dict) string list pairs of states with target values added to plots containing these states.
        """
        if mode == "plot":
            if viewer is not None:
                self.viewer = viewer
            elif self.viewer is None:
                self.viewer = {"fig": plt.figure(figsize=(9, 16))}
                subfig_count = len(self.plots)
                self.viewer["gs"] = matplotlib.gridspec.GridSpec(subfig_count, 1)

            for i, p in enumerate(self.plots):
                sub_fig = self.viewer["gs"][i, 0] if p.axis is None else None
                if targets is not None:
                    plot_variables = [v.name for v in p.variables]
                    plot_targets = list(set(targets).intersection(plot_variables))
                    p.plot(fig=sub_fig, targets={k: v for k, v in targets.items() if k in plot_targets})
                else:
                    p.plot(fig=sub_fig)

            if viewer is None:
                plt.show(block=block)
            if close:
                for p in self.plots:
                    p.close()
                self.viewer = None
        elif mode == "animation":
            raise NotImplementedError
        else:
            raise ValueError("Unexpected value {} for mode".format(mode))

    def step(self, commands):
        """
        Perform one integration step from t to t + dt.

        :param commands: ([float]) actuator setpoints
        :return: (bool, dict) whether integration step was successfully performed, reason for step failure
        """
        success = True
        info = {}

        # Record command history and apply conditions on actuator setpoints
        control_inputs = self.actuation.set_and_constrain_commands(commands)

        y0 = list(self.state["attitude"].value)
        y0.extend(self.get_states_vector(["omega_p", "omega_q", "omega_r", "position_n", "position_e", "position_d",
                                          "velocity_u", "velocity_v", "velocity_w"]))
        y0.extend(self.actuation.get_values())
        y0 = np.array(y0)

        try:
            sol = scipy.integrate.solve_ivp(fun=lambda t, y: self._dynamics(t, y), t_span=(0, self.dt),
                                            y0=y0)
            self._set_states_from_ode_solution(sol.y[:, -1], save=True)

            Theta = self.get_states_vector(["roll", "pitch", "yaw"])
            vel = np.array(self.get_states_vector(["velocity_u", "velocity_v", "velocity_w"]))

            Va, alpha, beta = self._calculate_airspeed_factors(Theta, vel)
            self.state["Va"].set_value(Va)
            self.state["alpha"].set_value(alpha)
            self.state["beta"].set_value(beta)

            for energy_state in self.energy_states:
                self.state[energy_state].set_value(self.state[energy_state].calculate_value(), save=True)

            self.wind.set_value(self.cur_sim_step)
        except ConstraintException as e:
            success = False
            info = {"termination": e.variable}

        self.cur_sim_step += 1

        return success, info

    def get_states_vector(self, states, attribute="value"):
        """
        Get attribute of multiple states.

        :param states: ([string]) list of state names
        :param attribute: (string) state attribute to retrieve
        :return: ([?]) list of attribute for each state
        """
        return [getattr(self.state[state_name], attribute) for state_name in states]

    def save_history(self, path, states):
        """
        Save simulator state history to file.

        :param path: (string) path to save history to
        :param states: (string or [string]) names of states to save
        """
        res = {}
        if states == "all":
            save_states = self.state.keys()
        else:
            save_states = states

        for state in save_states:
            res[state] = self.state[state].history

        np.save(path, res)

    def _dynamics(self, t, y, control_sp=None):
        """
        Right hand side of dynamics differential equation.

        :param t: (float) current integration time
        :param y: ([float]) current values of integration states
        :param control_sp: ([float]) setpoints for actuators
        :return: ([float]) right hand side of differential equations
        """

        if t > 0:
            self._set_states_from_ode_solution(y, save=False)

        attitude = y[:4]

        omega = self.get_states_vector(["omega_p", "omega_q", "omega_r"])
        vel = np.array(self.get_states_vector(["velocity_u", "velocity_v", "velocity_w"]))
        u_states = self.get_states_vector(self.model_inputs)

        f, tau = self._forces(attitude, omega, vel, u_states)

        return np.concatenate([
            self._f_attitude_dot(t, attitude, omega),
            self._f_omega_dot(t, omega, tau),
            self._f_p_dot(t, vel, attitude),
            self._f_v_dot(t, vel, omega, f),
            self._f_u_dot(t, control_sp)
        ])

    def _forces(self, attitude, omega, vel, controls):
        """
        Get aerodynamic forces acting on aircraft.

        :param attitude: ([float]) attitude quaternion of aircraft
        :param omega: ([float]) angular velocity of aircraft
        :param vel: ([float]) linear velocity of aircraft
        :param controls: ([float]) state of actutators
        :return: ([float], [float]) forces and moments in x, y, z of aircraft frame
        """
        elevator, aileron, rudder, throttle = controls

        p, q, r = omega

        if self.wind.turbulence:
            p_w, q_w, r_w = self.wind.get_turbulence_angular(self.cur_sim_step)
            p, q, r = p - p_w, q - q_w, r - r_w

        Va, alpha, beta = self._calculate_airspeed_factors(attitude, vel)

        Va = self.state["Va"].apply_conditions(Va)
        alpha = self.state["alpha"].apply_conditions(alpha)
        beta = self.state["beta"].apply_conditions(beta)

        pre_fac = 0.5 * self.rho * Va ** 2 * self.params["S_wing"]

        e0, e1, e2, e3 = attitude
        fg_b = self.params["mass"] * self.g * np.array([2 * (e1 * e3 - e2 * e0),
                                                        2 * (e2 * e3 + e1 * e0),
                                                        e3 ** 2 + e0 ** 2 - e1 ** 2 - e2 ** 2])

        C_L_alpha_lin = self.params["C_L_0"] + self.params["C_L_alpha"] * alpha

        # Nonlinear version of lift coefficient with stall
        a_0 = self.params["a_0"]
        M = self.params["M"]
        e = self.params["e"]  # oswald efficiency
        ar = self.params["ar"]
        C_D_p = self.params["C_D_p"]
        C_m_fp = self.params["C_m_fp"]
        C_m_alpha = self.params["C_m_alpha"]
        C_m_0 = self.params["C_m_0"]

        sigma = (1 + np.exp(-M * (alpha - a_0)) + np.exp(M * (alpha + a_0))) / (
                    (1 + np.exp(-M * (alpha - a_0))) * (1 + np.exp(M * (alpha + a_0))))
        C_L_alpha = (1 - sigma) * C_L_alpha_lin + sigma * (2 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha))

        f_lift_s = pre_fac * (C_L_alpha + self.params["C_L_q"] * self.params["c"] / (2 * Va) * q + self.params[
            "C_L_delta_e"] * elevator)
        # C_D_alpha = self.params["C_D_0"] + self.params["C_D_alpha1"] * alpha + self.params["C_D_alpha2"] * alpha ** 2
        C_D_alpha = C_D_p + (1 - sigma) * (self.params["C_L_0"] + self.params["C_L_alpha"] * alpha) ** 2 / (
                    np.pi * e * ar) + sigma * (2 * np.sign(alpha) * math.pow(np.sin(alpha), 3))
        C_D_beta = self.params["C_D_beta1"] * beta + self.params["C_D_beta2"] * beta ** 2
        f_drag_s = pre_fac * (
                    C_D_alpha + C_D_beta + self.params["C_D_q"] * self.params["c"] / (2 * Va) * q + self.params[
                "C_D_delta_e"] * elevator ** 2)

        C_m = (1 - sigma) * (C_m_0 + C_m_alpha * alpha) + sigma * (C_m_fp * np.sign(alpha) * np.sin(alpha) ** 2)
        m = pre_fac * self.params["c"] * (C_m + self.params["C_m_q"] * self.params["b"] / (2 * Va) * q + self.params[
            "C_m_delta_e"] * elevator)

        f_y = pre_fac * (
                    self.params["C_Y_0"] + self.params["C_Y_beta"] * beta + self.params["C_Y_p"] * self.params["b"] / (
                        2 * Va) * p + self.params["C_Y_r"] * self.params["b"] / (2 * Va) * r + self.params[
                        "C_Y_delta_a"] * aileron + self.params["C_Y_delta_r"] * rudder)
        l = pre_fac * self.params["b"] * (
                    self.params["C_l_0"] + self.params["C_l_beta"] * beta + self.params["C_l_p"] * self.params["b"] / (
                        2 * Va) * p + self.params["C_l_r"] * self.params["b"] / (2 * Va) * r + self.params[
                        "C_l_delta_a"] * aileron + self.params["C_l_delta_r"] * rudder)
        n = pre_fac * self.params["b"] * (
                    self.params["C_n_0"] + self.params["C_n_beta"] * beta + self.params["C_n_p"] * self.params["b"] / (
                        2 * Va) * p + self.params["C_n_r"] * self.params["b"] / (2 * Va) * r + self.params[
                        "C_n_delta_a"] * aileron + self.params["C_n_delta_r"] * rudder)

        f_aero = np.dot(self._rot_b_v(np.array([0, alpha, beta])), np.array([-f_drag_s, f_y, -f_lift_s]))
        tau_aero = np.array([l, m, n])

        Vd = Va + throttle * (self.params["k_motor"] - Va)
        f_prop = np.array([0.5 * self.rho * self.params["S_prop"] * self.params["C_prop"] * Vd * (Vd - Va), 0, 0])
        tau_prop = np.array([-self.params["k_T_P"] * (self.params["k_Omega"] * throttle) ** 2, 0, 0])

        f = f_prop + fg_b + f_aero
        tau = tau_aero + tau_prop

        return f, tau

    def _f_attitude_dot(self, t, attitude, omega):
        """
        Right hand side of quaternion attitude differential equation.

        :param t: (float) time of integration
        :param attitude: ([float]) attitude quaternion
        :param omega: ([float]) angular velocity
        :return: ([float]) right hand side of quaternion attitude differential equation.
        """
        p, q, r = omega
        T = np.array([[0, -p, -q, -r],
                      [p, 0, r, -q],
                      [q, -r, 0, p],
                      [r, q, -p, 0]
                      ])
        return 0.5 * np.dot(T, attitude)

    def _f_omega_dot(self, t, omega, tau):
        """
        Right hand side of angular velocity differential equation.

        :param t: (float) time of integration
        :param omega: ([float]) angular velocity
        :param tau: ([float]) moments acting on aircraft
        :return: ([float]) right hand side of angular velocity differential equation.
        """
        return np.array([
            self.gammas[1] * omega[0] * omega[1] - self.gammas[2] * omega[1] * omega[2] + self.gammas[3] * tau[0] +
            self.gammas[4] * tau[2],
            self.gammas[5] * omega[0] * omega[2] - self.gammas[6] * (omega[0] ** 2 - omega[2] ** 2) + tau[1] / self.I[
                1, 1],
            self.gammas[7] * omega[0] * omega[1] - self.gammas[1] * omega[1] * omega[2] + self.gammas[4] * tau[0] +
            self.gammas[8] * tau[2]
        ])

    def _f_v_dot(self, t, v, omega, f):
        """
        Right hand side of linear velocity differential equation.

        :param t: (float) time of integration
        :param v: ([float]) linear velocity
        :param omega: ([float]) angular velocity
        :param f: ([float]) forces acting on aircraft
        :return: ([float]) right hand side of linear velocity differntial equation.
        """
        v_dot = np.array([
            omega[2] * v[1] - omega[1] * v[2] + f[0] / self.params["mass"],
            omega[0] * v[2] - omega[2] * v[0] + f[1] / self.params["mass"],
            omega[1] * v[0] - omega[0] * v[1] + f[2] / self.params["mass"]
        ])

        return v_dot

    def _f_p_dot(self, t, v, attitude):
        """
        Right hand side of position differential equation.

        :param t: (float) time of integration
        :param v: ([float]) linear velocity
        :param attitude: ([float]) attitude quaternion
        :return: ([float]) right hand side of position differntial equation.
        """
        e0, e1, e2, e3 = attitude
        T = np.array([[e1 ** 2 + e0 ** 2 - e2 ** 2 - e3 ** 2, 2 * (e1 * e2 - e3 * e0), 2 * (e1 * e3 + e2 * e0)],
                      [2 * (e1 * e2 + e3 * e0), e2 ** 2 + e0 ** 2 - e1 ** 2 - e3 ** 2, 2 * (e2 * e3 - e1 * e0)],
                      [2 * (e1 * e3 - e2 * e0), 2 * (e2 * e3 + e1 * e0), e3 ** 2 + e0 ** 2 - e1 ** 2 - e2 ** 2]
                      ])
        return np.dot(T, v)

    def _f_u_dot(self, t, setpoints):
        """
        Right hand side of actuator differential equation.

        :param t: (float) time of integration
        :param setpoints: ([float]) setpoint for actuators
        :return: ([float]) right hand side of actuator differential equation.
        """
        return self.actuation.rhs(setpoints)

    def _rot_b_v(self, attitude):
        """
        Rotate vector from body frame to vehicle frame.

        :param Theta: ([float]) vector to rotate, either as Euler angles or quaternion
        :return: ([float]) rotated vector
        """
        if len(attitude) == 3:
            phi, th, psi = attitude
            return np.array([
                [np.cos(th) * np.cos(psi), np.cos(th) * np.sin(psi), -np.sin(th)],
                [np.sin(phi) * np.sin(th) * np.cos(psi) - np.cos(phi) * np.sin(psi),
                 np.sin(phi) * np.sin(th) * np.sin(psi) + np.cos(phi) * np.cos(psi), np.sin(phi) * np.cos(th)],
                [np.cos(phi) * np.sin(th) * np.cos(psi) + np.sin(phi) * np.sin(psi),
                 np.cos(phi) * np.sin(th) * np.sin(psi) - np.sin(phi) * np.cos(psi), np.cos(phi) * np.cos(th)]
            ])
        elif len(attitude) == 4:
            e0, e1, e2, e3 = attitude
            return np.array([[-1 + 2 * (e0 ** 2 + e1 ** 2), 2 * (e1 * e2 + e3 * e0), 2 * (e1 * e3 - e2 * e0)],
                             [2 * (e1 * e2 - e3 * e0), -1 + 2 * (e0 ** 2 + e2 ** 2), 2 * (e2 * e3 + e1 * e0)],
                             [2 * (e1 * e3 + e2 * e0), 2 * (e2 * e3 - e1 * e0), -1 + 2 * (e0 ** 2 + e3 ** 2)]])

        else:
            raise ValueError("Attitude is neither Euler angles nor Quaternion")

    def _rot_v_b(self, Theta):
        """
        Rotate vector from vehicle frame to body frame.

        :param Theta: ([float]) vector to rotate
        :return: ([float]) rotated vector
        """
        phi, th, psi = Theta
        return np.array([
            [np.cos(th) * np.cos(psi), np.sin(phi) * np.sin(th) * np.cos(psi) - np.cos(phi) * np.sin(psi),
             np.cos(phi) * np.sin(th) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
            [np.cos(th) * np.sin(psi), np.sin(phi) * np.sin(th) * np.sin(psi) + np.cos(phi) * np.cos(psi),
             np.cos(phi) * np.sin(th) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
            [-np.sin(th), np.sin(phi) * np.cos(th), np.cos(phi) * np.cos(th)]
        ])

    def _calculate_airspeed_factors(self, attitude, vel):
        """
        Calculate the airspeed factors airspeed (Va), angle of attack (alpha) and sideslip angle (beta).

        :param attitude: ([float]) attitude quaternion
        :param vel: ([float]) linear velocity
        :return: ([float]) airspeed factors Va, alpha, beta
        """
        if self.wind.turbulence:
            turbulence = self.wind.get_turbulence_linear(self.cur_sim_step)
        else:
            turbulence = np.zeros(3)

        wind_vec = np.dot(self._rot_b_v(attitude), self.wind.steady) + turbulence
        airspeed_vec = vel - wind_vec

        Va = np.linalg.norm(airspeed_vec)
        alpha = np.arctan2(airspeed_vec[2], airspeed_vec[0])
        beta = np.arcsin(airspeed_vec[1] / Va)

        return Va, alpha, beta

    def _set_states_from_ode_solution(self, ode_sol, save):
        """
        Set states from ODE solution vector.

        :param ode_sol: ([float]) solution vector from ODE solver
        :param save: (bool) whether to save values to state history, i.e. whether solution represents final step
        solution or intermediate values during integration.
        """
        self.state["attitude"].set_value(ode_sol[:4] / np.linalg.norm(ode_sol[:4]))
        if save:
            euler_angles = self.state["attitude"].as_euler_angle()
            self.state["roll"].set_value(euler_angles["roll"], save=save)
            self.state["pitch"].set_value(euler_angles["pitch"], save=save)
            self.state["yaw"].set_value(euler_angles["yaw"], save=save)
        else:
            for state in self.attitude_states_with_constraints:
                self.state[state].set_value(self.state["attitude"].as_euler_angle(state))
        start_i = 4

        self.state["omega_p"].set_value(ode_sol[start_i], save=save)
        self.state["omega_q"].set_value(ode_sol[start_i + 1], save=save)
        self.state["omega_r"].set_value(ode_sol[start_i + 2], save=save)
        self.state["position_n"].set_value(ode_sol[start_i + 3], save=save)
        self.state["position_e"].set_value(ode_sol[start_i + 4], save=save)
        self.state["position_d"].set_value(ode_sol[start_i + 5], save=save)
        self.state["velocity_u"].set_value(ode_sol[start_i + 6], save=save)
        self.state["velocity_v"].set_value(ode_sol[start_i + 7], save=save)
        self.state["velocity_w"].set_value(ode_sol[start_i + 8], save=save)
        self.actuation.set_states(ode_sol[start_i + 9:], save=save)


if __name__ == "__main__":
    from pid_controller import PIDController
    from dryden import DrydenGustModel
    pfly = PyFly("pyfly/pyfly_config.json", "pyfly/x8_param.mat")
    pfly.seed(0)

    pid = PIDController(pfly.dt)
    pid.set_reference(phi=0.2, theta=0, va=22)

    pfly.reset(state={"roll": -0.5, "pitch": 0.15})

    horizon = int(1/pfly.cfg["dt"]) * 5
    for i in range(horizon):
        phi = pfly.state["roll"].value
        theta = pfly.state["pitch"].value
        Va = pfly.state["Va"].value
        omega = [pfly.state["omega_p"].value, pfly.state["omega_q"].value, pfly.state["omega_r"].value]

        action = pid.get_action(phi, theta, Va, omega)
        success, step_info = pfly.step(action)

        if not success:
            break

    pfly.render(block=True)
else:
    from dryden import DrydenGustModel
