import numpy as np
import matplotlib.pyplot as plt
from dryden import DrydenGustModel


class ConstraintException(Exception):
    def __init__(self, variable, value, limit):
        self.message = "Constraint on {} violated ({}/{})".format(variable, value, limit)
        self.variable = variable
        

class Variable:
    def __init__(self, name, value_min=None, value_max=None, init_min=None, init_max=None, constraint_min=None,
                 constraint_max=None, convert_to_radians=False, unit=None, label=None, wrap=False):
        """
        PyFly state object managing state history, constraints and visualizations.

        :param name: (string) name of state
        :param value_min: (float) lowest possible value of state, values will be clipped to this limit
        :param value_max: (float) highest possible value of state, values will be clipped to this limit
        :param init_min: (float) lowest possible initial value of state
        :param init_max: (float) highest possible initial value of state
        :param constraint_min: (float) lower constraint of state, which if violated will raise ConstraintException
        :param constraint_max: (float) upper constraint of state, which if violated will raise ConstraintException
        :param convert_to_radians: (bool) whether to convert values for attributes from configuration file from degrees
        to radians
        :param unit: (string) unit of the state, for plotting purposes
        :param label: (string) label given to state in plots
        :param wrap: (bool) whether to wrap state value in region [-pi, pi]
        """
        self.value_min = value_min
        self.value_max = value_max

        self.init_min = init_min if init_min is not None else value_min
        self.init_max = init_max if init_max is not None else value_max

        self.constraint_min = constraint_min
        self.constraint_max = constraint_max

        if convert_to_radians:
            for attr_name, val in self.__dict__.items():
                if val is not None:
                    setattr(self, attr_name, np.radians(val))

        self.name = name

        self.value = None

        self.wrap = wrap

        self.unit = unit
        self.label = label if label is not None else self.name
        self.lines = {"self": None}
        self.target_lines = {"self": None}
        self.target_bounds = {"self": None}

        self.np_random = None
        self.seed()

        self.history = None

    def reset(self, value=None):
        """
        Reset object to initial state.

        :param value: (float) initial value of state
        """
        self.history = []

        if value is None:
            try:
                value = self.np_random.uniform(self.init_min, self.init_max)
            except TypeError:
                raise Exception("Variable init_min and init_max can not be None if no value is provided on reset")
        else:
            value = self.apply_conditions(value)

        self.value = value

        self.history.append(value)

    def seed(self, seed=None):
        """
        Seed random number generator of state

        :param seed: (int) seed of random state
        """
        self.np_random = np.random.RandomState(seed)

    def apply_conditions(self, value):
        """
        Apply state limits and constraints to value. Will raise ConstraintException if constraints are violated

        :param value: (float) value to which limits and constraints are applied
        :return: (float) value after applying limits and constraints
        """
        if self.constraint_min is not None and value < self.constraint_min:
            raise ConstraintException(self.name, value, self.constraint_min)

        if self.constraint_max is not None and value > self.constraint_max:
            raise ConstraintException(self.name, value, self.constraint_max)

        if self.value_min is not None or self.value_max is not None:
            value = np.clip(value, self.value_min, self.value_max)

        if self.wrap and np.abs(value) > np.pi:
            value = np.sign(value) * (np.abs(value) % np.pi - np.pi)

        return value

    def set_value(self, value, save=True):
        """
        Set value of state, after applying limits and constraints to value. Raises ConstraintException if constraints
        are violated

        :param value: (float) new value of state
        :param save: (bool) whether to commit value to history of state
        """
        value = self.apply_conditions(value)

        if save:
            self.history.append(value)

        self.value = value

    def plot(self, axis=None, y_unit=None, target=None, plot_id=None, **plot_kw):
        """
        Plot state history.

        :param axis: (matplotlib.pyplot.axis or None) axis object to plot state to. If None create new axis
        :param y_unit: (string) unit state should be plotted in, will convert values if different from internal
        representation
        :param target: (list) target values for state, must be of equal size to state history
        :param plot_id: (string or int or None) identifier of parent plot object. Allows state to plot to multiple
        figures at a time.
        :param plot_kw: (dict) plot keyword arguments passed to matplotlib.pyplot.plot
        """

        def linear_scaling(val, old_min, old_max, new_min, new_max):
            return (new_max - np.sign(old_min) * (- new_min)) / (old_max - old_min) * (
                        np.array(val) - old_max) + new_max

        if y_unit is None:
            y_unit = self.unit if y_unit is None else y_unit

        x, y = self._get_plot_x_y_data()
        if "degrees" in y_unit:
            y = np.degrees(y)
            if target is not None:
                target["data"] = np.degrees(target["data"])
                if "bound" in target:
                    target["bound"] = np.degrees(target["bound"])
        elif y_unit == "%":  # TODO: scale positive according to positive limit and negative according to lowest minimum value
            y = linear_scaling(y, self.value_min, self.value_max, -100, 100)
            if target is not None:
                target["data"] = linear_scaling(target["data"], self.value_min, self.value_max, -100, 100)
                if "bound" in target:
                    target["bound"] = linear_scaling(target["bound"], self.value_min, self.value_max, -100, 100)
        else:
            y = y

        plot_object = axis
        if axis is None:
            plot_object = plt
            plot_id = "self"
            fig_kw = {"title": self.name, "ylabel": y_unit}

        if self.lines.get(plot_id, None) is None:
            line, = plot_object.plot(x, y, label=self.label, **plot_kw)
            self.lines[plot_id] = line

            if target is not None:
                tar_line, = plot_object.plot(x, target["data"], color=self.lines[plot_id].get_color(), linestyle="dashed",
                                             marker="x", markevery=0.2)

                if "bound" in target:
                    tar_bound = plot_object.fill_between(np.arange(target["bound"].shape[0]),
                                                         target["data"] + target["bound"],
                                                         target["data"] - target["bound"], alpha=0.15,
                                                         facecolor=self.lines[plot_id].get_color()
                                                        )
                    self.target_bounds[plot_id] = tar_bound
                self.target_lines[plot_id] = tar_line
        else:
            self.lines[plot_id].set_data(x, y)
            if target is not None:
                self.target_lines[plot_id].set_data(x, target)
                if "bound" in target:  # TODO: fix this?
                    self.target_bounds[plot_id].set_data(np.arange(target["bound"].shape[0]),
                                                         target["data"] + target["bound"],
                                                         target["data"] - target["bound"])
        if axis is None:
            for k, v in fig_kw.items():
                getattr(plot_object, format(k))(v)
            plt.show()

    def close_plot(self, plot_id="self"):
        """
        Close plot with id plot_id.

        :param plot_id: (string or int) identifier of parent plot object
        """
        self.lines[plot_id] = None
        self.target_lines[plot_id] = None
        self.target_bounds[plot_id] = None

    def _get_plot_x_y_data(self):
        """
        Get plot data from variable history.

        :return: ([int], [float]) x plot data, y plot data
        """
        x = list(range(len(self.history)))
        y = self.history
        return x, y


class ControlVariable(Variable):
    def __init__(self, order=None, tau=None, omega_0=None, zeta=None, dot_max=None, disabled=False, **kwargs):
        """
        PyFly actuator state variable.

        :param order: (int) order of state transfer function
        :param tau: (float) time constant for first order transfer functions
        :param omega_0: (float) undamped natural frequency of second order transfer functions
        :param zeta: (float) damping factor of second order transfer function
        :param dot_max: (float) constraint on magnitude of derivative of second order transfer function
        :param disabled: (bool) if actuator is disabled for aircraft, e.g. aircraft has no rudder
        :param kwargs: (dict) keyword arguments for Variable class
        """
        assert (disabled or (order == 1 or order == 2))
        super().__init__(**kwargs)
        self.order = order
        self.tau = tau
        self.omega_0 = omega_0
        self.zeta = zeta
        self.dot_max = dot_max

        if order == 1:
            assert (tau is not None)
            self.coefs = [[-1 / self.tau, 0, 1 / self.tau], [0, 0, 0]]
        elif order == 2:
            assert (omega_0 is not None and zeta is not None)
            self.coefs = [[0, 1, 0], [-self.omega_0 ** 2, -2 * self.zeta * self.omega_0, self.omega_0 ** 2]]
        self.dot = None
        self.command = None
        self.disabled = disabled
        if self.disabled:
            self.value = 0
        self.plot_quantity = "value"

    def apply_conditions(self, values):
        """
        Apply state limits and constraints to value. Will raise ConstraintException if constraints are violated

        :param value: (float) value to which limits and constraints is applied
        :return: (float) value after applying limits and constraints
        """
        try:
            value, dot = values
        except:
            value, dot = values, 0
        value = super().apply_conditions(value)

        if self.dot_max is not None:
            dot = np.clip(dot, -self.dot_max, self.dot_max)

        return [value, dot]

    def set_command(self, command):
        """
        Set setpoint for actuator and commit to history of state

        :param command: setpoint for actuator
        """
        command = super().apply_conditions(command)
        self.command = command
        self.history["command"].append(command)

    def reset(self, value=None):
        """
        Reset object to initial state.

        :param value: (list) initial value, derivative and setpoint of state
        """
        self.history = {"value": [], "dot": [], "command": []}

        if not self.disabled:
            if value is None:
                value = self.np_random.uniform(self.init_min, self.init_max), 0
            else:
                value = self.apply_conditions(value)

            self.value = value[0]
            self.dot = value[1]
            command = None
            self.command = command
        else:
            value, dot, command = 0, 0, None
            self.value = value
            self.dot = dot
            self.command = command

        self.history["value"].append(self.value)
        self.history["dot"].append(self.dot)

    def set_value(self, value, save=True):
        """
        Set value of state, after applying limits and constraints to value. Raises ConstraintException if constraints
        are violated

        :param value: (float) new value and derivative of state
        :param save: (bool) whether to commit value to history of state
        """
        value, dot = self.apply_conditions(value)

        self.value = value
        self.dot = dot

        if save:
            self.history["value"].append(value)
            self.history["dot"].append(dot)

    def _get_plot_x_y_data(self):
        """
        Get plot data from variable history, for the quantity designated by the attribute plot_quantity.

        :return: ([int], [float]) x plot data, y plot data
        """
        y = self.history[self.plot_quantity]
        x = list(range(len(y)))
        return x, y

    def get_coeffs(self):
        if self.order == 1:
            return
        else:
            return []


class EnergyVariable(Variable):
    def __init__(self, mass=None, inertia_matrix=None, gravity=None, **kwargs):
        super().__init__(**kwargs)
        self.required_variables = []
        self.variables = {}
        if self.name == "energy_potential" or self.name == "energy_total":
            assert(mass is not None and gravity is not None)
            self.mass = mass
            self.gravity = gravity
            self.required_variables.append("position_d")
        if self.name == "energy_kinetic" or self.name == "energy_total":
            assert (mass is not None and inertia_matrix is not None)
            self.mass = mass
            self.inertia_matrix = inertia_matrix
            self.required_variables.extend(["Va", "omega_p", "omega_q", "omega_r"])
        if self.name == "energy_kinetic_rotational":
            assert(inertia_matrix is not None)
            self.inertia_matrix = inertia_matrix
            self.required_variables.extend(["omega_p", "omega_q", "omega_r"])
        if self.name == "energy_kinetic_translational":
            assert(mass is not None)
            self.mass = mass
            self.required_variables.append("Va")

    def add_requirement(self, name, variable):
        self.variables[name] = variable

    def calculate_value(self):
        val = 0
        if self.name == "energy_potential" or self.name == "energy_total":
            val += self.mass * self.gravity * (-self.variables["position_d"].value)
        if self.name == "energy_kinetic_rotational" or self.name == "energy_kinetic" or self.name == "energy_total":
            for i, axis in enumerate(["omega_p", "omega_q", "omega_r"]):
                m_i = self.inertia_matrix[i, i]
                val += 1 / 2 * m_i * self.variables[axis].value ** 2
        if self.name == "energy_kinetic_translational" or self.name == "energy_kinetic" or self.name == "energy_total":
            val += 1 / 2 * self.mass * self.variables["Va"].value ** 2

        return val


class Actuation:
    def __init__(self, model_inputs, actuator_inputs, dynamics):
        """
        PyFly actuation object, responsible for verifying validity of configured actuator model, processing inputs and
        actuator dynamics.

        :param model_inputs: ([string]) the states used by PyFly as inputs to dynamics
        :param actuator_inputs: ([string]) the user configured actuator input states
        :param dynamics: ([string]) the user configured actuator states to simulate dynamics for
        """
        self.states = {}
        self.coefficients = [[np.array([]) for _ in range(3)] for __ in range(2)]
        self.elevon_dynamics = False
        self.dynamics = dynamics
        self.inputs = actuator_inputs
        self.model_inputs = model_inputs
        self.input_indices = {s: i for i, s in enumerate(actuator_inputs)}
        self.dynamics_indices = {s: i for i, s in enumerate(dynamics)}

    def set_states(self, values, save=True):
        """
        Set values of actuator states.

        :param values: ([float]) list of state values + list of state derivatives
        :param save: (bool) whether to commit values to state history
        :return:
        """
        for i, state in enumerate(self.dynamics):
            self.states[state].set_value((values[i], values[len(self.dynamics) + i]), save=save)

        # Simulator model operates on elevator and aileron angles, if aircraft has elevon dynamics need to map
        if self.elevon_dynamics:
            elevator, aileron = self._map_elevon_to_elevail(er=self.states["elevon_right"].value,
                                                            el=self.states["elevon_left"].value)

            self.states["aileron"].set_value((aileron, 0), save=save)
            self.states["elevator"].set_value((elevator, 0), save=save)

    def add_state(self, state):
        """
        Add actuator state, and configure dynamics if state has dynamics.

        :param state: (ControlVariable) actuator state
        :return:
        """
        self.states[state.name] = state
        if state.name in self.dynamics:
            for i in range(2):
                for j in range(3):
                    self.coefficients[i][j] = np.append(self.coefficients[i][j], state.coefs[i][j])

    def get_values(self):
        """
        Get state values and derivatives for states in actuator dynamics.

        :return: ([float]) list of state values + list of state derivatives
        """
        return [self.states[state].value for state in self.dynamics] + [self.states[state].dot for state in
                                                                            self.dynamics]

    def rhs(self, setpoints=None):
        """
        Right hand side of actuator differential equation.

        :param setpoints: ([float] or None) setpoints for actuators. If None, setpoints are set as the current command
        of the dynamics variable
        :return: ([float]) right hand side of actuator differential equation.
        """
        if setpoints is None:
            setpoints = [self.states[state].command for state in self.dynamics]
        states = [self.states[state].value for state in self.dynamics]
        dots = [self.states[state].dot for state in self.dynamics]
        dot = np.multiply(states,
                          self.coefficients[0][0]) + np.multiply(setpoints,
                                                                 self.coefficients[0][2]) + np.multiply(dots, self.coefficients[0][1])
        ddot = np.multiply(states,
                           self.coefficients[1][0]) + np.multiply(setpoints,
                                                                  self.coefficients[1][2]) + np.multiply(dots, self.coefficients[1][1])

        return np.concatenate((dot, ddot))

    def set_and_constrain_commands(self, commands):
        """
        Take  raw actuator commands and constrain them according to the state limits and constraints, and update state
        values and history.

        :param commands: ([float]) raw commands
        :return: ([float]) constrained commands
        """
        dynamics_commands = {}
        if self.elevon_dynamics and "elevator" and "aileron" in self.inputs:
            elev_c, ail_c = commands[self.input_indices["elevator"]], commands[self.input_indices["aileron"]]
            elevon_r_c, elevon_l_c = self._map_elevail_to_elevon(elev=elev_c, ail=ail_c)
            dynamics_commands = {"elevon_right": elevon_r_c, "elevon_left": elevon_l_c}

        for state in self.dynamics:
            if state in self.input_indices:
                state_c = commands[self.input_indices[state]]
            else:  # Elevail inputs with elevon dynamics
                state_c = dynamics_commands[state]
            self.states[state].set_command(state_c)
            dynamics_commands[state] = self.states[state].command

        # The elevator and aileron commands constrained by limitatons on physical elevons
        if self.elevon_dynamics:
            elev_c, ail_c = self._map_elevon_to_elevail(er=dynamics_commands["elevon_right"],
                                                        el=dynamics_commands["elevon_left"])
            self.states["elevator"].set_command(elev_c)
            self.states["aileron"].set_command(ail_c)

        for state, i in self.input_indices.items():
            commands[i] = self.states[state].command

        return commands

    def finalize(self):
        """
        Assert valid configuration of actuator dynamics and set actuator state limits if applicable.
        """
        if "elevon_left" in self.dynamics or "elevon_right" in self.dynamics:
            assert("elevon_left" in self.dynamics and "elevon_right" in self.dynamics and not ("aileron" in self.dynamics
                   or "elevator" in self.dynamics))
            assert ("elevon_left" in self.states and "elevon_right" in self.states)
            self.elevon_dynamics = True

            # Set elevator and aileron limits from elevon limits for plotting purposes etc.
            if "elevator" in self.states:
                elev_min, _ = self._map_elevon_to_elevail(er=self.states["elevon_right"].value_min,
                                                          el=self.states["elevon_left"].value_min)
                elev_max, _ = self._map_elevon_to_elevail(er=self.states["elevon_right"].value_max,
                                                          el=self.states["elevon_left"].value_max)
                self.states["elevator"].value_min = elev_min
                self.states["elevator"].value_max = elev_max
            if "aileron" in self.states:
                _, ail_min = self._map_elevon_to_elevail(er=self.states["elevon_right"].value_max,
                                                         el=self.states["elevon_left"].value_min)
                _, ail_max = self._map_elevon_to_elevail(er=self.states["elevon_right"].value_min,
                                                         el=self.states["elevon_left"].value_max)
                self.states["aileron"].value_min = ail_min
                self.states["aileron"].value_max = ail_max

    def reset(self, state_init=None):
        for state in self.dynamics:
            init = None
            if state_init is not None and state in state_init:
                init = state_init[state]
            self.states[state].reset(value=init)

        if self.elevon_dynamics:
            elev, ail = self._map_elevon_to_elevail(er=self.states["elevon_right"].value, el=self.states["elevon_left"].value)
            self.states["elevator"].reset(value=elev)
            self.states["aileron"].reset(value=ail)

    def _map_elevail_to_elevon(self, elev, ail):
        er = -1 * ail + elev
        el = ail + elev
        return er, el

    def _map_elevon_to_elevail(self, er, el):
        ail = (-er + el) / 2
        elev = (er + el) / 2
        return elev, ail


class AttitudeQuaternion:
    def __init__(self):
        """
        Quaternion attitude representation used by PyFly.
        """
        self.quaternion = None
        self.euler_angles = {"roll": None, "pitch": None, "yaw": None}
        self.history = None

    def seed(self, seed):
        return

    def reset(self, euler_init):
        """
        Reset state of attitude quaternion to value given by euler angles.

        :param euler_init: ([float]) the roll, pitch, yaw values to initialize quaternion to.
        """
        if euler_init is not None:
            self._from_euler_angles(euler_init)
        else:
            raise NotImplementedError
        self.history = [self.quaternion]

    def as_euler_angle(self, angle="all", timestep=-1):
        """
        Get attitude quaternion as euler angles, roll, pitch and yaw.

        :param angle: (string) which euler angle to return or all.
        :param timestep: (int) timestep
        :return: (float or dict) requested euler angles.
        """
        e0, e1, e2, e3 = self.history[timestep]
        res = {}
        if angle == "roll" or angle == "all":
            res["roll"] = np.arctan2(2 * (e0 * e1 + e2 * e3), e0 ** 2 + e3 ** 2 - e1 ** 2 - e2 ** 2)
        if angle == "pitch" or angle == "all":
            res["pitch"] = np.arcsin(2 * (e0 * e2 - e1 * e3))
        if angle == "yaw" or angle == "all":
            res["yaw"] = np.arctan2(2 * (e0 * e3 + e1 * e2), e0 ** 2 + e1 ** 2 - e2 ** 2 - e3 ** 2)

        return res if angle == "all" else res[angle]

    @property
    def value(self):
        return self.quaternion

    def _from_euler_angles(self, euler):
        """
        Set value of attitude quaternion from euler angles.

        :param euler: ([float]) euler angles roll, pitch, yaw.
        """
        phi, theta, psi = euler
        e0 = np.cos(psi / 2) * np.cos(theta / 2) * np.cos(phi / 2) + np.sin(psi / 2) * np.sin(theta / 2) * np.sin(
            phi / 2)
        e1 = np.cos(psi / 2) * np.cos(theta / 2) * np.sin(phi / 2) - np.sin(psi / 2) * np.sin(theta / 2) * np.cos(
            phi / 2)
        e2 = np.cos(psi / 2) * np.sin(theta / 2) * np.cos(phi / 2) + np.sin(psi / 2) * np.cos(theta / 2) * np.sin(
            phi / 2)
        e3 = np.sin(psi / 2) * np.cos(theta / 2) * np.cos(phi / 2) - np.cos(psi / 2) * np.sin(theta / 2) * np.sin(
            phi / 2)

        self.quaternion = (e0, e1, e2, e3)

    def set_value(self, quaternion, save=True):
        """
        Set value of attitude quaternion.

        :param quaternion: ([float]) new quaternion value
        :param save: (bool) whether to commit value to history of attitude.
        """
        self.quaternion = quaternion
        if save:
            self.history.append(self.quaternion)


class Wind:
    def __init__(self, turbulence, mag_min=None, mag_max=None, b=None, turbulence_intensity=None, sim_length=300, dt=None):
        """
        Wind and turbulence object used by PyFly.

        :param turbulence: (bool) whether turbulence is enabled
        :param mag_min: (float) minimum magnitude of steady wind component
        :param mag_max: (float) maximum magnitude of steady wind component
        :param b: (float) wingspan of aircraft
        :param turbulence_intensity: (string) intensity of turbulence
        :param dt: (float) integration step length
        """
        self.turbulence = turbulence
        self.mag_min = mag_min
        self.mag_max = mag_max
        self.steady = None
        self.components = []
        self.turbulence_sim_length = sim_length

        if self.turbulence:
            self.dryden = DrydenGustModel(self.turbulence_sim_length, dt, b, intensity=turbulence_intensity)
        else:
            self.dryden = None

        self.np_random = None
        self.seed()

    def seed(self, seed=None):
        """
        Seed random number generator of object

        """
        self.np_random = np.random.RandomState(seed)
        if self.turbulence:
            self.dryden.seed(seed)

    def reset(self, value=None, noise=None):
        """
        Reset wind object to initial state

        :param value: ([float] or float) strength and direction of the n, e and d components or magnitude of the steady wind.
        """
        if value is None or isinstance(value, float) or isinstance(value, int):
            if value is None and self.mag_min is None and self.mag_max is None:
                value = []
                for comp in self.components:
                    comp.reset()
                    value.append(comp.value)
            else:
                if value is None:
                    magnitude = self.np_random.uniform(self.mag_min, self.mag_max)
                else:
                    magnitude = value
                w_n = self.np_random.uniform(-magnitude, magnitude)
                w_e_max = np.sqrt(magnitude ** 2 - w_n ** 2)
                w_e = self.np_random.uniform(-w_e_max, w_e_max)
                w_d = np.sqrt(magnitude ** 2 - w_n ** 2 - w_e ** 2)
                value = [w_n, w_e, w_d]

        if self.turbulence:
            self.dryden.reset(noise)

        self.steady = value
        for i, comp in enumerate(self.components):
            comp.reset(value[i])

    def set_value(self, timestep):
        """
        Set value to wind value at timestep t

        :param timestep: (int) timestep
        """
        value = self.steady

        if self.turbulence:
            value += self._get_turbulence(timestep, "linear")

        for i, comp in enumerate(self.components):
            comp.set_value(value[i])

    def get_turbulence_linear(self, timestep):
        """
        Get linear component of turbulence model at given timestep

        :param timestep: (int) timestep
        :return: ([float]) linear component of turbulence at given timestep
        """
        return self._get_turbulence(timestep, "linear")

    def get_turbulence_angular(self, timestep):
        """
        Get angular component of turbulence model at given timestep

        :param timestep: (int) timestep
        :return: ([float]) angular component of turbulence at given timestep
        """
        return self._get_turbulence(timestep, "angular")

    def _get_turbulence(self, timestep, component):
        """
        Get turbulence at given timestep.

        :param timestep: (int) timestep
        :param component: (string) which component to return, linear or angular.
        :return: ([float]) turbulence component at timestep
        """
        if timestep >= self.dryden.sim_length:
            self.dryden.simulate(self.turbulence_sim_length)

        if component == "linear":
            return self.dryden.vel_lin[:, timestep]
        else:
            return self.dryden.vel_ang[:, timestep]
