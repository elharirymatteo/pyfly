import matplotlib.pyplot as plt

class Plot:
    def __init__(self, id, variables=None, title=None, x_unit=None, xlabel=None, ylabel=None, dt=None,
                 plot_quantity=None):
        """
        Plot object used by PyFly to house (sub)figures.

        :param id: (string or int) identifier of figure
        :param variables: ([string]) list of names of states included in figure
        :param title: (string) title of figure
        :param x_unit: (string) unit of x-axis, one of timesteps or seconds
        :param xlabel: (string) label for x-axis
        :param ylabel: (string) label for y-axis
        :param dt: (float) integration step length, required when x_unit is seconds.
        :param plot_quantity: (string) the attribute of the states that is plotted
        """
        self.id = id
        self.title = title

        if x_unit is None:
            x_unit = "timesteps"
        elif x_unit not in ["timesteps", "seconds"]:
            raise Exception("Unsupported x unit (one of timesteps/seconds)")
        elif x_unit == "seconds" and dt is None:
            raise Exception("Parameter dt can not be none when x unit is seconds")

        self.x_unit = x_unit
        self.y_units = []
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.variables = variables
        self.axis = None
        self.dt = dt
        self.plot_quantity = plot_quantity

    def add_variable(self, var):
        """
        Add state to plot

        :param var: (string) name of state
        """
        if self.variables is None:
            self.variables = []
        self.variables.append(var)

        if var.unit not in self.y_units:
            self.y_units.append(var.unit)

        if len(self.y_units) > 2:
            raise Exception("More than two different units in plot")

    def close(self):
        """
        Close figure
        """
        for var in self.variables:
            var.close_plot(self.id)

        self.axis = None

    def plot(self, fig=None, targets=None):
        """
        Plot history of states in figure

        :param fig: (matplotlib.pyplot.figure) optional parent figure
        :param targets: (dict) target values for states in state name - values pairs
        """
        first = False
        if self.axis is None:
            first = True

            if self.xlabel is not None:
                xlabel = self.xlabel
            else:
                if self.x_unit == "timesteps":
                    xlabel = self.x_unit
                elif self.x_unit == "seconds":
                    xlabel = "Time (s)"

            if self.ylabel is not None:
                ylabel = self.ylabel
            else:
                ylabel = self.y_units[0]

            if fig is not None:
                self.axis = {self.y_units[0]: plt.subplot(fig, title=self.title, xlabel=xlabel, ylabel=ylabel)}
            else:
                self.axis = {self.y_units[0]: plt.plot(title=self.title, xlabel=xlabel, ylabel=ylabel)}

            if len(self.y_units) > 1:
                self.axis[self.y_units[1]] = self.axis[self.y_units[0]].twinx()
                self.axis[self.y_units[1]].set_ylabel(self.y_units[1])

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, v in enumerate(self.variables):
            if self.plot_quantity is not None:
                v.plot_quantity = self.plot_quantity
            target = targets[v.name] if targets is not None and v.name in targets else None
            v.plot(self.axis[v.unit], plot_id=self.id, target=target, color=colors[i])

        if first:
            if len(self.y_units) > 1:
                labeled_lines = []
                for ax in self.axis.values():
                    labeled_lines.extend([l for l in ax.lines if "_line" not in l.get_label()])
                self.axis[self.y_units[0]].legend(labeled_lines, [l.get_label() for l in labeled_lines])
            else:
                self.axis[self.y_units[0]].legend()

        else:
            for ax in self.axis.values():
                ax.relim()
                ax.autoscale_view()

        if self.x_unit == "seconds":
            xticks = self.axis[self.y_units[0]].get_xticks()
            self.axis[self.y_units[0]].set_xticklabels(["{0:.1f}".format(tick * self.dt) for tick in xticks])

