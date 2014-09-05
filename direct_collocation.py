import sympy as sym


class SymbolicContinousSystem():

    def __init__(self, mass_matrix, forcing_vector, coordinates, speeds, specified, constants):
        self.mass_matrx = mass_matrix
        self.forcing_vector = forcing_vector
        self.coordinates = coordinates
        self.speeds = speeds
        self.states = coordinates + speeds
        self.specified = specified
        self.constants = constants

    def state_derivatives(self):
        """Returns functions of time which represent the time derivatives of
        the states."""
        return [state.diff() for state in self.states]

    def constants_dict(constant_values):
        """Returns an ordered dictionary which maps the system constant symbols
        to numerical values."""
        return OrderedDict(zip(self.constants, constant_values))

    def f_minus_ma(self):
        """Returns Fr + Fr* (F - ma) from the mass_matrix and forcing
        vector."""
        xdot = self.state_derivatives()
        return self.mass_matrix * xdot - self.forcing_vector

    def closed_loop(self, controller_dict, equilibrium_dict=None):
        """Returns the equation of motion expressions in closed loop form.

        Parameters
        ----------
        controller_dict : dictionary
            A dictionary mapping the specified inputs to the controller
            equations.
        equilbrium_dict : dictionary
            A dictionary mapping the states to equilibrium values.

        """

        if equilibrium_dict is not None:
            for k, v in controller_dict.items():
                controller_dict[k] = v.subs(equilibrium_dict)

        return self.f_minus_ma.subs(controller_dict)

    def discrete_symbols(self, interval='h'):
        """Returns discrete symbols for each state and specified input along
        with an interval symbol.

        Parameters
        ----------
        states : list of sympy.Functions
            The n functions of time representing the system's states.
        specified : list of sympy.Functions
            The m functions of time representing the system's specified inputs.
        interval : string, optional
            The string to use for the discrete time interval symbol.

        Returns
        -------
        current_states : list of sympy.Symbols
            The n symbols representing the system's ith states.
        previous_states : list of sympy.Symbols
            The n symbols representing the system's (ith - 1) states.
        current_specified : list of sympy.Symbols
            The m symbols representing the system's ith specified inputs.
        interval : sympy.Symbol
            The symbol for the time interval.

        """

        xi = [sym.Symbol(f.__class__.__name__ + 'i') for f in self.states]
        xp = [sym.Symbol(f.__class__.__name__ + 'p') for f in self.states]
        si = [sym.Symbol(f.__class__.__name__ + 'i') for f in self.specified]
        h = sym.Symbol(interval)

        return xi, xp, si, h

    def discretize(self, interval='h'):
        """Returns the constraint equations in a discretized form. Backward
        Euler discretization is used.

        Parameters
        ----------
        states : list of sympy.Functions
            The n functions of time representing the system's states.
        specified : list of sympy.Functions
            The m functions of time representing the system's specified inputs.
        interval : string, optional
            The string to use for the discrete time interval symbol.

        Returns
        -------
        discrete_eoms : sympy.Matrix
            The column vector of the constraint expressions.

        """
        xi, xp, si, h = self.discrete_symbols(interval=interval)

        euler_formula = [(i - p) / h for i, p in zip(xi, xp)]

        # Note : The Derivatives must be substituted before the symbols.
        eoms = eoms.subs(dict(zip(self.state_derivatives(), euler_formula)))

        eoms = eoms.subs(dict(zip(self.states + self.specified, xi + si)))

        return eoms


class DirectCollocater():

    def __init__(mass_matrix, forcing_vector, states, specified, constants):
        pass
