# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 15:01:23 2016

@author: huawei
"""

from collections import OrderedDict

import sympy as sym
import sympy.physics.mechanics as me

sym_kwargs = {'positive': True, 'real': True}
me.dynamicsymbols._t = sym.symbols('t', **sym_kwargs)

class HomotopyTransfer():
    
    '''This class transfer orginal model's first order implicited dynamic equations
    into homotopy first order implicited dynamics equations.
    
    Reference:
    ----------------------------------------------    
    Vyasarayani, Chandrika P., et al. "Parameter identification in dynamic systems
    using the homotopy optimization approach." Multibody System Dynamics
    26.4 (2011): 411-424.
    --------------------------------------------------
    
    Main functions:
    --------------------------------------------------
    1. Add tracking Sympy symbols into specified group according to state numbers.
    
    2. Add homotopy_control, tracking_dynamic_control symbolic names into known 
       parameter group and map them with constant values.
    
    3. Add homotopy Tracking part into dynamic equations (symbolic type)

        homotopy motion equations = [1 0 0    0   ]   [f1(x, xd, p, r) ]   
			      	         0 1 0    0     *  f2(x, xd, p, r)   
 			      	         0 0 1-la 0        f3(x, xd, p, r)  
			      	        [0 0 0    1-la]   [f4(x, xd, p, r) ]

				          [0 0 0  0 ]     [x1_dot]       [e1]   [x1]
 			             +   0 0 0  0   * (  x2_dot  - K*(  e2  -  x2  ) )
			                 0 0 la 0        x3_dot         e3     x3
				          [0 0 0  la]     [x4_dot]       [e4]   [x4]
    ------------------------------------------------
    '''
    
    def __init__(self, model_dynamics, state_symbols, model_sepcified,
                 known_parameter, homotopy_control=0, tracking_dynamic_control=10):
                     
        '''
            
    model_dynamics : sympy.Matrix, shape(n, 1)
            A column matrix of SymPy expressions defining the right hand
            side of the equations of motion when the left hand side is zero,
            e.g. 0 = x'(t) - f(x(t), u(t), p) or 0 = f(x'(t), x(t), u(t),
            p). These should be in first order form but not necessairly
            explicit.
    state_symbols : iterable
            An iterable containing all of the SymPy functions of time which
            represent the states in the equations of motion. 
    model_sepcified: iterable
            An iterable containing all the non-state SymPy functions of time to
            ndarrays of floats of shape(N,). Any time varying parameters in
            the equations of motion not provided in this dictionary will
            become free trajectories optimization variables. 
    known_parameter : dictionary, optional
            A dictionary that maps the SymPy symbols representing the known
            constant parameters to floats. Any parameters in the equations
            of motion not provided in this dictionary will become free
            optimization variables.
    homotopy_control: float, optional
	       A parameter in homotopy method that controls the change of 
	       motion equations. The default value of it is 0, which means 
             the homotopy method does not apply. 
    tracking_dynamic_control: float, optional
	       A parameter in homotopy method that adjust the dynamics of extra
	       'data tracking' term. The default value of it is 10.
        
        '''

        self.state_symbols = list(state_symbols)
        
        self.num_states = len(self.state_symbols)
        
        self.model_par_map = known_parameter
        
        self.model_dynamics = model_dynamics
                  
        self.specified = model_sepcified
        
        self.lamda = homotopy_control
        
        self.dynamic = tracking_dynamic_control
        
        self._run()
        
    def _run(self):
        
        '''Run all the functions below'''
        
        self._state_derivative()
        self._add_sepcified()
        self._add_parameter()
        self._numberical_par()
        self._generate_matrix()
        self._dynamic_transfer()
        
        
    def _state_derivative(self):
        
        ''' Get symbols of state derivative''' 
        
        self.time = me.dynamicsymbols._t
        
        self.state_derivative_symbols = sym.Matrix([s.diff(self.time) for
                                               s in self.state_symbols])
                                               
        return self
    
    def _add_sepcified(self):
        ''' 
        Add state tracking symbols accroding to the number of states, the state tracking
        symbols have the same order as the state symbols
        '''
        
        self.time = me.dynamicsymbols._t
        
        symbolnames = sym.symbols('s_T_1:%d'%(self.num_states+1))
        
        time_varying = [s(self.time) for s in symbolnames]
                      
        self.names = sym.symbols('state_tracing_1:%d'%(self.num_states+1))                                          
        
        for k in range(0,self.num_states):
            self.specified[str(self.names[k])] = time_varying[k]
        
        return self
        
    def _add_parameter(self):
        ''' Add homotopy_control and tracking_dynamic_control into constant parameter group'''
        
        self.parameters = OrderedDict()
        self.parameters['homotopy_control'] = sym.symbols('l_A', **sym_kwargs)
        self.parameters['tracking_dynamic_control'] = sym.symbols('k_T', **sym_kwargs)
        
        return self      
        
    def _numberical_par(self):
        ''' Vaule homotopy_control and tracking_dynamic_control accroding input'''
        
        p = {'homotopy_control': self.lamda,
             'tracking_dynamic_control': self.dynamic}

        for k, v in self.parameters.items():
            self.model_par_map[v] = p[k]   
            
        return self
            
    def _generate_matrix(self):
        '''Generate Matrixs for form homotopy equations'''
        
        self.Lamda_matrix = sym.zeros(self.num_states)
        self.Lamda_matrix[self.num_states/2:, self.num_states/2:] = self.parameters['homotopy_control']*sym.eye(self.num_states/2)
        
        self.Inv_Lamda_matrix = sym.eye(self.num_states)
        self.Inv_Lamda_matrix[self.num_states/2:, self.num_states/2:] = (1-self.parameters['homotopy_control'])*sym.eye(self.num_states/2)
        
        self.k_matrix = self.parameters['tracking_dynamic_control'] * sym.eye(self.num_states)
        
        return self
        
        
    def _dynamic_transfer(self):
        ''' generate homotopy equations by adding tracking part'''
        
        self.tracking_equation = []
        
        for k in range(0,self.num_states):
            self.tracking_equation.append(self.specified[str(self.names[k])] - self.state_symbols[k])

                            
        self.weighted_model_dynamics = self.Inv_Lamda_matrix*self.model_dynamics
        
        self.derivative_matrix = sym.Matrix(self.state_derivative_symbols)
       
        self.weighted_tracking_dynamics = self.Lamda_matrix*(self.state_derivative_symbols
        -self.k_matrix*sym.Matrix(self.tracking_equation))
        
        self.first_order_implicit = self.weighted_model_dynamics + self.weighted_tracking_dynamics
        
        return self