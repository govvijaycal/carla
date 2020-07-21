import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pdb
import time
import inspect

# DEBUG: GETTING PRIMAL INFEASIBLE...
class LTVKinematicMPC(object):
	def __init__(self,
		lr = 1.5,
		dt = 0.2,
		horizon= 10,
		v_bounds = [0., 25.], 
		beta_bounds = [-0.25, 0.25], 
		beta_rate_bound = 0.25, 
		acc_bounds = [-3.0, 2.0],
		jerk_bound = 1.5,
		Q = np.diag([0.0, 1.0, 10.0, 0.01]), # s, e_y, e_psi, v
		R = np.diag([10.0, 50.]), # d/dt u_a, d/dt u_beta
		disc_fact = 0.9,
		trust_region_eps=1.0):

		for key in list(locals().keys()):
			if key == 'self':
				pass
			else:
				setattr(self, '%s' % key, locals()[key])

		self.nx = 4
		self.nu = 2

		''' Optimization Problem Setup'''
		# 1) Decision Variables
		self.zs = cp.Variable((self.horizon + 1, self.nx))               # states: z_{0,..., N}
		self.us = cp.Variable((self.horizon, self.nu))                   # inputs: u_{0,...,N-1}
		#self.slacks = cp.Variable((self.horizon,self.nu), nonneg=True)   # input_rate slack: s_{0, ..., N-1}

		# 2) Problem Parameters including Linearization.
		self.zrefs  = cp.Parameter((self.horizon, self.nx))         # target states: zref_{1, ..., N}
		self.z_init = cp.Parameter(self.nx)                         # measured initial state: z_{0}
		self.u_prev = cp.Parameter(self.nu)                         # prev. applied input: u_{-1}

		self.zlins = cp.Parameter((self.horizon, self.nx))          # state trajectory {0,...,N-1} about which linearization was done
		self.ulins = cp.Parameter((self.horizon, self.nu))          # input trajectory {0,...,N-1} about which linearization was done
		
		self.Alins = {}
		self.Blins = {}
		self.glins = {}
		for t in range(self.horizon):
			self.Alins[t] = cp.Parameter((self.nx, self.nx)) # df/dz: jacobian of f wrt state at trim (zlins[t], ulins[t])
			self.Blins[t] = cp.Parameter((self.nx, self.nu)) # df/du: jacobian of f wrt input at trim (zlins[t], ulins[t])
			self.glins[t] = cp.Parameter(self.nx)            # f: affine component, i.e. f evaluated at trim (zlins[t], ulins[t])
		     
		# 3) Objective
		cost = cp.quad_form( self.us[0,:] - self.u_prev, self.R) # input rate cost wrt to prev. applied input
		for t in range(self.horizon):
			cost += cp.quad_form( self.zs[t+1,:] - self.zrefs[t,:], disc_fact**t * self.Q) # tracking cost: off by one since zs has an additional initial state

			if t < (self.horizon - 1):
				cost += cp.quad_form( self.us[t+1,:] - self.us[t,:], disc_fact**t * self.R) # input rate cost
		
		#cost += cp.sum(self.slacks) # penalizing slack variables

		# 4) Constraints
		constraints = []
		constraints += [self.zs[0,:] == self.z_init] # initial state constraint

		# State dynamics.
		for t in range(self.horizon):
			constraints += [self.zs[t+1,:] == self.zs[t, :] + self.dt * 
			               ( self.glins[t] + self.Alins[t] @ (self.zs[t,:] - self.zlins[t,:]) + self.Blins[t] @ (self.us[t,:] - self.ulins[t,:]) )]

		# # State constraint on velocity.
		# for t in range(self.horizon + 1):
		# 	constraints += [self.zs[t,3] >= self.v_bounds[0],
		# 	                self.zs[t,3] <= self.v_bounds[1]]

		# # State Constraint on Lateral Error
		# for t in range(1, self.horizon + 1):
		# 	constraints += [self.zs[t,1] >= -1.5,
		# 	                self.zs[t,1] <=  1.5]

		# Input constraints.
		for t in range(self.horizon):
			constraints += [self.us[t,0] >= self.acc_bounds[0],
			                self.us[t,0] <= self.acc_bounds[1],
			                self.us[t,1] >= self.beta_bounds[0],
			                self.us[t,1] <= self.beta_bounds[1]]

		# Soft Input rate constraints.
		jerk_constraint = np.round(self.jerk_bound * self.dt, 3)
		brate_constraint = np.round(self.beta_rate_bound * self.dt, 3)
		constraints += [ cp.abs(self.us[0,0] - self.u_prev[0]) <= jerk_constraint,
		                 cp.abs(self.us[0,1] - self.u_prev[1]) <= brate_constraint]

		for t in range(self.horizon - 1):
			constraints += [ cp.abs(self.us[t+1,0] - self.us[t,0]) <= jerk_constraint,
							 cp.abs(self.us[t+1,1] - self.us[t,1]) <= brate_constraint]

		# Trust Region on inputs.
		constraints += [cp.norm(self.ulins - self.us,1) <= self.trust_region_eps]

		self.prob = cp.Problem(cp.Minimize(cost), constraints)

	def update(self, zrefs, z_init, u_prev, zlins, ulins, Alins, Blins, glins, curvs):
		
		self.zrefs.value  = zrefs
		self.z_init.value = z_init
		self.u_prev.value = u_prev
		self.zlins.value  = zlins
		self.ulins.value  = ulins

		for t in range(self.horizon):
			self.Alins[t].value = Alins[t]
			self.Blins[t].value = Blins[t]
			self.glins[t].value = glins[t]

		self.curvs = curvs

	def solve(self, max_num_iters=1, eps=1e-1, debug=False):
		u_sols = [self.ulins.value]
		z_sols = [self.zlins.value]

		st = time.time()
		for it in range(max_num_iters):
			try:
				self.prob.solve(solver='OSQP', warm_start=False, verbose=False, time_limit=0.02)#,  linsys_solver="mkl pardiso" time_limit=0.002)
				u_next = self.us.value
				z_next = self.zs.value[:-1]

				u_sols.append(u_next)
				z_sols.append(z_next)

				if np.linalg.norm(u_next - u_sols[-1]) <= eps or max_num_iters - 1 == it:
					break
				
				#z_next = self.simulate(self.z_init.value, u_next, return_with_init_state = True)[:-1]
				Als, Bls, gls = self.linearize(z_next, u_next, self.curvs)

				self.zlins.value = z_next
				self.ulins.value = u_next

				for t in range(self.horizon):
					self.Alins[t].value = Als[t]
					self.Blins[t].value = Bls[t]
					self.glins[t].value = gls[t]
			except:
				break


		et = time.time()
		
		if debug:
			print(et - st)
			plt.figure()
			plt.plot(self.zrefs.value[:,0], self.zrefs.value[:,1], 'k')
			for i_zs, zs in enumerate(z_sols):
				plt.plot(zs[:,0], zs[:,1], 'x', label='z%d' % i_zs)
			plt.legend()

			plt.figure()
			plt.plot(np.arange(1, self.horizon + 1), self.zrefs.value[:,2], 'k')
			for i_zs, zs in enumerate(z_sols):
				plt.plot(zs[:,2], 'x', label='z%d' % i_zs)
			plt.title('e_psi')
			plt.legend()

			plt.figure()
			plt.plot(np.arange(1, self.horizon + 1), self.zrefs.value[:,3], 'k')
			for i_zs, zs in enumerate(z_sols):
				plt.plot(zs[:,3], 'x', label='z%d' % i_zs)
			plt.title('v')
			plt.legend()

			plt.figure()
			for i_us, us in enumerate(u_sols):
				plt.subplot(121)
				plt.plot(us[:,0], label='u%d' % i_us)
				plt.subplot(122)
				plt.plot(us[:,1], label='u%d' % i_us)
			plt.legend()

			plt.show()
		return z_sols[-1], u_sols[-1]

	def simulate(self, z_init, u_traj, curvs, return_with_init_state = False):
		N = len(u_traj)
		curr_tm_ind = 0
		if return_with_init_state:
			z_traj = np.ones((N+1, self.nx)) * np.nan
			z_traj[curr_tm_ind,:] = z_init
			curr_tm_ind += 1
		else:
			z_traj = np.ones((N, self.nx)) * np.nan

		z_curr = np.copy(z_init)
		for u_curr, curv in zip(u_traj, curvs):
			s, ey, ep, v = z_curr
			u_a, u_b     = u_curr

			s_n  = s  + self.dt * (v * np.cos(ep + u_b)) 
			ey_n = ey + self.dt * (v * np.sin(ep + u_b))
			ep_n = ep + self.dt * (v / self.lr * np.sin(u_b) - curv * v * np.cos(ep + u_b))
			v_n  = v  + self.dt * (u_a)

			z_curr = np.array([s_n, ey_n, ep_n, v_n])
			z_traj[curr_tm_ind, :] = np.array([s_n, ey_n, ep_n, v_n])
			curr_tm_ind += 1

		return z_traj 

		# returns z_{k+1} given z_{k} and u_{k}

	def linearize(self, z_traj, u_traj, curvs, debug=False):
		# linearizes the model about trajectory z,u and (constant) curvature and returns A, B, g
		N = len(z_traj)
		A_lin = np.ones((N, self.nx, self.nx)) * np.nan
		B_lin = np.ones((N, self.nx, self.nu)) * np.nan
		g_lin = np.ones((N, self.nx)) * np.nan

		for t, (z_curr, u_curr, curv) in enumerate(zip(z_traj, u_traj, curvs)):
			s, ey, ep, v = z_curr
			u_a, u_b   = u_curr

			A_lin[t] = np.array([[0., 0., -v * np.sin(ep + u_b), np.cos(ep + u_b)], 
				                 [0., 0.,  v * np.cos(ep + u_b), np.sin(ep + u_b)],
				                 [0., 0.,  0., 0.],
				                 [0., 0.,  0., 0.]])
			A_lin[t][2, 2] = curv * v * np.sin(ep + u_b)
			A_lin[t][2, 3] = np.sin(u_b) / self.lr - curv * np.cos(ep + u_b)
		
			B_lin[t] = np.array([[0.,-v * np.sin(ep + u_b)],
				                 [0., v * np.cos(ep + u_b)],
				                 [0., 0.],
				                 [1., 0.]])
			B_lin[t][2,1] = v / self.lr * np.cos(u_b) + curv * v * np.sin(ep + u_b)

			g_lin[t] = np.array([ v * np.cos(ep + u_b),
				                  v * np.sin(ep + u_b),
				                  v / self.lr * np.sin(u_b) - curv * v * np.cos(ep + u_b),
				                  u_a])

			if debug:
				A_disc, B_disc = self._discrete_model_num_jacobian(z_curr, u_curr, curv)
				A_num = (A_disc - np.eye(A_disc.shape[0])) / self.dt
				B_num = B_disc / self.dt

				print('\tA norm difference: %f' % np.linalg.norm(np.round(A_lin[t], 6) - np.round(A_num, 6)))
				print('\tB norm difference: %f' % np.linalg.norm(np.round(B_lin[t], 6) - np.round(B_num, 6)))
				
		return A_lin, B_lin, g_lin

	def _simulate_linearized_model(self, z_init, u_traj, A_lin, B_lin, g_lin, z_traj_lin, u_traj_lin, return_with_init_state = False):
		z_traj = []
		if return_with_init_state:
			z_traj.append(z_init)

		z_curr = np.copy(z_init)
		for u_curr, A, B, g, z_lin, u_lin in zip(u_traj, A_lin, B_lin, g_lin, z_traj_lin, u_traj_lin):
			z_next = z_curr + ( np.dot(A, z_curr - z_lin) + np.dot(B, u_curr - u_lin) + g ) * self.dt
			z_traj.append(z_next)
			z_curr = z_next

		return np.array(z_traj) 

	def _discrete_model_num_jacobian(self, state, inp, curv, eps=1e-6):
		nx = len(state)
		nu = len(inp)
		
		if len(inp.shape) == 1:
			inp = inp.reshape(1, nu)

		state_jac = np.zeros([nx,nx])
		inp_jac   = np.zeros([nx,nu])
		for i in range(nx):
			splus  = state + eps * np.array([int(ind==i) for ind in range(nx)])
			sminus = state - eps * np.array([int(ind==i) for ind in range(nx)])

			f_plus  = self.simulate(splus,  inp, curv)
			f_minus = self.simulate(sminus, inp, curv)

			state_jac[:,i] = (f_plus - f_minus) / (2.*eps)
		for i in range(nu):
			iplus  = inp + eps * np.array([[int(ind==i) for ind in range(nu)]])
			iminus = inp - eps * np.array([[int(ind==i) for ind in range(nu)]])
			
			f_plus  = self.simulate(state, iplus, curv)
			f_minus = self.simulate(state, iminus, curv)

			inp_jac[:,i] = (f_plus - f_minus) / (2.*eps)
		return state_jac, inp_jac 


if __name__ == '__main__':
	ltv_kmpc = LTVKinematicMPC()
	
	run_linearization_tests = True
	run_solver_tests = False

	if run_linearization_tests:
		num_tests = 5
		for ind in range(num_tests):
			print('TEST %d' % ind)
			init_state = np.random.random((4))
			input_trajectory = np.random.random((5, 2))
			curvs = np.random.random((5,1)) * 0.25
			
			state_trajectory = ltv_kmpc.simulate(init_state, input_trajectory, curvs, return_with_init_state = True) # [0, ..., N]
			Al, Bl, gl = ltv_kmpc.linearize(state_trajectory[:-1], input_trajectory, curvs, debug=True) # linearize with [0, ..., N-1]

			for gain in [0.0, 1e-3, 1e-1, 1.]:
				perturb_init_state           = gain * np.random.random((4)) + init_state
				perturb_input_trajectory     = gain * np.random.random((5,2)) + input_trajectory
				perturb_state_trajectory     = ltv_kmpc.simulate(perturb_init_state, perturb_input_trajectory, curvs)
				perturb_state_trajectory_lin = ltv_kmpc._simulate_linearized_model(perturb_init_state, perturb_input_trajectory, Al, Bl, gl, state_trajectory[:-1], input_trajectory)

				print('\tPerturb Gain: %f, Traj Diff Norm: %f' % (gain, np.linalg.norm(perturb_state_trajectory - perturb_state_trajectory_lin)))


	if run_solver_tests:
		N_horizon = 10
		nx = 4
		nu = 2
		dt = 0.1
		# TEST 1: feasible trajectory
		init_state = np.array([1.0, -0.1, 0.1, 10.0])
		curvs  = np.random.random((N_horizon, 1)) * 0.25
		init_input = np.zeros(nu)
		
		# make the reference
		input_rate = (np.random.random((N_horizon, 2)) - 0.4) * [2.0 * dt, 0.20 * dt]
		inputs_ref = init_input + np.cumsum(input_rate, axis = 0)
		states_ref = ltv_kmpc.simulate(init_state, inputs_ref, curvs)

		# make the linearization
		inputs_lin = np.zeros((N_horizon, 2))
		states_lin = ltv_kmpc.simulate(init_state, inputs_lin, curvs, return_with_init_state = True)[:-1]
		Als, Bls, gls = ltv_kmpc.linearize(states_lin, inputs_lin, curvs, debug=False)

		ltv_kmpc.update(states_ref, init_state, init_input, states_lin, inputs_lin, Als, Bls, gls, curvs)
		ltv_kmpc.solve(debug=True)



