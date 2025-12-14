import numpy as np

from rnnFramework.vec import vec

def run_rnn(n_Wru_v, n_Wrr_n, m_Wzr_n, n_bx0_c, n_bx_1, m_bz_1, dt, tau,
            noise_sigma, inputs, conditionIds):
    # n_Wru_v: [n_units, n_inputs], input weights
    # n_Wrr_n: [n_units, n_units], recurrent weights
    # m_Wzr_n: [n_outputs, n_inputs], output weights
    # n_x0_c: [n_units, n_contexts], initial conditions per context
    # n_bx_1: [n_units, 1], bias of hidden units
    # m_bz_1: [n_outputs, 1], bias of output units
    # dt: (float), simulation time step
    # tau: (float), time constant
    # noise_sigma: (float), input noise sigma
    # inputs: [n_units, n_timesteps, n_trials], inputs to network u_t
    # conditionIds: [n_trials], condition id per trial (ctxt 1 or 2)

    # Outputs:
    # n_r_t: [n_units, n_timesteps, n_trials], activities of recurrent units (including transfer function)
    # m_z_t: [n_outputs, n_timesteps, n_trials], activities of readout unit
    # n_r0_1: [n_initial_conditions, n_trials], initial condition(s)
    # n_x_t: [n_units, n_timesteps, n_trials], membrane potentials of recurrent units (excluding transfer function)
    # n_x0_1: [n_units, n_trials], initial condition(s)

    [_, n_timesteps, n_trials] = np.shape(inputs)
    [n_outputs, n_units] = np.shape(m_Wzr_n)

    n_x_t = np.zeros([n_units, n_timesteps, n_trials])
    n_r_t = np.zeros([n_units, n_timesteps, n_trials])
    m_z_t = np.zeros([n_outputs, n_timesteps, n_trials])
    n_x0_1 = np.zeros([n_units, n_trials])
    n_r0_1 = np.zeros([n_units, n_trials])

    for trial_nr in range(n_trials):
        n_x0_1[:, trial_nr] = n_bx0_c[:, int(conditionIds[0, trial_nr] - 1)]
        n_r0_1[:, trial_nr] = np.tanh(n_x0_1[:, trial_nr])
        n_x_1 = n_x0_1[:, trial_nr]
        n_r_1 = n_r0_1[:, trial_nr]
        n_Wu_t = np.matmul(n_Wru_v, inputs[:, :, trial_nr])
        n_nnoise_t = noise_sigma * np.random.normal(size=[n_units, n_timesteps])
        for t in range(n_timesteps):
            n_x_1 = (1.0 - (dt / tau)) * n_x_1 + (dt / tau) * (n_Wu_t[:, t]
                        + np.matmul(n_Wrr_n, n_r_1) + np.squeeze(n_bx_1) + n_nnoise_t[:, t])
            n_r_1 = np.tanh(n_x_1)
            n_x_t[:, t, trial_nr] = n_x_1
            n_r_t[:, t, trial_nr] = n_r_1
            m_z_t[:, t, trial_nr] = np.matmul(m_Wzr_n, n_r_t[:, t, trial_nr]) + m_bz_1

    return n_r_t, m_z_t, n_r0_1, n_x_t, n_x0_1