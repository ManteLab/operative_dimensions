function mse = get_mse(m_z_t, targets, valid_trial_ids)            
    % mse = mean squared error --> mean((y_hat-y)**2)
    % m_z_t:  [n_outputs, n_timesteps, n_trials], network output over t and trials
    % targets: [n_outputs, n_timesteps, n_trials], correct network output targets over t and trials
    % valid_trial_ids: [list], list of trials numbers which should be considered to calculate the mse

    if strcmp(num2str(valid_trial_ids), 'all')
        n_trials = size(m_z_t, 3);
        valid_trial_ids = 1:n_trials;
    end
    mse = mean((m_z_t(1, :, valid_trial_ids) - targets(1, :, valid_trial_ids)).^2, 'all');
end

