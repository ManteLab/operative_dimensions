function [state_distances] = get_state_distance_between_trajs(traj_A, traj_B)
    % calculate Euclidean distance between network trajectories A and B (between
    % every time step per trial, then average over all time steps and trials)
    % traj_A/B: [n_units, n_timesteps, n_trajectories_per_group_to_compare], network trajectories 
    %                                          to compare (population activities over time of trials)    

    [~, n_timesteps, n_samples] = size(traj_A);
    state_distances = zeros([n_timesteps, n_samples]);
    for t_nr = 1:n_timesteps
        for sample_nr = 1:n_samples
            state_distances(t_nr, sample_nr) = norm((traj_A(:, t_nr, sample_nr) - traj_B(:, t_nr, sample_nr)));
        end
    end    
    state_distances = mean(mean(state_distances, 2), 1);
    
end
