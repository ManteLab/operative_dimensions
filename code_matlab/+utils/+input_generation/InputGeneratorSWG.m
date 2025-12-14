classdef InputGeneratorSWG
    % class to generate input/targets for sine wave generation
    
    methods
       function [simParams] = get_simulation_parameters(obj, n_trials)
            simParams         = [];
            simParams.dt      = 0.001;  % simulation time step
            simParams.time    = 0.5;  % trial duration
            simParams.nTrials = n_trials;
       end 
        
        function [inputParams] = get_input_parameters(obj)
            inputParams = [];
            inputParams.nFreqs = 51;
            inputParams.max_freq = 6;
            inputParams.min_freq = 1;
            inputParams.seed_freqs = 2001;            
        end
        
        function [all_freqs, all_freq_ids, inputs, targets, conditionIds] = get_sine_wave_generator_inputOutputDataset(obj, n_trials)
            % generate input/targets for sine wave genration as follows:
            % input for nFreqs different frequencies equally spread between
            % min_freq - max_freq.
            % inputs are then set to a static value of
            % (freq_id/nFreqs)+0.25.
            % targets = sin(2*pi*freq*t)

            % n_trials: (int), number of trial in total (uniformly sampled across all nFreqs)
            
            % all_freqs: [1, n_trials], frequencies of each trial
            % all_freq_ids: [n_trials, 1], all frequency IDs of each trial
            % input: [1, n_timesteps, n_trials]
            % targets: [1, n_timesteps, n_trials]
            % conditionIds: [1, n_trials], input condition (always 1)
            

            [simParams] = obj.get_simulation_parameters(n_trials);
            [inputParams] = obj.get_input_parameters();
            
            if ~isempty(inputParams.seed_freqs)
                rng(inputParams.seed_freqs)
            end

            freq_range   = inputParams.max_freq - inputParams.min_freq;
            delta_freq   = freq_range/(inputParams.nFreqs-1);
            freq_per_idx = inputParams.min_freq:delta_freq:inputParams.max_freq;
            
            % set frequency IDs per trial
            all_freq_ids(:, 1) = randsample(1:inputParams.nFreqs, n_trials, true);
            
            % generate inputs and target over time per trial
            t = simParams.dt:simParams.dt:simParams.time;
            all_freqs = nan([1, n_trials]);
            inputs    = nan([1, numel(t), n_trials]);
            targets   = nan([1, numel(t), n_trials]);
            for trial_nr = 1:n_trials
                freq_id = all_freq_ids(trial_nr,1);
                freq    = freq_per_idx(freq_id);

                all_freqs(1, trial_nr)  = 2*pi*freq;
                inputs(1, :, trial_nr)  = (freq_id/inputParams.nFreqs)+0.25;
                targets(1, :, trial_nr) = sin(2*pi*freq*t);
            end

            conditionIds = ones([1, size(inputs, 3)]); 
            
        end
    end
end
