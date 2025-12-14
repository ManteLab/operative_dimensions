classdef InputGeneratorCtxt
    % class to generate input/targets for context-dependent integration
    
    properties       
        n_inputs       = 5;
        n_outputs      = 1;
        nIntegrators   = 2;
        maxBound       = 1;
        minBound       = -1;
        integrationTau = 5 * 0.01;  
        
        burnLength       = 0.650;
        dt               = 0.001;
        tau              = 0.01;  
        time             = 0.75;
        
        all_coherencies = [-0.5 -0.12 -0.03 0.03 0.12 0.5]*0.3;
    end
    
    
    methods    
        function [coherencies_trial, conditionIds, inputs, targets] = get_ctxt_dep_integrator_inputOutputDataset(obj, n_trials, with_inputnoise)
            % main method to generate input/outputs  
            % n_trials: (integer), number of trials per context and per coherency
            % with_inputnoise: (bool), if true: add noise to sensory inputs
            
            % coherencies_trial = [nIntegrators x N], input coherencies of sensory input 1 and 2 over trials
            % coherencies are shuffled. in particular the different coherencies within a
            % trial are independent.   
            % conditionIds = [1 x N], context ID per trial
            % inputs: [2*nIntegrators+1 x T x N], sensory and context input over time and trials
            % targets: [1 x T x N], target output over time and trials

            % noise settings
            input_noise = 0;
            if with_inputnoise
                input_noise = 31.6228 * sqrt(obj.dt);
            end
            noiseSigma  = input_noise;

            % generate inputs/targets
            n_trials_total = n_trials * numel(obj.all_coherencies) * obj.nIntegrators;
            nTimesteps     = round(obj.time/obj.dt) + (obj.burnLength/obj.dt);
            inputs         = zeros(obj.n_inputs,  nTimesteps, n_trials_total);
            targets        = zeros(obj.n_outputs, nTimesteps, n_trials_total);
            conditionIds   = repmat(1:obj.nIntegrators, 1, n_trials_total/obj.nIntegrators);
            
            % set input coherencies per trial
            set_all_cohs = zeros([1, n_trials_total]);
            for coh_nr = 1:numel(obj.all_coherencies)
                set_all_cohs(coh_nr*(obj.nIntegrators*n_trials)+1:(coh_nr+1)*(obj.nIntegrators*n_trials)) = ones([1, obj.nIntegrators*n_trials])*obj.all_coherencies(coh_nr);
            end
            coherencies_trial = zeros(obj.nIntegrators, n_trials_total);
            for i = 1:obj.nIntegrators
                coherencies_trial(i, find(conditionIds == 1)) = set_all_cohs(randperm(n_trials_total, n_trials*numel(obj.all_coherencies)));
                coherencies_trial(i, find(conditionIds == 2)) = set_all_cohs(randperm(n_trials_total, n_trials*numel(obj.all_coherencies)));
            end
            
            % generate one trial
            for trial_nr = 1:n_trials_total
                [inputs(:, :, trial_nr), targets(:, :, trial_nr)] = obj.generate_one_trial(coherencies_trial(1:2, trial_nr), conditionIds(trial_nr), noiseSigma);
            end
            
        end

        
        function [inputs, targets] = generate_one_trial(obj, all_drifts, conditionId, noiseSigma)
            % method to generate input and targets over time for one trial
            % all_drifts: [n_integrators, 1], input coherency for sensory input 1 and 2
            % conditionId: (int), context of trial (1 or 2)
            % noiseSigma: (float), sigma of input noise
            
            % inputs: (n_inputs, n_timesteps)
            % targets: (n_outputs, n_timesteps)
            
            
            nTimesteps = round(obj.time/obj.dt) + (obj.burnLength/obj.dt);
            inputs     = zeros(obj.n_inputs, nTimesteps);
            targets    = zeros(obj.n_outputs, nTimesteps);

            % set context inputs
            inputs((obj.n_inputs-obj.nIntegrators-1)+conditionId,:) = ones(1,nTimesteps);
            
            % set sensory inputs
            for i = 1:obj.nIntegrators
                inputs(i,(obj.burnLength/obj.dt)+1:end) = all_drifts(i) + randn(1, (nTimesteps-((obj.burnLength/obj.dt)))) * noiseSigma;
            end

            % set target values
            hit_bound = 0;
            for t = (obj.burnLength/obj.dt)+1:nTimesteps
                
                if ~hit_bound
                    targets(t) = targets(t-1) + obj.dt/obj.integrationTau*inputs(conditionId, t);
                    if (targets(t) >= obj.maxBound)
                        targets(t) = obj.maxBound;
                        hit_bound = 1;
                    elseif (targets(t) <= obj.minBound )
                        targets(t) = obj.minBound;
                        hit_bound = 1;
                    end
                    
                else
                    targets(:, t) = targets(:, t-1);
                end                
                
            end   
            
        end
    end
end