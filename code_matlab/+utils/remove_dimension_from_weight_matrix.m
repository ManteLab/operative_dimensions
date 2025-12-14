function [W_minusDims] = remove_dimension_from_weight_matrix(W, all_dims_to_remove, dim_type)
    % remove a given dimension from matrix W, assuming W has
    % dimensions as columns or rows (specificed by dim_type ('columns' or 'rows')
    % W: [n_units, n_units], weight matrix from which dimensions will be removed
    % all_dims_to_remove: [n_units, n_dims], dimensions as columns, number of columns = num dims to remove
    % dim_type: (str), 'columns' or 'rows'
    
    % W_minusDims: [n_units, n_units], reduced-rank weight matrix W
    
    if strcmp(dim_type, 'rows')
        W = W.';
    end

    n_dims_total = size(W, 2);
    n_dims_to_remove = size(all_dims_to_remove, 2);

    % remove dims from columns of W
    % project every column of W onto dim_to_remove to obtain
    % scaling_factor (proj_onto_dim_to_remove)
    % then remove this dim_to_remove*scaling_factor from each column of W
    W_minusDims = W;
    for dim_nr_to_remove = 1:n_dims_to_remove
        for dim_nr = 1:n_dims_total
            dim_to_remove = all_dims_to_remove(:, dim_nr_to_remove);
            proj_onto_dim_to_remove = dot(W_minusDims(:, dim_nr), dim_to_remove);
            W_minusDims(:, dim_nr) = W_minusDims(:, dim_nr) - proj_onto_dim_to_remove * dim_to_remove;    
        end
    end

    if strcmp(dim_type, 'rows')
        W_minusDims = W_minusDims.';
    end
end
