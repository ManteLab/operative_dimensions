function  [normed_a] = make_unit_length(a)
    %  Take vector a and make unit length by dividing by its norm
    normed_a = a / norm(a);
end
