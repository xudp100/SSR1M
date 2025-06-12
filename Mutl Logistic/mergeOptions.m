function opts = mergeOptions(opts1, opts2)
    if isempty(opts1)
        opts1 = struct();
    end
    if isempty(opts2)
        opts2 = struct();
    end

    opts = opts1;
    fields = fieldnames(opts2);
    for i = 1 : length(fields)
        opts.(fields{i}) = opts2.(fields{i});
    end
    
end
