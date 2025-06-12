function step = stepsize_alg(iter, epoch, options)
    if ~isfield(options, 'step_init')
        step_init = 0.1;
    else
        step_init = options.step_init;
    end
    if ~isfield(options, 'step_alg')
        step_alg = 'fix';
    else
        if strcmp(options.step_alg, 'decay')
            step_alg = 'decay';
        elseif strcmp(options.step_alg, 'decay-2')
            step_alg = 'decay-2';
        elseif strcmp(options.step_alg, 'decay-3')
            step_alg = 'decay-3';
        elseif strcmp(options.step_alg, 'decay-epoch')
            step_alg = 'decay-epoch';
        elseif strcmp(options.step_alg, 'fix-decay-fix')
            step_alg = 'fix-decay-fix';
        elseif strcmp(options.step_alg, 'fix-decay-fix-decay')
            step_alg = 'fix-decay-fix-decay';
        elseif strcmp(options.step_alg, 'fix')
            step_alg = 'fix';
        else
            step_alg = 'decay';
        end
    end
    if ~isfield(options, 'lambda')
        lambda = 0.1;
    else
        lambda = options.lambda;
    end
    if strcmp(step_alg, 'fix')
        step = step_init;
    elseif strcmp(step_alg, 'decay')
        step = step_init / (1 + step_init * lambda * iter);
    elseif strcmp(step_alg, 'decay-2')
        step = step_init / (1 + iter);
    elseif strcmp(step_alg, 'decay-3')
        step = step_init / (lambda + iter);
    elseif strcmp(step_alg, 'decay-epoch')
        step = step_init / (1 + epoch);
    elseif strcmp(step_alg, 'fix-decay-fix')
        first_decay_epoch = options.first_decay_epoch;
        if epoch >= first_decay_epoch
            step = step_init / 10;
        else
            step = step_init;
        end
    elseif strcmp(step_alg, 'fix-decay-fix-decay')
        first_decay_epoch = options.first_decay_epoch;
        second_decay_epoch = options.second_decay_epoch;
        if (epoch >= first_decay_epoch) && (epoch < second_decay_epoch)
            step = step_init / 5;
        elseif epoch >= second_decay_epoch
            step = step_init / 100;
        else
            step = step_init;
        end
    end
end