function [w, infos, train_acc, test_acc] = SSR1M(problem, in_options, z)
    rng(z)
    d = problem.dim();
    n = problem.samples();
    local_options.beta1 = 0.9;
    local_options.beta2 = 0.9;
    local_options.epsilon = 0.9;
    options = mergeOptions(get_default_options(d), local_options);
    options = mergeOptions(options, in_options);
    s = 0;
    y = 0;
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    b = zeros(size(w));
    num_of_bachces = floor(n / options.batch_size);
    m = zeros(d, 1);
    clear infos;
    [infos, train_loss, test_loss, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0, total_iter);
    start_time = tic();
    if options.verbose > 0
        fprintf('%s: Epoch = %03d, train_loss = %.2e,  test_loss = %.2e\n', options.sub_mode, epoch, train_loss, test_loss);
    end
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
        if options.permute_on
            perm_idx = randperm(n);
        else
            perm_idx = 1:n;
        end
        for j = 1:num_of_bachces
            step = options.stepsizefun(total_iter, epoch, options);
            start_index = (j-1) * options.batch_size + 1;
            indice_j = perm_idx(start_index:start_index + options.batch_size - 1);
            grad = problem.grad(w, indice_j);
            total_iter = total_iter + 1;
            m = options.beta1 .* m + (1 - options.beta1) .* grad;
            if strcmp(options.sub_mode, 'SSR1M')
                s = s + w;
                y = y + grad;
                bb = (y - b .* s).^2;
                bb = bb / (norm(bb) + eps);
                b = options.theta * b + (1 - options.theta) * bb;
                s = -w;
                y = -grad;
                w = w - step * m ./ (sqrt(b + options.epsilon));
            end
            if ismethod(problem, 'prox')
                w = problem.prox(w, step);
            end
        end
        elapsed_time = toc(start_time);
        grad_calc_count = grad_calc_count + j * options.batch_size;
        epoch = epoch + 1;
        f_old = train_loss;
        [infos, train_loss, test_loss, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time, total_iter);
        train_preds = problem.train_prediction(w);
        train_acc = problem.train_accuracy(train_preds);
        infos.train_accuracy(epoch) = train_acc;
        test_preds = problem.test_prediction(w);
        test_acc = problem.test_accuracy(test_preds);
        infos.test_accuracy(epoch) = test_acc;
        if options.verbose > 0
            fprintf('%s: Epoch = %03d, train_loss = %.2e, test_loss = %.2e, train_acc = %.2f%%, test_acc = %.2f%%\n', options.sub_mode, epoch, train_loss, test_loss, train_acc, test_acc);
        end
    end
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epoch = %g\n', options.max_epoch);
    end
end