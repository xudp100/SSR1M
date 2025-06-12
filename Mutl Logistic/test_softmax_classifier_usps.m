function test_softmax_classifier_usps()
    clc;
    close all;
    algorithms = {'SSR1M'};
    z = [899475];
    for i = 1:length(z)
        rng(z(i))
        epoch_base = 1;
        [y_in,x_in] = libsvmread('data/usps');
        [yt_in,xt_in] = libsvmread('data/usps.t');
        x_in = full(x_in)';
        y_in = y_in';
        xt_in = full(xt_in)';
        yt_in = yt_in';
        n = length(y_in);
        nt = length(yt_in);
        n_train = floor(n);
        nt_train = floor(nt);
        x_train = x_in(:,1:n_train)';
        y_train = y_in(1:n_train)';
        x_test = xt_in(:,1:nt_train)';
        y_test = yt_in(1:nt_train)';
        K = 10;
        epoch_ssr1 = epoch_base * 10;
        batch_size = 512;
        lambda = 0.01;
        problem = custom_softmax_regression(x_train, y_train, x_test, y_test, K, lambda);
        f_opt = 0;
        variance = 0.02;
        w_init = randn(problem.dim, 1) * sqrt(variance);
        w_list = cell(length(algorithms),1);
        info_list = cell(length(algorithms),1);
        train_acc = cell(length(algorithms),1);
        lr_ssr1 = 0.001;
        for alg_idx=1:length(algorithms)
            fprintf('\n\n### [%02d] %s ###\n\n', alg_idx, algorithms{alg_idx});
            clear options;
            options.w_init = w_init;
            options.batch_size = batch_size;
            options.batch_hess_size = batch_size * 20;
            options.tol_optgap = 10^-36;
            options.verbose = true;
            options.lambda = lambda;
            options.permute_on = 1;
            options.f_opt = f_opt;
            switch algorithms{alg_idx}
                case {'SSR1M'}
                    options.max_epoch = epoch_ssr1;
                    options.batch_size = batch_size;
                    options.sub_mode = 'SSR1M';
                    options.beta1 = 0.9;
                    options.theta = 0.001;
                    options.epsilon = 1e-8;
                    options.step_init = lr_ssr1;
                    options.step_alg = 'fix';
                    algorithm_func = @SSR1M;
                    algorithm_name = 'SSR1M';           
                otherwise
                    warn_str = [algorithms{alg_idx}, ' is not supported.'];
                    warning(warn_str);
                    w_list{alg_idx} = '';
                    info_list{alg_idx} = '';
                    accuracy_list = '';
                    return;
            end
            [w_list{alg_idx}, info_list{alg_idx}, train_acc, test_acc] = algorithm_func(problem, options, z(i));
        end
    end
    fprintf('\n\n');
end