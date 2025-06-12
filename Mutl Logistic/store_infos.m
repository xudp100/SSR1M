function [infos, train_loss, test_loss, optgap, grad, gnorm, subgrad, subgnorm, smooth_grad, smooth_gnorm] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time, iter)
    subgrad = [];
    subgnorm = [];
    if ~epoch
        infos.epoch = epoch;
        infos.iter = iter;
        infos.time = 0;
        infos.grad_calc_count = grad_calc_count;
        train_loss = problem.train_cost(w);
        test_loss = problem.test_cost(w);
        optgap = train_loss - options.f_opt;
        grad = problem.full_grad(w);
        gnorm = norm(grad);
        if ismethod(problem, 'full_subgrad')
            subgrad = problem.full_subgrad(w);
            subgnorm = norm(subgrad);
            infos.subgnorm = subgnorm;
        end
        if ismethod(problem, 'full_smooth_grad')
            smooth_grad = problem.full_smooth_grad(w);
            smooth_gnorm = norm(smooth_grad);
            infos.smooth_gnorm = smooth_gnorm;
        end
        infos.optgap = optgap;
        infos.best_optgap = optgap;
        infos.absoptgap = abs(optgap);
        infos.gnorm = gnorm;
        infos.train_loss = train_loss;
        infos.test_loss = test_loss;
        infos.best_train_cost = train_loss;
        infos.best_test_cost = test_loss;
        if ismethod(problem, 'reg')
            infos.reg = problem.reg(w);
        end
        if options.store_w
            infos.w = w;
        end
        if options.store_grad
            infos.grad = grad;
        end
    else
        infos.epoch = [infos.epoch epoch];
        infos.iter = [infos.iter iter];
        infos.time = [infos.time elapsed_time];
        infos.grad_calc_count = [infos.grad_calc_count grad_calc_count];
        train_loss = problem.train_cost(w);
        test_loss = problem.test_cost(w);
        optgap = train_loss - options.f_opt;
        grad = problem.full_grad(w);
        gnorm = norm(grad);
        if ismethod(problem, 'full_subgrad')
            subgrad = problem.full_subgrad(w);
            subgnorm = norm(subgrad);
            infos.subgnorm = [infos.subgnorm subgnorm];
        end
        if ismethod(problem, 'full_smooth_grad')
            smooth_grad = problem.full_smooth_grad(w);
            smooth_gnorm = norm(smooth_grad);
            infos.smooth_gnorm = [infos.smooth_gnorm smooth_gnorm];
        end
        infos.optgap = [infos.optgap optgap];
        infos.absoptgap = [infos.absoptgap abs(optgap)];
        if optgap < infos.best_optgap(end)
            infos.best_optgap = [infos.best_optgap optgap];
        else
            infos.best_optgap = [infos.best_optgap infos.best_optgap(end)];
        end
        infos.train_loss = [infos.train_loss train_loss];
        if train_loss < infos.best_train_cost(end)
            infos.best_train_cost = [infos.best_train_cost train_loss];
        else
            infos.best_train_cost = [infos.best_train_cost infos.best_train_cost(end)];
        end
        infos.test_loss = [infos.test_loss test_loss];
        if test_loss < infos.best_test_cost(end)
            infos.best_test_cost = [infos.best_test_cost test_loss];
        else
            infos.best_test_cost = [infos.best_test_cost infos.best_test_cost(end)];
        end
        infos.gnorm = [infos.gnorm gnorm];
        if ismethod(problem, 'reg')
            reg = problem.reg(w);
            infos.reg = [infos.reg reg];
        end
        if options.store_w
            infos.w = [infos.w w];
        end
        if options.store_grad
            infos.grad = [infos.grad grad];
        end
    end
end