classdef custom_softmax_regression
    properties
        name = 'custom_softmax';
        lambda;
        d;
        dim;
        samples;
        n_train;
        n_test;
        num_class;
        x_train;
        y_train;
        x_test;
        y_test;
    end

    methods
        function obj = custom_softmax_regression(x_train, y_train, x_test, y_test, num_class, varargin)
            if nargin < 6
                lambda = 0.1;
            else
                lambda = varargin{1};
            end
            obj.lambda   = lambda;
            obj.x_train  = x_train;
            obj.y_train  = y_train;
            obj.x_test   = x_test;
            obj.y_test   = y_test;
            obj.n_train  = size(x_train, 1);
            obj.n_test   = size(x_test, 1);
            obj.d        = size(x_train, 2);
            obj.num_class= num_class;
            obj.samples  = obj.n_train;
            obj.dim      = obj.d * obj.num_class;
        end

        function f = train_cost(obj, w)
            W      = reshape(w, [obj.d, obj.num_class]);
            Z      = obj.x_train * W;
            Y_pred = softmax(Z')';
            labels = double(obj.y_train(:));
            rows   = (1:obj.n_train)';
            if numel(rows) == numel(labels)
                Y_hot      = zeros(obj.n_train, obj.num_class);
                idx        = sub2ind(size(Y_hot), rows, labels);
                Y_hot(idx) = 1;
            else
                error('Rows and labels dimensions do not match.');
            end
            loss = -sum(sum(Y_hot .* log(Y_pred + 1e-8))) / obj.n_train;
            reg  = obj.lambda * sum(sum((W.^2) ./ (1 + W.^2)));
            f    = loss + reg;
        end

        function g = grad(obj, w, indices)
            if nargin < 3
                indices = 1:obj.n_train;
            end
            X       = obj.x_train(indices, :);
            Y       = obj.y_train(indices);
            W       = reshape(w, [obj.d, obj.num_class]);
            Y_hot   = full(ind2vec(double(Y(:))', obj.num_class))';
            Y_pred  = softmax((X * W)')';
            diff    = Y_pred - Y_hot;
            grad_core = X' * diff / length(indices);
            reg_grad  = 2 * obj.lambda * W ./ ((1 + W.^2).^2);
            g         = grad_core + reg_grad;
            g         = g(:);
        end

        function f = test_cost(obj, w)
            W           = reshape(w, [obj.d, obj.num_class]);
            Z           = obj.x_test * W;
            Y_test_pred = softmax(Z')';
            labels      = double(obj.y_test(:));
            rows        = (1:obj.n_test)';
            if numel(rows) == numel(labels)
                Y_hot      = zeros(obj.n_test, obj.num_class);
                idx        = sub2ind(size(Y_hot), rows, labels);
                Y_hot(idx) = 1;
            else
                error('Rows and labels dimensions do not match.');
            end
            loss = -sum(sum(Y_hot .* log(Y_test_pred + 1e-8))) / obj.n_test;
            reg  = obj.lambda * sum(sum((W.^2) ./ (1 + W.^2)));
            f    = loss + reg;
        end

        function g = full_grad(obj, w)
            g = obj.grad(w, 1:obj.n_train);
        end

        function train_preds = train_prediction(obj, w)
            W            = reshape(w, [obj.d, obj.num_class]);
            Z            = obj.x_train * W;
            Y_train_pred = softmax(Z, 2);
            [~, train_preds] = max(Y_train_pred, [], 2);
        end

        function test_preds = test_prediction(obj, w)
            W            = reshape(w, [obj.d, obj.num_class]);
            Z            = obj.x_test * W;
            Y_test_pred  = softmax(Z, 2);
            [~, test_preds] = max(Y_test_pred, [], 2);
        end

        function test_a = test_accuracy(obj, preds)
            test_a = sum(preds == obj.y_test) / obj.n_test * 100;
        end

        function train_a = train_accuracy(obj, preds)
            train_a = sum(preds == obj.y_train) / obj.n_train * 100;
        end

        function [w_opt, info] = calc_solution(obj, maxiter, method)
            options.max_iter = maxiter;
            options.step_alg = 'backtracking';
            w_init = zeros(obj.dim, 1);
            if strcmp(method, 'sd')
                [w_opt, info] = sd(obj, options, w_init);
            elseif strcmp(method, 'lbfgs')
                [w_opt, info] = lbfgs(obj, options, w_init);
            else
                error('Unsupported method: %s', method);
            end
        end
    end
end