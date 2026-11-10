function [x_train,y_train,x_test,y_test,trainini_y,testini_y] = mnist_fixed_small_format(train_x_file, train_y_file, test_x_file, test_y_file)
%MNIST_FIXED_SMALL_FORMAT
% Load fixed small MNIST .mat files and convert them to the same format as
% mnist_datanum.m.
%
% Output format:
%   x_train     : 784 x N_train
%   y_train     : 10 x N_train one-hot labels
%   x_test      : 784 x N_test
%   y_test      : 10 x N_test one-hot labels
%   trainini_y  : N_train x 1 labels in 1..10
%   testini_y   : N_test x 1 labels in 1..10
%
% This function does NOT randomly select samples.
% It uses exactly the samples already saved in the four small .mat files.

    % ------------------------------------------------------------
    % Load arrays
    % ------------------------------------------------------------
    train_x_raw = load_first_array(train_x_file);
    train_y_raw = load_first_array(train_y_file);
    test_x_raw  = load_first_array(test_x_file);
    test_y_raw  = load_first_array(test_y_file);

    % ------------------------------------------------------------
    % Convert x data to N x 784 first
    % ------------------------------------------------------------
    train_x_raw = ensure_samples_by_features(train_x_raw, 'train_x');
    test_x_raw  = ensure_samples_by_features(test_x_raw,  'test_x');

    % ------------------------------------------------------------
    % Convert labels to column vectors
    % ------------------------------------------------------------
    train_y_raw = train_y_raw(:);
    test_y_raw  = test_y_raw(:);

    % ------------------------------------------------------------
    % Check sample counts
    % ------------------------------------------------------------
    if size(train_x_raw,1) ~= length(train_y_raw)
        error('train_x and train_y sample counts do not match: %d vs %d', ...
              size(train_x_raw,1), length(train_y_raw));
    end

    if size(test_x_raw,1) ~= length(test_y_raw)
        error('test_x and test_y sample counts do not match: %d vs %d', ...
              size(test_x_raw,1), length(test_y_raw));
    end

    % ------------------------------------------------------------
    % Convert labels to MATLAB class labels 1..10
    % ------------------------------------------------------------
    trainini_y = convert_labels_to_1_to_10(train_y_raw);
    testini_y  = convert_labels_to_1_to_10(test_y_raw);

    % ------------------------------------------------------------
    % Scale image values if needed
    % ------------------------------------------------------------
    train_x_raw = double(train_x_raw);
    test_x_raw  = double(test_x_raw);

    % If data are 0..255, scale to 0..1.
    % If data are already 0..1, do not divide again.
    if max(train_x_raw(:)) > 1.5
        train_x_raw = train_x_raw / 255;
    end

    if max(test_x_raw(:)) > 1.5
        test_x_raw = test_x_raw / 255;
    end

    % ------------------------------------------------------------
    % Transpose to match mnist_datanum output:
    %   784 x number_of_samples
    % ------------------------------------------------------------
    x_train = train_x_raw';
    x_test  = test_x_raw';

    % ------------------------------------------------------------
    % Create one-hot label matrices
    % ------------------------------------------------------------
    K = 10;

    N_train = length(trainini_y);
    N_test  = length(testini_y);

    y_train = zeros(K, N_train);
    y_test  = zeros(K, N_test);

    for j = 1:N_train
        y_train(trainini_y(j), j) = 1;
    end

    for j = 1:N_test
        y_test(testini_y(j), j) = 1;
    end

    % ------------------------------------------------------------
    % Print summary
    % ------------------------------------------------------------
    fprintf('\nFixed small MNIST data loaded and converted.\n');
    fprintf('x_train size:    %d x %d\n', size(x_train,1), size(x_train,2));
    fprintf('y_train size:    %d x %d\n', size(y_train,1), size(y_train,2));
    fprintf('x_test size:     %d x %d\n', size(x_test,1), size(x_test,2));
    fprintf('y_test size:     %d x %d\n', size(y_test,1), size(y_test,2));
    fprintf('trainini_y size: %d x %d\n', size(trainini_y,1), size(trainini_y,2));
    fprintf('testini_y size:  %d x %d\n', size(testini_y,1), size(testini_y,2));

    fprintf('\nTraining label distribution:\n');
    disp(tabulate(trainini_y));

    fprintf('Testing label distribution:\n');
    disp(tabulate(testini_y));

end


% ========================================================================
% Helper function: load first non-system variable from .mat file
% ========================================================================
function array = load_first_array(mat_file)
    data = load(mat_file);
    names = fieldnames(data);

    for i = 1:length(names)
        name = names{i};

        if startsWith(name, '__')
            continue;
        end

        array = data.(name);
        fprintf('Using variable "%s" from %s\n', name, mat_file);
        return;
    end

    error('No usable variable found in %s', mat_file);
end


% ========================================================================
% Helper function: make x shape N x 784
% ========================================================================
function x = ensure_samples_by_features(x, name)
    x = double(x);

    if ndims(x) ~= 2
        error('%s must be a 2D matrix.', name);
    end

    if size(x,2) == 784
        % Already N x 784
        return;
    elseif size(x,1) == 784
        % Convert 784 x N to N x 784
        x = x';
    else
        error('%s must have shape N x 784 or 784 x N. Got %d x %d.', ...
              name, size(x,1), size(x,2));
    end
end


% ========================================================================
% Helper function: convert labels to 1..10
% ========================================================================
function y_out = convert_labels_to_1_to_10(y)
    y = double(y(:));

    labels = unique(y);

    % Case 1: labels are 0..9
    if min(labels) == 0 && max(labels) == 9
        y_out = y + 1;
        return;
    end

    % Case 2: labels are already 1..10
    if min(labels) == 1 && max(labels) == 10
        y_out = y;
        return;
    end

    error('Labels must be either 0..9 or 1..10. Found labels:');
    disp(labels');
end