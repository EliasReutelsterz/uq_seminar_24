clear all;


% Run the examples and so on here:
% example_7_31()
%example_7_42_A_entries_bigger()
example_7_42_increasing_m()


%% Example 7.22 (symmetric BTTB matrices)
% just the first case (7.13, 7.14)

function example_7_22()
    C_0 = [1, 0.2, 0.1; 0.2, 1, 0.2; 0.1, 0.2, 1];
    C_1 = [0.5, 0.1, 0.8; 0.2, 0.5, 0.1; 0.6, 0.2, 0.5];
    C = [C_0, C_1.'; C_1, C_0];
    C_red = [0.6, 0.1, 0.8; 0.2, 0.2, 0.1; 0.5, 1, 0.5; 0.1, 0.2, 0.2; 0.8, 0.1, 0.6];
end

%% Example 7.24 (symmetric BCCB matrices)
% just the first case (7.18, 7.19)

function example_7_24()
    C_0 = [1, 0.2, 0.2; 0.2, 1, 0.2; 0.2, 0.2, 1];
    C_1 = [0.5, 0.2, 0.1; 0.1, 0.5, 0.2; 0.2, 0.1, 0.5];
    C = [C_0, C_1.', C_1; C_1, C_0, C_1.'; C_1.', C_1, C_0];
    format bank;
    disp('C:');
    disp(C);
    C_red = [1, 0.5, 0.5; 0.2, 0.1, 0.2; 0.2, 0.2, 0.1];
    disp('C_red:');
    disp(C_red);
end

%% Example 7.30 (eigenvalues of a symmetric BCCB matrix)
% For the symmetric BCCB matrix in Example 7.24, compute eigenvalues of C
% with inverse DFT to C_red

function example_7_30()
    C_red = [1, 0.5, 0.5; 0.2, 0.1, 0.2; 0.2, 0.2, 0.1];
    Lambda = 9 * ifft2(C_red);
    disp(Lambda(:))
end
%% Example 7.31
% The following symmetric BCCB matrix C is a valid covariance matrix and
% using Algorithm 7.1 can generate two independent samples from N(0, C) as
% follows

function example_7_31()
    C_red = [1, 0.5, 0.5; 0.2, 0.1, 0.2; 0.2, 0.2, 0.1];
    [X, Y] = circ_cov_sample_2d(C_red, 3, 3);
    X = X(:);
    Y = Y(:);
    disp(X);
end

%% Algorithm 7.1
% Code to generate a pair of samples from N(0, C) where C \in R^{N \times
% N} (N = n_1 n_2) is a BCCB covariance matrix. The input C_red is the
% reduced matrix which generates C, and X, Y are independent samples from
% N(0, C)

function [X, Y] = circ_cov_sample_2d(C_red, n1, n2)
    N = n1 *n2;
    Lambda = N * ifft2(C_red);
    d = Lambda(:);
    d_minus = max(-d, 0);
    if (max(d_minus) > 0)
        fprintf('Invalid covariance; Smallest eigenvalue rho(D_minus) = %0.5g \n', -max(d_minus));
    end
    xi = randn(n1, n2) + 1i .* randn(n1, n2); % where 1i is the imaginary unit
    V = (Lambda.^0.5) .* xi;
    Z = fft2(V) / sqrt(N);
    Z = Z(:);
    X = real(Z);
    Y = imag(Z);
end

%% Example 7.32 (circulant extensions)
% Consider the 3 x 3 Toeplitz matrix C generated by c = [3, 2, 1, 4, 5]^T.
% Its minimal circulant extension is the 5 x 5 circulant matrix C_tilde
% generated by c_1 = [1, 4, 5, 3, 2]^T, which contains the entries of the
% first column of C, followed by the first row, in reverse order:

function example_7_32()
    C = [1, 2, 3; 4, 1, 2; 5, 4, 1];
    C_tilde = [1, 2, 3, 5, 4; 4, 1, 2, 3, 5; 5, 4, 1, 2, 3; 3, 5, 4, 1, 2; 2, 3, 5, 4, 1];
    x = 0;
    % neither C nor C_tilde is symmetric. Now use an arbitrary x for:
    C_tilde = [1, 2, 3, x, 5, 4; 4, 1, 2, 3, x, 5; 5, 4, 1, 2, 3, x; x, 5, 4, 1, 2, 3; 3, x, 5, 4, 1, 2; 2, 3, x, 5, 4, 1];
    disp(C_tilde)
end


%% Example 7.34 (even circulant extension)
% Consider the 3 x 3 Toeplitz matrices C_0 and C_1 in Example 7.22. C_0 is
% symmetric but C_1 is not. Using Definition 7.33 (even circular extension)
% as follows:

function example_7_34()
    C_0 = [1, 0.2, 0.1; 0.2, 1, 0.2; 0.1, 0.2, 1];
    B_0 = [0, 0.1, 0.2; 0.1, 0, 0.1; 0.2, 0.1, 0];
    C_tilde_0 = [C_0, B_0; B_0, C_0];
    C_1 = [0.5, 0.1, 0.8; 0.2, 0.5, 0.1; 0.6, 0.2, 0.5];
    B_1 = [0, 0.6, 0.2; 0.8, 0, 0.6; 0.1, 0.8, 0];
    C_tilde_1 = [C_1, B_1; B_1, C_1];
end

%% Example 7.36 (even BCCB extension)
% Consider the symmetric BCCB matrix C from Example 7.22, which can be
% generated by C_red. Its BCCB extension is the symmetric 24 x 24 BCCB
% matrix C_tilde. To store C_tilde we only need C_tilde_red

function example_7_36()
    C_0 = [1, 0.2, 0.1; 0.2, 1, 0.2; 0.1, 0.2, 1];
    C_1 = [0.5, 0.1, 0.8; 0.2, 0.5, 0.1; 0.6, 0.2, 0.5];
    C = [C_0, C_1.'; C_1, C_0];
    C_red = [0.6, 0.1, 0.8; 0.2, 0.2, 0.1; 0.5, 1, 0.5; 0.1 ,0.2, 0.2; 0.8, 0.1, 0.6];
    B_0 = [0, 0.1, 0.2; 0.1, 0, 0.1; 0.2, 0.1, 0];
    B_1 = [0, 0.6, 0.2; 0.8, 0, 0.6; 0.1, 0.8, 0];
    C_tilde_0 = [C_0, B_0; B_0, C_0];
    C_tilde_1 = [C_1, B_1; B_1, C_1];
    null_tilde = zeros(6);
    C_tilde = [C_tilde_0, C_tilde_1.', null_tilde, C_tilde_1; C_tilde_1, C_tilde_0, C_tilde_1.', null_tilde; null_tilde, C_tilde_1, C_tilde_0, C_tilde_1.'; C_tilde_1.', null_tilde, C_tilde_1, C_tilde_0];
    C_tilde_red = [1, 0.5, 0, 0.5; 0.2, 0.2, 0, 0.1; 0.1, 0.6, 0, 0.8; 0, 0, 0, 0; 0.1, 0.8, 0, 0.6; 0.2, 0.1, 0, 0.2];
end

%% Example 7.38 (BTTB covariance matrix
% Let D = [0,1]^2 and set n_1 = 3 and n_2 = 2. Then delta_x_1 = 1/2 and
% delta_x_2 = 1 and the sample points are x_0 = [0,0]^T, x_1 = [1/2, 0]^T,
% ..., x_5 = [1, 1]^T. The covariance matrix of u(x) where u(x) is a
% stationary process with mean zero and stationary covariance c(x) is:

% Too much to write... but can use Algorithm 7.2 for creating C_red at
% least

%% Algorithm 7.2
% Code to generate the reduced symmetric BTTB covariance matrix C_red
% associated with the stationary coavriance function fhandle. The inputs
% are n1, n2 the number of sample points in each direction, and dx1, dx2,
% the grid spacings delta x1 and delta x2.

function C_red = reduced_cov(n1, n2, dx1, dx2, fhandle)
    C_red = zeros(2*n1 - 1, 2*n2 - 1);
    for i = 1:2*n1-1
        for j = 1:2*n2-1
            C_red(i, j) = feval(fhandle, (i-n1)*dx1, (j-n2)*dx2);
        end
    end
end

%% Algorithm 7.3
% Code to evaluate the stationary coariance c(x) in Example 7.40 at the
% point (x1, x2)

function c = gaussA_exp(x1, x2, a11, a22, a12)
    c = exp(-((x1^2*a11+x2^2*a22) - 2*x1*x2*a12));
end

%% Example 7.40 (reduced BTTB covariance matrix)
% Let u(x) be a mean-zero random field with stationary covariance function
% c(x), let A be positive definite. We can compute the 5 x 3 reduced
% covariance matrix using Algorithms 7.2 and 7.3 as follows

function example_7_40()
    fhandle =@(x1, x2)gaussA_exp(x1, x2, 1, 1, 0.5);
    C_red = reduced_cov(3, 2, 1/2, 1, fhandle);
    disp(C_red)
end

%% Algorithm 7.4 
% Code to generate a pair of samples from N(0, C) where C in R^NxN (N =
% n1*n2) is a BTTB covariance matrix. The input C_red is the reduced matrix
% which generates C. The outputs u1, u2 are independent samples from
% N(0,C), in matrix format.
function [u1, u2] = circ_embed_sample_2d(C_red, n1, n2)
    N =n1*n2;
    % form reduced matrix of BCCB extension of BTTB matrix C
    tilde_C_red = zeros(2*n1, 2*n2);
    tilde_C_red(2:2*n1, 2:2*n2) = C_red;
    tilde_C_red = fftshift(tilde_C_red);
    % sample from N(0, tilde_C)
    [u1, u2] = circ_cov_sample_2d(tilde_C_red, 2*n1, 2*n2);
    % recover samples from N(0, C)
    u1 =u1(:);
    u2 = u2(:);
    u1 = u1(1:end/2);
    u1 = reshape(u1, n1, 2*n2);
    u1 = u1(:, 1:2:end);
    u2 = u2(1:end/2);
    u2 = reshape(u2, n1, 2*n2);
    u2 = u2(:, 1:2:end);
end

%% Algorithm 7.5
% Code to evaluate the stationary covariance c(x) at the point (x1, x2)
% with c(x) = exp(|x1|/l1 - |x2|/l2)

function c = sep_exp(x1, x2, l1, l2)
    c = exp(-abs(x1) / l1 - abs(x2) / l2);
end

%% Example 7.41 
% Let u(x) be a mean-zero Gaussian random field in two dimensions with
% stationary covariance c(x) = exp(|x1|/l1 - |x2|/l2). Let l1 = 1/5, l2 =
% 1/10 on a uniformly spaced grid on D = [0,1] x [0,2] with n1 = 201 and n2
% = 401. The resulting covariance matrix C has an even BCCB extension
% tilde_C which is non-negative definite.

function example_7_41()
    fhandle1 =@(x1, x2)sep_exp(x1, x2, 1/5, 1/10);
    C_red = reduced_cov(201, 401, 1/200, 1/200, fhandle1);
    [u1, u2] = circ_embed_sample_2d(C_red, 201, 401);
    x = linspace(0, 1, size(u1, 2));
    y = linspace(0, 2, size(u1, 1));

    figure;
    num_levels = 10;
    contourf(x, y, u1, num_levels, 'LineColor', 'none');
    %title('Heatmap of u1');
    xlabel('X', 'FontSize', 24);
    ylabel('Y', 'FontSize', 24);
    ax = gca;
    ax.FontSize = 24;
    %colorbar;
    colormap(jet);
    saveas(gcf,['circulant_embedding_example7_41.png'])
end


%% Algorithm 7.6
% Code to generate a pair of samples from N(0, C) where C in R^NxN (N
% =n1*n2) is symmetric, non-negative definite and BTTB. The input C_red is
% the reduced matrix which generates C. n1 and n2 provide the dimensions of
% C, m1 and m2 are the padding parameters. The outputs u1, u2 are
% independent samples from N(0,C), in matrix format

function [u1, u2] = circ_embed_sample_2dB(C_red, n1, n2, m1, m2)
    nn1 = n1+m1;
    nn2 = n2+m2;
    N = nn1*nn2;
    % form reduced matrix of BCCB extension of BTTB matrix C*
    tilde_C_red = zeros(2*nn1, 2*nn2);
    tilde_C_red(2:2*nn1, 2:2*nn2) = C_red;
    tilde_C_red = fftshift(tilde_C_red);
    % sample from N(0, tilde_C)
    [u1, u2] = circ_cov_sample_2d(tilde_C_red, 2*nn1, 2*nn2);
    %recover samples from N(0,C)
    u1 = u1(:);
    u2 = u2(:);
    u1 = u1(1:2*nn1*n2);
    u1 = reshape(u1, nn1, 2*n2);
    u1 = u1(1:n1, 1:2:end);
    u2 = u2(1:2*nn1*n2);
    u2 = reshape(u2, nn1, 2*n2);
    u2 = u2(1:n1, 1:2:end);
end

%% Example 7.42 (Gaussian covariance)
% Let u(x) be a mean-zero Gaussian random field with the stationary
% covariance c(x) and choose A.
% Consider sampling u(x) on [0,1]x[0,1] at the vertices of a 256 x 256
% uniform square grid. Then n1 = 257 = n2. The even BCCB extension with
% padding parameters m1 = 0 = m2 is indefinite.

function example_7_42()
    fhandle =@(x1, x2)gaussA_exp(x1, x2, 10, 10, 0);
    n1 = 257;
    n2 = 257;
    m1 = 0;
    m2 = 0;
    dx1= 1/(n1-1);
    dx2 = 1/(n2-1);
    C_red = reduced_cov(n1+m1, n2+m2, dx1, dx2, fhandle);
    [u1, u2] = circ_embed_sample_2dB(C_red, n1, n2, m1, m2);
    x = linspace(0, 1, size(u1, 2));
    y = linspace(0, 1, size(u1, 1));
    figure;
    num_levels = 10;
    contourf(x, y, u1, num_levels, 'LineColor', 'none');
    title('Heatmap of u1');
    xlabel('x');
    ylabel('y');
    colorbar;
    colormap(jet);
end

function example_7_42_increasing_m()
    fhandle =@(x1, x2)gaussA_exp(x1, x2, 4, 4, 0);
    % fhandle =@(x1, x2)gaussA_exp(x1, x2, 10, 10, 0);
    n1 = 257;
    n2 = 257;
    m1 = 0;
    m2 = 0;
    %m1 = 8 * n1;
    %m2 = 8 * n2;
    dx1= 1/(n1-1);
    dx2 = 1/(n2-1);
    C_red = reduced_cov(n1+m1, n2+m2, dx1, dx2, fhandle);
    [u1, u2] = circ_embed_sample_2dB(C_red, n1, n2, m1, m2);
    x = linspace(0, 1, size(u1, 2));
    y = linspace(0, 1, size(u1, 1));
    figure;
    num_levels = 10;
    contourf(x, y, u1, num_levels, 'LineColor', 'none');
    %title(['Heatmap of u1, m = ', num2str(m1)]);
    xlabel('X', 'FontSize', 24);
    ylabel('Y', 'FontSize', 24);
    ax = gca;
    ax.FontSize = 24;
    %colorbar;
    colormap(jet);
    disp(m1)
    saveas(gcf,['circulant_embedding_example7_42_padding_1.png'])
end

function example_7_42_A_entries_bigger()
    %sigma = 10;
    %sigma = 100;
    sigma = 1000;
    fhandle =@(x1, x2)gaussA_exp(x1, x2, sigma, sigma, 0);
    n1 = 257;
    n2 = 257;
    m1 = 0;
    m2 = 0;
    dx1= 1/(n1-1);
    dx2 = 1/(n2-1);
    C_red = reduced_cov(n1+m1, n2+m2, dx1, dx2, fhandle);
    [u1, u2] = circ_embed_sample_2dB(C_red, n1, n2, m1, m2);
    x = linspace(0, 1, size(u1, 2));
    y = linspace(0, 1, size(u1, 1));
    figure;
    num_levels = 10;
    contourf(x, y, u1, num_levels, 'LineColor', 'none');
    xlabel('X', 'FontSize', 24);
    ylabel('Y', 'FontSize', 24);
    ax = gca;
    ax.FontSize = 24;
    %title('Heatmap of u1');
    colorbar;
    colormap(jet);
    saveas(gcf,['circulant_embedding_example7_42_3.png'])
end








