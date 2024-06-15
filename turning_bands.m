clear all;


% Run the examples and so on here:
% example_figure_10()
figure_7_14()

%% Algorithm 7.7
% Code to sample v(x) at a rectangular grid of points [x1, x2]^T
% in R^2 with e ~ U(S^1) and X(t) = 1/sqrt(2) * cos(t) * xi1 + 1/sqrt(2) *
% sin(t) * xi2. The inputs grid1 and grid2 are column vectors of
% coordinates of x1 and x2. The output v is a matrix with entries v(x1, x2)

function [v, e, X, t] = turn_band_simple(grid1, grid2)
    theta = 2 * pi * rand;
    e = [cos(theta); sin(theta)]; %sample e
    [xx, yy] = ndgrid(grid1, grid2);
    tt = [xx(:), yy(:)] * e; %project
    xi = randn(2,1);
    v = sqrt(1/2) * [cos(tt), sin(tt)] * xi; %sample v
    v = reshape(v, length(grid1), length(grid2)).';
    v = v.';
    t = [0: 0.05: 10];
    X = 1/sqrt(2) * cos(t) * xi(1) + 1/sqrt(2) * sin(t) * xi(2);
end

%% Example Figure 7.10

function example_figure_7_10()
    grid = [0:0.05:10]';
    [u, e, X, t] = turn_band_simple(grid, grid);
    x = linspace(0, 10, size(u, 1));
    y = linspace(0, 10, size(u, 1));
    figure;
    num_levels = 10;
    contourf(x, y, u, num_levels, 'LineColor', 'none');
    xlabel('x1', 'FontSize', 24);
    ylabel('x2', 'FontSize', 24);
    ax = gca;
    ax.FontSize = 24;
    % title('Heatmap of u');
    colorbar;
    colormap(jet);
    saveas(gcf,'turning_bands_sample_e_3.png')
    figure;
    plot(t, X, 'LineWidth', 3);
    % title('Stationary process X(t)');
    xlabel('t', 'FontSize', 24);
    ylabel('X(t)', 'FontSize', 24);
    ax = gca;
    ax.FontSize = 24;
    saveas(gcf,'turning_bands_sample_e_2.png')

    % Create a figure
    figure;
    hold on;
    axis equal;
    
    % Plot the unit circle
    theta = linspace(0, 2*pi, 100);
    x = cos(theta);
    y = sin(theta);
    plot(x, y, 'b-', 'LineWidth', 2.5);
    
    % Plot the vector
    quiver(0, 0, e(2), e(1), 'r', 'LineWidth', 3, 'MaxHeadSize', 0.5); 
    
    % Set axis limits
    axis([-1.5 1.5 -1.5 1.5]);
    
    % Add labels and title
    xlabel('x1', 'FontSize', 24);
    ylabel('x2', 'FontSize', 24);
    ax = gca;
    ax.FontSize = 24;
    % title('Vector e in Unit Circle');
    saveas(gcf,'turning_bands_sample_e_1.png')
    
    hold off;
end

%% Algorithm 7.8
% Code to sample v_M(x) = 1/sqrt(M) * sum_{j=1}^M X_j(x^Te_j) at a
% rectangular grid of points [x_1, x_2, x_3]^T in R^3 when X(t) is the
% mean-zero Gaussian process with covariance c_x(t) = T_3 * exp(-t/l). The
% inputs grid1, grid2, grid3 are column vectors of coordinates x1, x2, x3.
% The circulant embedding method with interpolation is used to approximate
% X(t), with a grid of length M and padding of size Mpad. The input l sets
% the length scale l. The output v is the data array v_M(x1, x2, x3).

function v = turn_band_exp_3d(grid1, grid2, grid3, M, Mpad, l)
    [xx, yy, zz] = ndgrid(grid1, grid2, grid3); %x, y, z points
    sum = zeros(size(xx(:))); % initialise
    T = norm(max(abs([grid1, grid2, grid3])));
    gridt = -T+(2*T/(M-1)) * (0:(M+Mpad-1))'; % radius T encloses all points
    c = cov(gridt, l); % evaluate covariance
    for j = 1:M
        X = circulant_embed_approx(c); % sample using Algorithm 6.10
        e = uniform_sphere(); % sample e using 4.6
        tt = [xx(:), yy(:), zz(:)] * e; % project
        Xi = interp1(gridt, X, tt);
        sum = sum+Xi;
    end
    v = sum/sqrt(M);
    v = reshape(v, length(grid1), length(grid2), length(grid3));
end


%% Example 7.48 (isotropic exponential d = 3)

function example_7_48()
    M = 10;
    l = 0.01;
    grid = [0:0.04:1]';
    u = turn_band_exp_3d(grid, grid, grid, M, 0, l);

    % Plot the data using scatter3
    [X, Y, Z] = meshgrid(grid, grid, grid);
    X_flat = X(:);
    Y_flat = Y(:);
    Z_flat = Z(:);
    u_flat = u(:);

    figure;
    scatter3(X_flat, Y_flat, Z_flat, 36, u_flat, 'filled');
    %title(['3D Plot of Data, l = ', num2str(l), ' M = ', num2str(M)]);
    %xlabel('X', 'FontSize', 24);
    %ylabel('Y', 'FontSize', 24);
    %zlabel('Z', 'FontSize', 24);
    %ax = gca;
    %ax.FontSize = 24;
    colorbar;
    colormap(jet);
    %saveas(gcf,['turning_bands_3d_M', num2str(M), '.png'])
end

%% Algorithm 7.9
% Code to sample v_M(x) for X(t). The inputs and outputs are as in
% Algorithm 7.7 with one additional input M to specify the number of bands.

function v = turn_band_simple2(grid1, grid2, M)
    [xx, yy] = ndgrid(grid1, grid2);
    sum = zeros(size(xx(:)));
    for j = 1:M
        xi = randn(2, 1);
        theta = pi*j/M;
        e = [cos(theta); sin(theta)];
        tt = [xx(:), yy(:)] * e;
        v = sqrt(1/2) * [cos(tt), sin(tt)] * xi;
        sum = sum + v;
    end
    v = sum/sqrt(M);
    v = reshape(v, length(grid1), length(grid2));
end

%% Figure 7.13
% Example of Algorithm 7.9 like in Figure 7.13

function figure_7_13()
    grid = [0:0.05:20]';
    u1 = turn_band_simple2(grid, grid, 1);
    u2 = turn_band_simple2(grid, grid, 2);
    u3 = turn_band_simple2(grid, grid, 10);
    x = linspace(0, 20, size(u1, 1));
    y = linspace(0, 20, size(u1, 1));
    figure;
    num_levels = 10;
    subplot(1, 3, 1);
    contourf(x, y, u1, num_levels, 'LineColor', 'none');
    title('M = 1');
    xlabel('x1');
    ylabel('x2');
    colormap(jet);
    subplot(1, 3, 2);
    contourf(x, y, u2, num_levels, 'LineColor', 'none');
    title('M = 2');
    xlabel('x1');
    ylabel('x2');
    colormap(jet);
    subplot(1, 3, 3);
    contourf(x, y, u3, num_levels, 'LineColor', 'none');
    title('M = 10');
    xlabel('x1');
    ylabel('x2');
    colormap(jet);

    cb = colorbar('southoutside');
    cb.Position = [0.3 0.05 0.4 0.03];
end


%% Figure 7.14
% Example of Algorithm 7.10 like in Figure 7.14

function figure_7_14()
    grid = [0:0.05:50]';
    u1 = turn_band_wm(grid, grid, 3, 2, 1/10); % M = 3 instead of M = 5 like in figure
    u2 = turn_band_wm(grid, grid, 50, 2, 1/10);
    x = linspace(0, 50, size(u1, 1));
    y = linspace(0, 50, size(u1, 1));
    figure;
    num_levels = 10;
    subplot(1, 2, 1);
    contourf(x, y, u1, num_levels, 'LineColor', 'none');
    title('M = 3');
    xlabel('x1');
    ylabel('x2');
    colormap(jet);
    subplot(1, 2, 2);
    contourf(x, y, u2, num_levels, 'LineColor', 'none');
    title('M = 50');
    xlabel('x1');
    ylabel('x2');
    colormap(jet);

    cb = colorbar('southoutside');
    cb.Position = [0.3 0.05 0.4 0.03];
end

%% Algorithm 7.10 
% Code to approximate the mean-zero Gaussian random field u(x) with
% isotropic covariance c_q^0(r) using the turning bands method with M
% equally spaced bands and the quadrature approximation to X(t). The inputs
% and outputs are like Algorithm 7.9 with additional inputs q and l to
% specify the Whittle-Matérn parameter q and correlation length l.
% Algorithm 6.4 is used to approximate X(t)

function u = turn_band_wm(grid1, grid2, M, q, l)
    [xx, yy] = ndgrid(grid1, grid2);
    sum = zeros(size(xx(:)));
    % choose random T to contain all grid points
    T = norm([norm(grid1, inf), norm(grid2, inf)]);
    for j = 1:M
        theta = j * pi/M;
        e = [cos(theta); sin(theta)];
        tt = [xx(:), yy(:)] * e;
        [gridt, Z] = squad(2 * T, 64, 64, @(s) f(s, q, l)); % Algorithm 6.4
        Xi = interp1(gridt - T, real(Z), tt);
        sum = sum + Xi;
    end
    u = sum/sqrt(M);
    u = reshape(u, length(grid1), length(grid2));
end

%% helpers

function f = f(s, q, l) % spectral density
    f = gamma(q + 1) / gamma(q) * (l^2 * s) / (1 + (l*s)^2)^(q + 1);
end

function f = cov(t, l) % covariance given by turning bands operator
    f = (1-t/l).*exp(-t/l);
end

% Algorithm 4.6
function X = uniform_sphere()
    z = -1 + 2*rand;
    theta = 2*pi*rand;
    r = sqrt(1-z*z);
    X = [r * cos(theta); r * sin(theta); z];
end

% Algorithm 6.4
function [t, Z] = squad(T, N, M, fhandle)
    dt = T/(N - 1);
    t = dt * [0:N-1]';
    R = pi / dt;
    dnu = 2 * pi / (N*dt*M);
    Z = zeros(N, 1);
    coeff = zeros(N, 1);
    for m = 1 : M
        for k = 1 : N
            nu = R + ((k - 1) * M + (m - 1)) * dnu;
            xi = randn(1, 2) * [1; sqrt(-1)];
            coeff(k) = sqrt(fhandle(nu) * dnu) * xi;
            if ((m == 1 && k == 1) || (m == M && k == N))
                coeff(k) = coeff(k) / sqrt(2);
            end
        end
        Zi = N * ifft(coeff);
        Z = Z + exp(sqrt(-1) * (- R + (m - 1) * dnu) * t).*Zi;
    end
end

% Algorithm 6.10
function [X, Y] = circulant_embed_approx(c)
    tilde_c = [c; c(end-1:-1:2)];
    tilde_N = length(tilde_c);
    d = ifft(tilde_c, 'symmetric') * tilde_N;
    d_minus = max(-d, 0);
    d_pos = max(d, 0);
    if (max(d_minus) > 0)
        fprintf('rho(D_minus) = %0.5g\n', max(d_minus));
    end
    xi = randn(tilde_N, 2) * [1; sqrt(-1)];
    Z = fft((d_pos.^0.5).*xi) / sqrt(tilde_N);
    N = length(c);
    X = real(Z(1:N));
    Y = imag(Z(1:N));
end


















