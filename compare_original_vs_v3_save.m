%% compare_original_vs_v3_save_ppg.m
% Purpose: Generate and automatically save fourteen figures comparing the
% Original vs. v3 implementations of CT and SCT on the PPG signal.

clear; close all; clc;

%% ————————— One, Load PPG Data and Preprocess —————————
matFilename = 'Tseg108_sub12_ep3.mat';
data = load(matFilename);

% Extract PPG signal and sampling rate
% According to the structure, we use data.ppg.target as the raw PPG waveform
signal = data.ppg.target;      % raw PPG
fs     = data.ppg.fs;          % sampling rate (Hz)

signal = signal(:);            % ensure column vector
N_total = length(signal);
fprintf('Loaded PPG signal, length %d samples, sampling rate %d Hz\n', N_total, fs);

% (Optional) Downsample
downsampleFactor = 1;
if downsampleFactor > 1
    signal = downsample(signal, downsampleFactor);
    fs = fs / downsampleFactor;
    N_total = length(signal);
    fprintf('Downsampled by %d×, new fs = %d Hz, length = %d\n', downsampleFactor, fs, N_total);
end

% Extract 10–20 second segment
t_start = 10;  % seconds
t_end   = 20;  % seconds
idx1 = max(1, round(t_start * fs));
idx2 = min(N_total, round(t_end   * fs));
signal = signal(idx1:idx2);
t_vec  = (0:length(signal)-1)'/fs + t_start;
fprintf('Segmented PPG from %.2f to %.2f s, segment length = %d samples\n', t_start, t_end, length(signal));

%% ————————— Two, Define Window Functions and Derivatives —————————
Hz = fs;
halfwin = 2;  
t_window = (-halfwin*Hz : halfwin*Hz)' / Hz;  % [-2s, 2s] support

alpha_win = 1;
% g0 (Gaussian) and its derivatives
g0   = exp(-pi * alpha_win * (t_window.^2));
dg0  = -(2*pi*alpha_win * t_window) .* g0;
ddg0 = ((2*pi*alpha_win * t_window).^2 - 2*pi*alpha_win) .* g0;

% g2 (t^2 * Gaussian) and its derivatives
g2  = (t_window.^2) .* exp(-pi * alpha_win * (t_window.^2));
dg2 = 2 * t_window .* exp(-pi*alpha_win*(t_window.^2)) + ...
      (t_window.^2) .* (-(2*pi*alpha_win*t_window) .* exp(-pi*alpha_win*(t_window.^2)));
ddg2 = diff([dg2(1); dg2]) * Hz;  % approximate second derivative

% Ensure column vectors
g0   = g0(:);   dg0  = dg0(:);   ddg0 = ddg0(:);
g2   = g2(:);   dg2  = dg2(:);   ddg2 = ddg2(:);

%% ————————— Three, Algorithm Parameters —————————
alpha_res = 2.5 / length(signal);  % resolution parameter
tDS = 1;                           % time downsampling factor

%% ————————— Four, Original CT/SCT and Save Figures —————————
% 1. Original CT (g0)
fprintf('\n— Running Original CT (g0) on PPG —\n');
[tfc0_orig, tfrtic0_orig, tcrtic0_orig, tfrsq0_orig, tfrsqtic0_orig] = ...
    sqSTCT(signal, 0, 0.5, alpha_res, tDS, g0, dg0, ddg0);
fprintf('Done Original CT (g0).\n');

midFrame = round(size(tfc0_orig, 3) / 2);

% Figure 1: Original CT (g0) at midFrame
f1 = figure('Name','orig_ct_ppg_g0_midframe');
imagesc(tfrtic0_orig*Hz, tcrtic0_orig*Hz^2, abs(squeeze(tfc0_orig(:,:,midFrame))));
axis xy; colormap(1-gray);
xlabel('Frequency (Hz)'); ylabel('Chirp Rate (Hz^2)');
title(sprintf('Original CT (g_0), PPG, t=%.2f s', t_vec(midFrame)));
drawnow;
saveas(f1, [matlab.lang.makeValidName(get(f1,'Name')) '.png']);

% Figure 2: Original SCT (g0) at midFrame
f2 = figure('Name','orig_sct_ppg_g0_midframe');
imagesc(tfrtic0_orig*Hz, tcrtic0_orig*Hz^2, abs(squeeze(tfrsq0_orig(:,:,midFrame))));
axis xy; colormap(1-gray);
xlabel('Frequency (Hz)'); ylabel('Chirp Rate (Hz^2)');
title(sprintf('Original SCT (g_0), PPG, t=%.2f s', t_vec(midFrame)));
drawnow;
saveas(f2, [matlab.lang.makeValidName(get(f2,'Name')) '.png']);

% Figure 3: Original SCT (g0) time-frequency projection
f3 = figure('Name','orig_sct_ppg_g0_projection_TF');
tfProj0_orig = squeeze(sum(abs(tfrsq0_orig), 1));
imagesc(t_vec, tfrsqtic0_orig*Hz, tfProj0_orig);
axis xy; colormap(1-gray);
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('Original SCT (g_0) Projection, PPG');
drawnow;
saveas(f3, [matlab.lang.makeValidName(get(f3,'Name')) '.png']);

% 2. Original CT (g2)
fprintf('\n— Running Original CT (g2) on PPG —\n');
[tfc2_orig, tfrtic2_orig, tcrtic2_orig, tfrsq2_orig, tfrsqtic2_orig] = ...
    sqSTCT(signal, 0, 0.5, alpha_res, tDS, g2, dg2, ddg2);
fprintf('Done Original CT (g2).\n');

% Figure 4: Original CT (g2) at midFrame
f4 = figure('Name','orig_ct_ppg_g2_midframe');
imagesc(tfrtic2_orig*Hz, tcrtic2_orig*Hz^2, abs(squeeze(tfc2_orig(:,:,midFrame))));
axis xy; colormap(1-gray);
xlabel('Frequency (Hz)'); ylabel('Chirp Rate (Hz^2)');
title(sprintf('Original CT (g_2), PPG, t=%.2f s', t_vec(midFrame)));
drawnow;
saveas(f4, [matlab.lang.makeValidName(get(f4,'Name')) '.png']);

% Figure 5: Original SCT (g2) at midFrame
f5 = figure('Name','orig_sct_ppg_g2_midframe');
imagesc(tfrtic2_orig*Hz, tcrtic2_orig*Hz^2, abs(squeeze(tfrsq2_orig(:,:,midFrame))));
axis xy; colormap(1-gray);
xlabel('Frequency (Hz)'); ylabel('Chirp Rate (Hz^2)');
title(sprintf('Original SCT (g_2), PPG, t=%.2f s', t_vec(midFrame)));
drawnow;
saveas(f5, [matlab.lang.makeValidName(get(f5,'Name')) '.png']);

% Figure 6: Original SCT (g2) time-frequency projection
f6 = figure('Name','orig_sct_ppg_g2_projection_TF');
tfProj2_orig = squeeze(sum(abs(tfrsq2_orig), 1));
imagesc(t_vec, tfrsqtic2_orig*Hz, tfProj2_orig);
axis xy; colormap(1-gray);
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('Original SCT (g_2) Projection, PPG');
drawnow;
saveas(f6, [matlab.lang.makeValidName(get(f6,'Name')) '.png']);

% Figure 7: Original fixed-time 2×2 comparison (PPG)
slice_t = midFrame;
f7 = figure('Name','orig_fixed_slice_ppg_g0_vs_g2');
subplot(2,2,1);
imagesc(tfrtic0_orig*Hz, tcrtic0_orig*Hz^2, abs(squeeze(tfc0_orig(:,:,slice_t))));
axis xy; colormap(1-gray);
xlabel('Frequency (Hz)'); ylabel('Chirp Rate (Hz^2)');
title('Original CT (g_0)');
subplot(2,2,2);
imagesc(tfrtic0_orig*Hz, tcrtic0_orig*Hz^2, abs(squeeze(tfrsq0_orig(:,:,slice_t))));
axis xy; colormap(1-gray);
xlabel('Frequency (Hz)'); ylabel('Chirp Rate (Hz^2)');
title('Original SCT (g_0)');
subplot(2,2,3);
imagesc(tfrtic2_orig*Hz, tcrtic2_orig*Hz^2, abs(squeeze(tfc2_orig(:,:,slice_t))));
axis xy; colormap(1-gray);
xlabel('Frequency (Hz)'); ylabel('Chirp Rate (Hz^2)');
title('Original CT (g_2)');
subplot(2,2,4);
imagesc(tfrtic2_orig*Hz, tcrtic2_orig*Hz^2, abs(squeeze(tfrsq2_orig(:,:,slice_t))));
axis xy; colormap(1-gray);
xlabel('Frequency (Hz)'); ylabel('Chirp Rate (Hz^2)');
title('Original SCT (g_2)');
drawnow;
saveas(f7, [matlab.lang.makeValidName(get(f7,'Name')) '.png']);

fprintf('— Original PPG CT/SCT figures (7) saved.\n\n');

%% ————————— Five, v3 CT/SCT and Save Figures —————————
% 1. v3 CT (g0)
fprintf('— Running v3 CT (g0) on PPG —\n');
[tfc0_v3, tfrtic0_v3, tcrtic0_v3, tfrsq0_v3, tfrsqtic0_v3] = ...
    sqSTCT_v3(signal, 0, 0.5, alpha_res, tDS, g0, dg0, ddg0);
fprintf('Done v3 CT (g0).\n');

% Figure 8: v3 CT (g0) at midFrame
f8 = figure('Name','v3_ct_ppg_g0_midframe');
imagesc(tfrtic0_v3*Hz, tcrtic0_v3*Hz^2, abs(squeeze(tfc0_v3(:,:,midFrame))));
axis xy; colormap(1-gray);
xlabel('Frequency (Hz)'); ylabel('Chirp Rate (Hz^2)');
title(sprintf('v3 CT (g_0), PPG, t=%.2f s', t_vec(midFrame)));
drawnow;
saveas(f8, [matlab.lang.makeValidName(get(f8,'Name')) '.png']);

% Figure 9: v3 SCT (g0) at midFrame
f9 = figure('Name','v3_sct_ppg_g0_midframe');
imagesc(tfrtic0_v3*Hz, tcrtic0_v3*Hz^2, abs(squeeze(tfrsq0_v3(:,:,midFrame))));
axis xy; colormap(1-gray);
xlabel('Frequency (Hz)'); ylabel('Chirp Rate (Hz^2)');
title(sprintf('v3 SCT (g_0), PPG, t=%.2f s', t_vec(midFrame)));
drawnow;
saveas(f9, [matlab.lang.makeValidName(get(f9,'Name')) '.png']);

% Figure 10: v3 SCT (g0) time-frequency projection
f10 = figure('Name','v3_sct_ppg_g0_projection_TF');
tfProj0_v3 = squeeze(sum(abs(tfrsq0_v3), 1));
imagesc(t_vec, tfrsqtic0_v3*Hz, tfProj0_v3);
axis xy; colormap(1-gray);
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('v3 SCT (g_0) Projection, PPG');
drawnow;
saveas(f10, [matlab.lang.makeValidName(get(f10,'Name')) '.png']);

% 2. v3 CT (g2)
fprintf('\n— Running v3 CT (g2) on PPG —\n');
[tfc2_v3, tfrtic2_v3, tcrtic2_v3, tfrsq2_v3, tfrsqtic2_v3] = ...
    sqSTCT_v3(signal, 0, 0.5, alpha_res, tDS, g2, dg2, ddg2);
fprintf('Done v3 CT (g2).\n');

% Figure 11: v3 CT (g2) at midFrame
f11 = figure('Name','v3_ct_ppg_g2_midframe');
imagesc(tfrtic2_v3*Hz, tcrtic2_v3*Hz^2, abs(squeeze(tfc2_v3(:,:,midFrame))));
axis xy; colormap(1-gray);
xlabel('Frequency (Hz)'); ylabel('Chirp Rate (Hz^2)');
title(sprintf('v3 CT (g_2), PPG, t=%.2f s', t_vec(midFrame)));
drawnow;
saveas(f11, [matlab.lang.makeValidName(get(f11,'Name')) '.png']);

% Figure 12: v3 SCT (g2) at midFrame
f12 = figure('Name','v3_sct_ppg_g2_midframe');
imagesc(tfrtic2_v3*Hz, tcrtic2_v3*Hz^2, abs(squeeze(tfrsq2_v3(:,:,midFrame))));
axis xy; colormap(1-gray);
xlabel('Frequency (Hz)'); ylabel('Chirp Rate (Hz^2)');
title(sprintf('v3 SCT (g_2), PPG, t=%.2f s', t_vec(midFrame)));
drawnow;
saveas(f12, [matlab.lang.makeValidName(get(f12,'Name')) '.png']);

% Figure 13: v3 SCT (g2) time-frequency projection
f13 = figure('Name','v3_sct_ppg_g2_projection_TF');
tfProj2_v3 = squeeze(sum(abs(tfrsq2_v3), 1));
imagesc(t_vec, tfrsqtic2_v3*Hz, tfProj2_v3);
axis xy; colormap(1-gray);
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('v3 SCT (g_2) Projection, PPG');
drawnow;
saveas(f13, [matlab.lang.makeValidName(get(f13,'Name')) '.png']);

% Figure 14: v3 fixed-time 2×2 comparison (PPG)
f14 = figure('Name','v3_fixed_slice_ppg_g0_vs_g2');
subplot(2,2,1);
imagesc(tfrtic0_v3*Hz, tcrtic0_v3*Hz^2, abs(squeeze(tfc0_v3(:,:,slice_t))));
axis xy; colormap(1-gray);
xlabel('Frequency (Hz)'); ylabel('Chirp Rate (Hz^2)');
title('v3 CT (g_0)');
subplot(2,2,2);
imagesc(tfrtic0_v3*Hz, tcrtic0_v3*Hz^2, abs(squeeze(tfrsq0_v3(:,:,slice_t))));
axis xy; colormap(1-gray);
xlabel('Frequency (Hz)'); ylabel('Chirp Rate (Hz^2)');
title('v3 SCT (g_0)');
subplot(2,2,3);
imagesc(tfrtic2_v3*Hz, tcrtic2_v3*Hz^2, abs(squeeze(tfc2_v3(:,:,slice_t))));
axis xy; colormap(1-gray);
xlabel('Frequency (Hz)'); ylabel('Chirp Rate (Hz^2)');
title('v3 CT (g_2)');
subplot(2,2,4);
imagesc(tfrtic2_v3*Hz, tcrtic2_v3*Hz^2, abs(squeeze(tfrsq2_v3(:,:,slice_t))));
axis xy; colormap(1-gray);
xlabel('Frequency (Hz)'); ylabel('Chirp Rate (Hz^2)');
title('v3 SCT (g_2)');
drawnow;
saveas(f14, [matlab.lang.makeValidName(get(f14,'Name')) '.png']);

fprintf('— v3 PPG CT/SCT figures (7) saved.\n\n');
fprintf('All 14 PPG figures have been automatically saved as PNG files in the current directory.\n');
