function [tfc, tfrtic, tcrtic, tfrsq, tfrsqtic] = sqSTCT_v3(x, lowFreq, highFreq, alpha, tDS, h, Dh, DDh)
% 使用 parfor 并行化时间帧
%
% 输入参数:
%   x      : 输入信号（列向量）
%   lowFreq: 最低频率（0）
%   highFreq: 最高频率（不超过 0.5）
%   alpha  : 频率和 chirp 率的分辨率
%   tDS    : 时间采样间隔（下采样因子）
%   h      : 窗函数（列向量）
%   Dh     : h 的一阶导数
%   DDh    : h 的二阶导数
%
% 输出参数:
%   tfc    : 原始 chirplet 变换（用于调试或后续重构）
%   tfrtic : 频率刻度
%   tcrtic : chirp率刻度
%   tfrsq  : 同步压缩后的 chirplet 变换（SCT）
%   tfrsqtic: SCT 输出的频率刻度

[xrow,xcol] = size(x);
t = 1:length(x);
tLen = length(t(1:tDS:end));

N = length(-0.5+alpha:alpha:0.5);
crate = ([1:N-1]-ceil(N/2))/N^2;

Lidx = ceil((N/2)*(lowFreq/0.5)) + 1;
Hidx = floor((N/2)*(highFreq/0.5));
fLen = Hidx - Lidx + 1;
cLen = length(crate);

if (xcol ~= 1)
    error('X must have only one column');
elseif highFreq > 0.5
    error('TopFreq must be a value in [0, 0.5]');
elseif (tDS < 1) || (rem(tDS,1))
    error('tDS must be an integer value >= 1');
end

[hrow,hcol] = size(h);
Lh = (hrow-1)/2;
if (hcol ~= 1) || (rem(hrow,2) == 0)
    error('H must be a smoothing window with odd length');
end
ht = -Lh:Lh;

tfc = zeros(N-1, N/2, tLen);
tfrsq = zeros(cLen, fLen, tLen);

tfrtic = linspace(0, 0.5, N/2)';
tcrtic = crate;
tfrsqtic = linspace(lowFreq, highFreq, fLen)';

Ex = mean(abs(x).^2);
Threshold = 1.0e-6 * Ex;

fprintf('Chirp-rate total: %d ...\n', N-1);

parfor tidx = 1:tLen
    tfc_local = zeros(N-1, N/2);
    tfrsq_local = zeros(cLen, fLen);
    
    ti = t((tidx-1)*tDS+1);
    for cidx = 1:N-1
        chirp = crate(cidx);
        tau = -min([round(N/2)-1, Lh, ti-1]):min([round(N/2)-1, Lh, xrow-ti]);
        indices = mod(N+tau, N) + 1;
        htau = ht(Lh+1+tau)';
        expchirp = exp(-pi*1i*chirp * htau.^2);
        
        xseg = x(ti+tau);
        W = [ ...
            conj(h(Lh+1+tau)), ...
            conj(Dh(Lh+1+tau)), ...
            conj(DDh(Lh+1+tau)), ...
            conj(h(Lh+1+tau)) .* htau, ...
            conj(Dh(Lh+1+tau)) .* htau, ...
            conj(h(Lh+1+tau)) .* htau.^2 ];
        
        XF = xseg .* expchirp;
        six_tf = zeros(N, 6);
        six_tf(indices, :) = XF .* W;
        
        six_fft = fft(six_tf);
        tf0 = six_fft(1:N/2, 1);
        tf1 = six_fft(1:N/2, 2);
        tf2 = six_fft(1:N/2, 3);
        tfx0 = six_fft(1:N/2, 4);
        tfx1 = six_fft(1:N/2, 5);
        tfx2 = six_fft(1:N/2, 6);
        
        lambda0 = (tf0.*tf2 - 4*pi*1i*chirp*tf0.*tfx1 - 2*pi*1i*chirp*tf0.*tf0 + ...
                   (2*pi*1i*chirp)^2*tf0.*tfx2 - tf1.*tf1 -(2*pi*1i*chirp)^2*...
                   tfx0.*tfx0 + 4*pi*1i*chirp*tf1.*tfx0) ./ ...
                   (-tf0.*tfx1 + 2*pi*1i*chirp*tf0.*tfx2 + ...
                   tfx0.*tf1 - 2*pi*1i*chirp*tfx0.*tfx0) ./ (2.0*pi);
               
        lambda = round(N^2 * imag(lambda0));
        omega = round(N * imag(tf1./tf0./(2.0*pi) - (chirp*1i-lambda0).*tfx0./tf0));
        
        abs_tf0 = abs(tf0);
        jcol = (1:N/2)';
        jcolhat = jcol - omega;
        lambda_idx = lambda + ceil(N/2);
        
        valid = abs_tf0 > Threshold & ...
                jcolhat >= Lidx & jcolhat <= Hidx & ...
                lambda_idx >= 1 & lambda_idx <= cLen;
        
        row_idx = lambda_idx(valid);
        col_idx = jcolhat(valid) - Lidx + 1;
        values = tf0(valid);
        lin_idx = sub2ind([cLen, fLen], row_idx, col_idx);
        
        sst = zeros(cLen, fLen);
        sst(:) = accumarray(lin_idx, values, [cLen*fLen, 1]);
        
        tfc_local(cidx,:) = tf0;
        tfrsq_local = tfrsq_local + sst;
    end
    tfc(:,:,tidx) = tfc_local;
    tfrsq(:,:,tidx) = tfrsq_local;
end
fprintf('Done.\\n');
end
