function [tfc, tfrtic, tcrtic, tfrsq, tfrsqtic] = sqSTCT_v5(x, lowFreq, highFreq, alpha, tDS, h, Dh, DDh)
% SQSTCT_V5 - v3+v4 融合优化
% - parfor 并行时间帧
% - 可选 gpuArray 加速
% - 预计算所有 chirp 对应的窗函数 W 和 exp(-pi*i*chirp*ht.^2)

useGPU = false;  % 切换是否启用 GPU

% —— 初始化 ——
[xrow,xcol] = size(x);
assert(xcol==1, 'X must be a column vector');

t = 1:length(x);
tIdx = 1:tDS:length(x);
tLen = numel(tIdx);

% 【修正】频轴生成方式与原始保持一致，确保 N 为偶数
freqVec = -0.5+alpha : alpha : 0.5;
N = numel(freqVec);             % 这么来，N 必为偶数
crate = ([1:N-1] - ceil(N/2)) / N^2;

Lidx = ceil((N/2)*(lowFreq/0.5)) + 1;
Hidx = floor((N/2)*(highFreq/0.5));
fLen = Hidx - Lidx + 1;
cLen = numel(crate);

assert(highFreq<=0.5, 'TopFreq must be ≤ 0.5');
assert(tDS>=1 && rem(tDS,1)==0, 'tDS must be integer ≥ 1');

% 窗函数尺寸
hLen = numel(h);
Lh   = (hLen-1)/2;
assert(rem(Lh,1)==0, 'Window length must be odd');
ht   = (-Lh:Lh)';

% 预分配输出
tfc   = zeros(N-1, N/2, tLen, 'like', x);
tfrsq = zeros(cLen, fLen, tLen, 'like', x);

tfrtic   = linspace(0,0.5,N/2).';
tcrtic   = crate(:);
tfrsqtic = linspace(lowFreq,highFreq,fLen).';

Ex = mean(abs(x).^2);
Threshold = 1e-6 * Ex;

% —— 预计算 W(table) 和 expTable —— 
% W: cLen × hLen × 6, expC: cLen × hLen
W    = zeros(cLen, hLen, 6, 'like', x);
expC = zeros(cLen, hLen, 'like', x);
for cidx = 1:cLen
    cr = crate(cidx);
    expC(cidx,:)   = exp(-pi*1i*cr * (ht.^2)).';   % 1×hLen
    W(cidx,:,1)   = conj(h).';                     % g
    W(cidx,:,2)   = conj(Dh).';                    % g'
    W(cidx,:,3)   = conj(DDh).';                   % g''
    W(cidx,:,4)   = (conj(h).*ht).';               % x·g
    W(cidx,:,5)   = (conj(Dh).*ht).';              % x·g'
    W(cidx,:,6)   = (conj(h).*(ht.^2)).';           % x^2·g
end

if useGPU
    x      = gpuArray(x);
    crate  = gpuArray(crate);
    W      = gpuArray(W);
    expC   = gpuArray(expC);
    tfc    = gpuArray(tfc);
    tfrsq  = gpuArray(tfrsq);
end

fprintf('v5: total chirps = %d\n', N-1);

% —— 并行处理各时间帧 —— 
parfor tidx = 1:tLen
    tfc_loc   = zeros(N-1, N/2, 'like', x);
    tfrsq_loc = zeros(cLen, fLen, 'like', x);
    
    ti = tIdx(tidx);
    for cidx = 1:cLen
        cr = crate(cidx);
        tau = -min([round(N/2)-1, Lh, ti-1]) : min([round(N/2)-1, Lh, xrow-ti]);
        idx = mod(N+tau, N) + 1;          % 1×numel(tau)
        
        eC = expC(cidx, Lh+1+tau);        % 1×|tau|
        wM = squeeze(W(cidx, Lh+1+tau, :));% |tau|×6
        
        xf = x(ti+tau) .* eC.';            % |tau|×1
        six = zeros(N,6,'like',x);
        six(idx,:) = xf * ones(1,6) .* wM; % N×6
        
        F6  = fft(six);
        tf0 = F6(1:N/2,1);
        tf1 = F6(1:N/2,2);
        tf2 = F6(1:N/2,3);
        tfx0= F6(1:N/2,4);
        tfx1= F6(1:N/2,5);
        tfx2= F6(1:N/2,6);
        
        % 重定位
        num = tf0.*tf2 - 4*pi*1i*cr*tf0.*tfx1 - 2*pi*1i*cr*(tf0.^2) + ...
              (2*pi*1i*cr)^2*(tf0.*tfx2) - (tf1.^2) - (2*pi*1i*cr)^2*(tfx0.^2) + ...
               4*pi*1i*cr*(tf1.*tfx0);
        den = -tf0.*tfx1 + 2*pi*1i*cr*(tf0.*tfx2) + tfx0.*tf1 - 2*pi*1i*cr*(tfx0.^2);
        lambda0 = num ./ den / (2*pi);
        
        lambda = round(N^2 * imag(lambda0));
        omega  = round(N   * imag(tf1./tf0/(2*pi) - (cr*1i - lambda0).*tfx0./tf0));
        
        abs0 = abs(tf0);
        jcol = (1:N/2).';
        jhat = jcol - omega;
        lidx = lambda + ceil(N/2);
        
        valid = abs0 > Threshold & jhat>=Lidx & jhat<=Hidx & lidx>=1 & lidx<=cLen;
        rows  = lidx(valid);
        cols  = jhat(valid) - Lidx + 1;
        vals  = tf0(valid);
        lin   = sub2ind([cLen, fLen], rows, cols);
        
        sst = zeros(cLen, fLen, 'like', x);
        sst(:) = accumarray(lin, vals, [cLen*fLen,1]);
        
        tfc_loc(cidx,:)   = tf0;
        tfrsq_loc         = tfrsq_loc + sst;
    end
    
    tfc(:,:,tidx)   = tfc_loc;
    tfrsq(:,:,tidx) = tfrsq_loc;
end

if useGPU
    tfc   = gather(tfc);
    tfrsq = gather(tfrsq);
end

fprintf('v5 Done.\n');
end
