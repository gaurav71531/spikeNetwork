function [alpha, beta, epsi, llh] = neuronEst(varargin)

% I = 2; % unknown neuron num
% Q = 50; % intrinsic history length (autoregression)
% R = 5; % extrinsic history length
% M = 5; % unknown history length
% K = 500; % spiking activity total time length
% tau = 0.05; % time interval

if nargin>1
    for i = 1:2:nargin
        switch varargin{i}
            case 'dN'
                dN = varargin{i+1};
            case 'Q'
                Q = varargin{i+1};
            case 'R'
                R = varargin{i+1};
            case 'M'
                M = varargin{i+1};
            case 'I'
                I = varargin{i+1};
            case 'tau'
                tau = varargin{i+1};
            case 'gammaUU'
                gammaUU = varargin{i+1};
            case 'loggamma'
                loggamma_a = varargin{i+1}(1);
                loggamma_b = varargin{i+1}(2);
        end
    end
else
    matObj = matfile('neuronSpikeSim_wUU_logGamma.mat');
%     dN = matObj.dN;
    Q = matObj.Q;
    R = matObj.R;
    M = matObj.M;
    I = matObj.I;
    tau = matObj.tau;
    gammaUU = matObj.gammaUU;
    loggamma_a = 1;
    loggamma_b = 2;
end

[C,K] =  size(dN);

dimPara = 1+Q+C*R; % len of YVec

param = struct('C', C, 'K', K, 'Q', Q, 'R', R,...
    'M', M, 'I', I, 'tau', tau, 'dimPara', dimPara);

%parameters init

MLEdU0 = 2*rand(I,K)-1; %init UU
% matObj1 = matfile('dUInitValTemp.mat');
% MLEdU0 = matObj1.MLEdU0;
MLEpara0 = ones(C,dimPara)*exp(1); %exp of actual parameters

for c=1:C
%     fixing the initial point scaling tau  = 0.05
    MLEpara0(c,1) = exp(mean(dN(c,:))/0.05);
    MLEpara0(c,1+Q+(c-1)*R+1:1+Q+c*R)=1; % 0 for extrinsic beta(c,c,:)
end

iterMaxEM = 50;
iterMaxE = 300;
iterMaxM = 100;
llh = zeros(iterMaxEM,1);

% initialize EM
dUHat = MLEdU0;
M_para = MLEpara0;

%initialize Y vec
Y = getYVec(dN, dimPara, Q, R);
lambdaUU = getLambdaU(dUHat, gammaUU, C, M);

% initialize EM params
[M_para,lambda,llh(1)] = fixPointIter_M(dN,Y,M_para,lambdaUU,param,iterMaxM);
%

for iterInd = 2:iterMaxEM
    % E-step
    [dUHat,lambdaUU] = fixPointIter_E(dN,dUHat,lambda,gammaUU,lambdaUU,param, loggamma_a, loggamma_b, iterMaxE);
    % M-step
    [M_para,lambda,llh(iterInd)] = fixPointIter_M(dN,Y,M_para,lambdaUU,param,iterMaxM); 
    
    if ~mod(iterInd,10)
        fprintf('Iterations complete = %d, likelihood = %f\n', iterInd, llh(iterInd));
    end
end
M_para = log(M_para);
alpha = M_para(:,1);
beta = M_para(:,2:Q+1);
epsi = M_para(:,Q+1+1:end);


function [dUHat,lambdaUU] = fixPointIter_E(dN, dUHat, lambda, gammaUU, lambdaUU, param, loggamma_a, loggamma_b, niter)

% mu: mu is a I by K vector with mu(i,k) = exp(dU(i,k))

muPrev = exp(dUHat);
D = param.K;
muCurrent = muPrev;

% t=zeros(param.I,D-1); % power
G_num=zeros(param.I,D-1);
G_den = zeros(param.I,D-1);
threshold = 0.001;%%%% maybe choose a stopping rule base on the MLE?

tConst = 0.1;


t_den =zeros(param.I, D-1);
for i = 1:param.I
    convol = zeros(1,D-1);
    for c = 1:param.C
        c1 = conv(flipud([0;reshape(gammaUU(c,i,:),param.M,1)]), dN(c,:));
        convol = convol + c1(param.M+1:end-1);
    end
    G_num(i,:) = loggamma_b + convol';
end

    tic
t = getTExp(param.I, param.C, D, param.M, param.K, gammaUU, dN, tConst, G_num);
% for i = 1:param.I
%     for q = 1:D-1
%         for c = 1:param.C
%             for p = max(q-param.M+1,1):min(q+param.M-1,param.K)
%                 for k = max(p+1,q+1):min([p+param.M,q+param.M,param.K])
%                     t_den(i,q) = t_den(i,q) + gammaUU(c,i,k-q)*gammaUU(c,i,k-p)*dN(c,k);
%                 end
%             end
%         end
%         t(i,:) = tConst * G_num(i,:)./t_den(i,:);
%     end
% end
  toc

F = zeros(niter,1);
for iter = 1:niter
    
    cvMatTemp = lambda.*lambdaUU;
    for i = 1:param.I 
        convol = zeros(1,D-1);
        for c = 1:param.C
            c1 = conv(flipud([0;reshape(gammaUU(c,i,:),param.M,1)]), cvMatTemp(c,:));
            convol = convol + c1(param.M+1:end-1);
        end
        G_den(i,:) = param.tau*convol + muPrev(i,1:end-1)/loggamma_a;
        muCurrent(i,2:end-1) = muPrev(i,2:end-1) .* (G_num(i,2:end)./G_den(i,2:end)).^t(i,2:end);
        muCurrent(i,t_den(i,:) < 1) = 1;
%         ignoring values with t_den < 1:
% Reason: For a given data, if for some unknown index-i and time-point k
% has t_den very small ia an indication of no need for unknown term at this
% (q,k) pair. Therefore, setting it to 0 (or exponential to 1)
    end
    
    dUHat = log(muCurrent);
    lambdaUU = getLambdaU(dUHat, gammaUU, param.C, param.M);
    F(iter) = sum(sum(dN(:,2:end-1).*log(lambdaUU(:,2:end-1).*lambda(:,2:end-1)) ...
        -param.tau*lambdaUU(:,2:end-1).*lambda(:,2:end-1))) ...
        +sum(sum(loggamma_b*log(muCurrent(:,2:end-1)) - muCurrent(:,2:end-1)/loggamma_a));
    muPrev = muCurrent;
    
end


function t = getTExp(I, C, D, M, K, gammaUU, dN, tConst, G_num)

t_den =zeros(I, D-1);
t=zeros(I,D-1);
for i = 1:I
    for q = 1:D-1
        for c = 1:C
            for p = max(q-M+1,1):min(q+M-1,K)
                for k = max(p+1,q+1):min([p+M,q+M,K])
                    t_den(i,q) = t_den(i,q) + gammaUU(c,i,k-q)*gammaUU(c,i,k-p)*dN(c,k);
                end
            end
        end
        t(i,:) = tConst * G_num(i,:)./t_den(i,:);
    end
end

t_den1 = zeros(I, D-1);
t1=zeros(I,D-1);
for i = 1:I
%     convol = zeros(1,D-1);
    for c = 1:C
        gammaVec = reshape(gammaUU(c,i,:),M,1);
        gammaMat = gammaVec*[gammaVec',fliplr(gammaVec(1:end-1)')];
        gammaMatUse = flipud([zeros(1,2*M-1);[triu(gammaMat(:,1:M)), [zeros(1,M-1);tril(gammaMat(2:end,M+1:end))]]]);
        c1 = conv2(dN(c,:)', gammaMatUse);
        t_den1(i,:)  = t_den1(i,:) + sum(c1(M+1:end-1,:), 2)';
    end
    t1(i,:) = tConst * G_num(i,:) ./t_den1(i,:); 
end
disp('gg');




function [gammaOut,lambda,llh] = fixPointIter_M(dN, Y, gammaPrev, lambdaU, param, niter)

gammaOut = zeros(size(gammaPrev));
threshold = 0.001;

likelihood = zeros(param.C, niter);
lambda = zeros(param.C,param.K);
YUseForC = zeros(param.K, param.dimPara);
for c  = 1:param.C
    YUseForC(:,:) = reshape(Y(c,:,:),param.K, param.dimPara);
    
    sumY = sum(YUseForC,2);
    G_num = dN(c,:)*YUseForC;
    betaDen = sum(YUseForC.* repmat(sumY.*dN(c,:)', 1, param.dimPara),1);
    beta = G_num./betaDen;
    
    for iter = 1:niter
        lambdaMat = repmat(gammaPrev(c,:), param.K, 1).^YUseForC;
        lambdaUse = prod(lambdaMat,2);

        G_den = (lambdaUse'.*lambdaU(c,:)) * YUseForC * param.tau;

        gammaOut(c,:) = gammaPrev(c,:) .* ((G_num./G_den).^beta);
        % for c' = c, set gamma = 1
        % to exclude from the summation
        gammaOut(c,1+param.Q+(c-1)*param.R+1:1+param.Q+c*param.R) = 1; 

        likelihood(c,iter) = sum(dN(c,2:end-1)'.*log(lambdaU(c,2:end-1)'.*lambdaUse(2:end-1))...
            - param.tau*lambdaU(c,2:end-1)'.*lambdaUse(2:end-1));

%         if abs(logP(iter)-logP(iter-1))<threshold
%             break;
%         end
        gammaPrev(c,:) = gammaOut(c,:);
    end
    lambdaMat = repmat(gammaOut(c,:), param.K, 1).^YUseForC;
    lambdaUse = prod(lambdaMat,2);
    lambda(c,:) = lambdaUse';
end
llh = sum(likelihood(:,iter));


function Y = getYVec(dN, D, Q, R)

[C, K] = size(dN);
Y = zeros(C, K, D);
for c=1:C
    % Y(c,k,:)=[1 ... dN(c,k-q) ... dN(c1,k-r)...dU(i,k-m)] D-length
    Y(c,1,1)=1;
    for k=2:K
        epsi_len = min(k-1,Q);
        beta_len = min(k-1,R);
        Y(c,k,1)=1;
        Y(c,k,2:epsi_len+1)=fliplr(dN(c,k-min(k-1,Q):(k-1)));
        for c1=1:C
            Y(c,k,1+Q+(c1-1)*R+1:1+Q+(c1-1)*R+beta_len)=fliplr(dN(c1,k-min(k-1,R):(k-1)));
        end 
    end
end


function lambdaU = getLambdaU(dU, gamma, C, M)

[I,K] = size(dU);
lambdaU = zeros(C,K);
for c = 1:C
    convol = zeros(1,K);
    for ii = 1:I
        gVec = [0;reshape(gamma(c,ii,:), M, 1)]; % no co-eff for current ind in convolution
        cvl = conv(gVec, dU(ii,:));
        convol = convol + cvl(1:K);
    end
    lambdaU(c,:) = exp(convol);
end

function plotSpike(dN)

[C,K] = size(dN);
figure;
for i=1:C
    subplot(C,1,i);
    spikeF = stem(1:K,dN(i,1:K));
    set(spikeF, 'Marker', 'none');
end