clear
% network size
C = 6; % known neuron num
I = 2; % unknown neuron num
Q = 50; % intrinsic history length (autoregression)
R = 5; % extrinsic history length
M = 5; % unknown history length
K = 10000; % spiking activity total time length
tau = 0.05; % time interval

% assign parameters
% alpha = [2,1,3,2,2,1]*2; % baseline
alpha = [3.5,2.4,2,3,2.8,2.5]; % baseline: spontaneous spiking rate
epsi = zeros(C,Q); % intrinsic paramter
beta = zeros(C,C,R); % extrinsic paramter
gammaUU = zeros(C,I,M); % unknown paramter


% intrinsic para
x = 1:Q;
z_epsi = [2,1.9,2.5,1.5,2,1.8];
for c=1:C
    epsi(c,:) = -sin(x/z_epsi(c))./(x/z_epsi(c));
end
% % figure;
% % for c = 1:C
% %     subplot(6,1,c);
% %     stem(epsi(c,:));
% % end

% extrinsic para
pos_beta = [0,0,0,0,0,0;
    1,0,1,0,0,0;
    -1,0,0,0,0,0;
    1,0,0,0,1,-1;
    1,0,0,-1,0,1;
    -1,0,0,1,-1,0];
z_beta = [0,0,0,0,0,0;
    1,0,0.9,0,0,0;
    0.8,0,0,0,0,0;
    0.9,0,0,0,0.7,0.6;
    0.8,0,0,0.5,0,0.7;
    0.9,0,0,0.6,0.5,0];
for c = 1:C
    for c1 = 1:C
        for r = 1:R
            beta(c,c1,r) = pos_beta(c,c1)*exp((-z_beta(c,c1))*(r));
        end
    end
end

% unknown para
pos_gamma = [1,1,1,1;
             0,1,0,0;
             1,0,1,0;
             1,1,1,0;
             0,1,0,1;
             1,0,1,1];
z_gamma = [0.3,0.5,0.6,0.2;
           0,1,0,0;
           0.8,0,0.4,0;
           0,0,0.9,0;
           0,0.8,0,0.6;
           0.2,0,0.8,0.4];
for c = 1:C
for i = 1:I
    for m = 1:M
        gammaUU(c,i,m) = pos_gamma(c,i)*exp((-z_gamma(c,i))*(m));
    end
end
end
% % figure;
% % for c=1:C
% %     subplot(C,1,c);
% %     stem(reshape((squeeze(gammaUU(c,:,:))).',[I*M,1]));
% % end

% generate the spikes

loggamma_a = 1;
loggamma_b = 50;
shiftLogGamma = -3.9;

% x=-5:0.05:5;
% shift mean by 2 to have distribution centered at 0
flg = @(x) (exp(loggamma_b*(x-shiftLogGamma)).*exp(-exp(x-shiftLogGamma)/loggamma_a))/(loggamma_a^loggamma_b*gamma(loggamma_b));
% int = integral(flg, -10, 10);

nsamples = K*I;
delta = .5;
proppdf = @(x,y) unifpdf(y-x,-delta,delta);
proprnd = @(x) x + rand*2*delta - delta;  
dU = mhsample(1, nsamples, 'pdf', flg, 'proprnd',proprnd, 'proppdf',proppdf);
dU = reshape(dU, I, K);

% U=gamrnd(dU_alpha,1/dU_beta,[I,K+1]);
% dU = (diff(U'))';
% log-gamma distribution
%%% ????


dN = zeros(C,K); % dN is a binary sequence
dN(:,1) = [0,1,0,0,1,0];

lambda = zeros(C,K); % conditional intensity function

for k = 2:K
    for c = 1:C
        
        in = dot(epsi(c,1:min(k-1,Q)),fliplr(dN(c,k-min(k-1,Q):k-1)));
        if(k==2)
            ex = sum(dot(squeeze(beta(:,c,1:min(k-1,R))),fliplr(dN(:,k-min(k-1,R):k-1))));%%%%
            un = sum(dot(squeeze(gammaUU(c,:,1:min(k-1,M))),fliplr(dU(:,k-min(k-1,M):k-1))));%%%%
        else
            ex = sum(dot(squeeze(beta(:,c,1:min(k-1,R))),fliplr(dN(:,k-min(k-1,R):k-1)),2));%%%%
            un = sum(dot(squeeze(gammaUU(c,:,1:min(k-1,M))),fliplr(dU(:,k-min(k-1,M):k-1))),2);%%%%
        end
        
        lambda(c,k) = exp(alpha(c) + in + ex + un);
        
    end
    
    for i=1:C
        u = rand(1);
        if(u<=lambda(i,k)*tau)
            dN(i,k) = 1;
        else
            dN(i,k) = 0;
        end
    end
    
end

% % % plot
% figure;
% for i=1:C
%     subplot(C,1,i);
%     plot(1:K,lambda(i,1:K));
% end
% figure;
% for i=1:C
%     subplot(C,1,i);
%     spikeF = stem(1:K,dN(i,1:K));
%     set(spikeF, 'Marker', 'none');
% end
% figure;
% for i=1:C
%     subplot(C,1,i);
%     hist(dN(i,1:K));
% end
saveStr = sprintf('neuronSpikeSim_wUU_logGamma_K_%d.mat', K);
save(saveStr, 'dN', 'Q', 'R', 'M', 'I', 'tau', 'gammaUU');