% % clear
% % 
%% Simulation
% matObj = matfile('neuronSpikeSim_wUU_logGamma.mat');
matObj = matfile('neuronSpikeSim_wUU_logGamma_K_10000.mat');
dN = matObj.dN;
Q = matObj.Q;
R = matObj.R;
M = matObj.M;
I = matObj.I;
tau = matObj.tau;
gammaUU = matObj.gammaUU;
loggamma = [1,2];

[alpha, beta, epsi, llh] = neuronEst('dN', dN, 'Q', Q, 'R', R, 'M', M, 'I', I, ...
    'tau', tau, 'gammaUU', gammaUU, 'loggamma', loggamma);

figure;
plot(llh);grid;
% % 
% % 
% % 
% Real-World Somatosensory-3
% % 
% % matObj = matfile('neuronPreProcessed_SSC3_d1.mat');
% % dN = matObj.dN;
% % Q = 50;
% % R = 5;
% % M = 5;
% % I = 2;
% % tau = 0.05; % sampling BW of the real data
% % loggamma = [1,2];
% % 
% % matObj1 = matfile('neuronSpikeSim_wUU_logGamma.mat');
% % gammaUU = matObj1.gammaUU;
% % gammaUU = [gammaUU;gammaUU(1:4,:,:)]; % total neurons = 10
% % dNUse = [];
% % ctr = 0;
% % bw = 30;  % re-scaling of the sampling window to take care of non-existence of data
% % while(ctr+bw<size(dN,2))
% %     dNUse = [dNUse, sum(dN(:,ctr+1:ctr+bw),2)];
% %     ctr = ctr + bw;
% % end
% % dN = dNUse;
% % tau = tau*30;
% % 
% % [alpha, beta, epsi, llh] = neuronEst('dN', dN, 'Q', Q, 'R', R, 'M', M, 'I', I, ...
% %     'tau', tau, 'gammaUU', gammaUU, 'loggamma', loggamma);
% % 
% % figure;
% % plot(llh);grid;

%% Real-World Retina:1

% % matObj = matfile('neuronPreProcessed_RET_1.mat');
% % dN = matObj.dN;
% % Q = 50;
% % R = 5;
% % M = 5;
% % I = 2;
% % tau = 0.001; % sampling BW of the real data
% % loggamma = [1,2];
% % 
% % matObj1 = matfile('neuronSpikeSim_wUU_logGamma.mat');
% % gammaUU = matObj1.gammaUU;
% % gammaUU = [gammaUU;gammaUU(1:3,:,:)]; % total neurons = 7
% % dNUse = [];
% % ctr = 0;
% % bw = 30;  % re-scaling of the sampling window to take care of non-existence of data
% % while(ctr+bw<size(dN,2))
% %     dNUse = [dNUse, sum(dN(:,ctr+1:ctr+bw),2)];
% %     ctr = ctr + bw;
% % end
% % dN = dNUse;
% % tau = tau*30;
% % 
% % [alpha, beta, epsi, llh] = neuronEst('dN', dN, 'Q', Q, 'R', R, 'M', M, 'I', I, ...
% %     'tau', tau, 'gammaUU', gammaUU, 'loggamma', loggamma);
% % 
% % figure;
% % plot(llh);grid;

%% Real-World Cat Retina

% % matObj = matfile('neuronPreProcessed_Cat_Retina.mat');
% % dN = matObj.dN;
% % Q = 50;
% % R = 5;
% % M = 5;
% % I = 2;
% % tau = 0.001; % sampling BW of the real data
% % loggamma = [1,2];
% % 
% % matObj1 = matfile('neuronSpikeSim_wUU_logGamma.mat');
% % gammaUU = matObj1.gammaUU;
% % gammaUU = gammaUU(1:5,:,:); % total neurons = 5
% % dNUse = [];
% % ctr = 0;
% % bw = 20;  % re-scaling of the sampling window to take care of non-existence of data
% % while(ctr+bw<size(dN,2))
% %     dNUse = [dNUse, sum(dN(:,ctr+1:ctr+bw),2)];
% %     ctr = ctr + bw;
% % end
% % dN = dNUse;
% % dN = dN(:,1:5000);
% % tau = tau*20;
% % 
% % % dN = dN(:,1:15000);
% % [alpha, beta, epsi, llh] = neuronEst('dN', dN, 'Q', Q, 'R', R, 'M', M, 'I', I, ...
% %     'tau', tau, 'gammaUU', gammaUU, 'loggamma', loggamma);
% % 
% % figure;
% % plot(llh);grid;