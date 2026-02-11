% MATLAB Script for Simulating Coupled Mackey-Glass Systems and Computing Connectivity

clear all; close all; clc;

% -----------------------------
% PARAMETERS
% -----------------------------
n = 2^12;            % Length of time series (default 2^11)
N = 5;               % Number of channels/nodes
C = 0.1;             % Coupling strength
DeltaV = [17 * ones(N, 1), 30 * ones(N, 1), 100 * ones(N, 1), 300 * ones(N, 1)];  % Delay parameter for each channel
ts = 4;              % Sampling time

maxtau = 200;        % Maximum lag for mutual information
mmax = 10;           % Max embedding dimension (for corr dim if used)
runcordim = 0;       % Set to 1 to run correlation dimension estimation
figno = 1;           % Initial figure number

% Watts Strogatz network parameters
k_neighbors = 4;    % Number of nearest neighbors in the ring
beta = 0.2;         % Rewiring probability

for i = 1:size(DeltaV, 2)
    % Create Network
    % G = WattsStrogatz(N, k_neighbors, beta);    % Network
    % cM = C * full(adjacency(G));                % True Causality Matrix
    cM = [0 0.2 0.2 0.2 0
        0 0 0 0 0
        0 0 0 0 0
        0 0 0 0 0.2
        0 0 0 0.2 0]; % 1->2,3,4 / 4->5 / 5->4
    
    % Simulate coupled Mackey-Glass system
    yM = gmanycoupledMGdde(n, DeltaV(:,i), cM, ts);
    
    % Save series and structure
    structure_name = fullfile('datasets','mackey_glass', sprintf('structure_%d.csv', i));
    dataset_name   = fullfile('datasets','mackey_glass', sprintf('dataset_%d.csv', i));
    writematrix(cM, structure_name);
    writematrix(yM, dataset_name);

    figure;
    imagesc(cM);
    colorbar;
end

% % -----------------------------
% % PLOTTING TIME SERIES
% % -----------------------------
% for iK = 1:K
%     figure(figno); figno = figno + 1; clf;
%     plot(yM(:,iK), '.-');
%     ctext = '';
%     for jK = 1:K
%         if jK ~= iK && cM(jK,iK) > 0
%             ctext = sprintf('%s c(X%d)=%1.2f', ctext, jK, cM(jK,iK));
%         end
%     end
%     title(sprintf('X%d, D(%d)=%d ts=%.1f %s', iK, iK, DeltaV(iK), ts, ctext));
% end
% 
% % -----------------------------
% % SCATTER PLOTS x(t) vs x(t+Delta)
% % -----------------------------
% for iK = 1:K
%     figure(figno); figno = figno + 1; clf;
%     lag = round(DeltaV(iK) / ts);
%     plot(yM(1:n-lag, iK), yM(1+lag:n, iK));
%     xlabel('x(t)'); ylabel('x(t+\Delta)');
%     title(sprintf('X%d Scatter Plot', iK));
% end
% 
% % -----------------------------
% % MUTUAL INFORMATION
% % -----------------------------
% miallM = NaN(maxtau+1, K);
% tauV = (0:maxtau)';
% for iK = 1:K
%     tmpM = mutualeprob(yM(:,iK), maxtau);
%     miallM(:, iK) = tmpM(:,2);
% end
% 
% % Cross Mutual Info Matrix
% mutxyM = mutualxyeprob(yM);
% 
% % Plot Delayed Mutual Information
% figure(figno); figno = figno + 1; clf;
% plot(tauV, miallM);
% xlabel('\tau'); ylabel('I(\tau)');
% legend(arrayfun(@(x) sprintf('X%d',x), 1:K, 'UniformOutput', false));
% title('Delayed Mutual Information');
% 
% % -----------------------------
% % PAIRWISE SCATTER AND CORRELATION
% % -----------------------------
% ccM = corrcoef(yM);
% count = 0;
% for iK = 1:K
%     for jK = iK+1:K
%         count = count + 1;
%         figure(figno); figno = figno + 1; clf;
%         plot(yM(:,iK), yM(:,jK), '.');
%         title(sprintf('X%d vs X%d, r=%1.3f I=%2.4f', iK, jK, ccM(iK,jK), mutxyM(count,3)));
%     end
% end

% -----------------------------
% HELPER FUNCTION (Watts-Strogatz Generator)
% -----------------------------
function G = WattsStrogatz(N, k_neighbors, beta)
    % H = WattsStrogatz(N,K,beta) returns a Watts-Strogatz model graph with N
    % nodes, N*K edges, mean node degree 2*K, and rewiring probability beta.
    %
    % beta = 0 is a ring lattice, and beta = 1 is a random graph.
    
    % Connect each node to its K next and previous neighbors. This constructs
    % indices for a ring lattice.
    s = repelem((1:N)', 1, k_neighbors);
    t = s + repmat(1:k_neighbors, N, 1);
    t = mod(t-1, N) + 1;
    
    % Rewire the target node of each edge with probability beta
    for source=1:N    
        switchEdge = rand(k_neighbors, 1) < beta;
        
        newTargets = rand(N, 1);
        newTargets(source) = 0;
        newTargets(s(t == source)) = 0;
        newTargets(t(source, ~switchEdge)) = 0;
        
        [~, ind] = sort(newTargets, 'descend');
        t(source, switchEdge) = ind(1:nnz(switchEdge));
    end
    
    G = digraph(s, t);
end

