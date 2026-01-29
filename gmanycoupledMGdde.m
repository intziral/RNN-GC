function [xM,sol] = gmanycoupledMGdde(n,DeltaV,cM,ts,sol0)
% [xM,sol] = gmanycoupledMGdde(n,DeltaV,c,ts,sol0)
% Runs a coupled Mackey Glass system comprised of a number of coupled 
% Mackey-Glass equations using dde23 and produces a data file
% of the multivariate time series.
% INPUTS
% - n       : The length of the time series
% - DeltaV  : Vector of delay parameter, one for each MG equation in sec (not
%             related to the given sample time), e.g. [17 30 100]
% - cM      : matrix of coupling strengths, not symmetric but with zeros in
%             the diagonal (no coupling with itself!), e.g.
%             [0 1 1; 0 0 1; 0 0 0] for X1->X2, X1->X3 and X2->X3, all with
%             strength of causal effect equal to 1.
% - ts      : The sampling time in sec
% OUTPUT
% - xM      : The generated multivariate time series

if nargin==4
    sol0=[];
elseif nargin==3
    sol0=[];
    ts = 1;
end
if isempty(ts), ts = 1; end
if size(cM,1)~=size(cM,2) 
    error('The matrix of coupling strengths should be square.');
end
if length(DeltaV)~=size(cM,1)
    error('The number of delays should match the dimension of the matrix of coupling strengths.');
end

% If zero in diagonal set it to 0.2;
if sum(abs(diag(cM)))==0
    for i=1:size(cM,1)
        cM(i,i)=0.2; 
    end
end

Trans = ts*30;     % Run for such long time (in sec) to reach steady state
Tsample = ts*n;  % save the time series for this time
% Run for transients only when initial conditions are not given
if isempty(sol0)
    sol0=rand(length(DeltaV),1);
end
sol = dde23(@ddemanycoupledMG,DeltaV,sol0,[0,Trans+Tsample],[],cM);
tV = [Trans+ts:ts:Trans+Tsample]';
xM = (deval(sol,tV))';

function dydt = ddemanycoupledMG(t,xtV,xlagM,cM)
    % Differential equations function for the coupled Mackey-Glass.
    xnowV = diag(xlagM);
    xnowV = xnowV ./ (1+xnowV.^10);
    dydt = -0.1*xtV + cM'*xnowV;
end
end