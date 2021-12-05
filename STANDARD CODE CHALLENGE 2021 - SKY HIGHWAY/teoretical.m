clear all
clc

fileID = fopen('data_scenarios_b_mumbai.in');
dim = fscanf(fileID,'%d',[1,2]);
nums = fscanf(fileID,'%d',[1,3]);
W = dim(1);
H = dim(2);
N = nums(1); %Number of buildings
M = nums(2); %Number of antennas
R = nums(3);
clear dim nums

buildings = fscanf(fileID,'%d',[4,N])';
antennas = fscanf(fileID,'%d',[2,M])';
fclose(fileID);

A = rand(N,M);
[~,idx] = max(A,[],2);
newA = zeros(N,M);
newA(idx,:) = 1;
A = newA;
dA = (buildings(:,3:4) * (antennas.*[1,-1])');
dA = dA/max(max(abs(dA)));
mu = @(i) 0.05*0.9^i;

for i = 1:100
    A = A+mu(i)*dA;
    [~,idx] = max(A,[],2);
    newA = zeros(N,M);
    newA(idx,:) = 1;
    %A = newA;
    %A = A./sum(A,2);
    trace((buildings(:,3:4))' * newA * (antennas.*[1,-1]))
end

% problem = optimproblem('ObjectiveSense','max');
% positions = optimvar('positions',M,2,'LowerBound',0);
% z = optimvar('z',N,M,'LowerBound',0);
% s = optimvar('s',N);
% problem.Constraints.boxConstr = positions <= repmat([W,H],M,1);
% sConstr = optimconstr(N,M);
% zConstr = optimconstr(N,M,4);
% for i=1:N
%     for j=1:M
%         sConstr(i,j) = s(i) >= buildings(i,4)*antennas(j,2) - buildings(i,3)*z(i,j);
%         zConstr(i,j,1) = z(i,j) >= +(buildings(i,1)-positions(j,1))+(buildings(i,2)-positions(j,2));
%         zConstr(i,j,2) = z(i,j) >= +(buildings(i,1)-positions(j,1))-(buildings(i,2)-positions(j,2));
%         zConstr(i,j,3) = z(i,j) >= -(buildings(i,1)-positions(j,1))+(buildings(i,2)-positions(j,2));
%         zConstr(i,j,4) = z(i,j) >= -(buildings(i,1)-positions(j,1))-(buildings(i,2)-positions(j,2));
%     end
% end
% problem.Constraints.sConstr = sConstr;
% problem.Constraints.zConstr = zConstr;

% problem.Constraints.boxConstr = positions <= repmat([W,H],M,1);
% problem.Constraints.testConstr1 = bonus <= choosen*ones(M,1);
% problem.Constraints.testConstr2 = choosen*ones(M,1) <= 1;
% problem.Constraints.absConstr1 = norm1 >= +(buildings(:,[1,2])-choosen*positions);
% problem.Constraints.absConstr2 = norm1 >= -(buildings(:,[1,2])-choosen*positions);
% problem.Constraints.areaConstr = norm1*[1;1] <= choosen * antennas(:,1);

% problem.Objective = sum(s);
% showproblem(problem)

% x0.positions = rand(M,2).*repmat([W,H],M,1);
% x0.choosen = zeros(N,M);
% x0.test = 1;
% [sol, score] = solve(problem);%,x0);
% score
% sol.positions
% 
