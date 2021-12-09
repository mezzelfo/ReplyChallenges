clear all
clc
global W H N M R buildings_features antennas_features
read_file('data_scenarios_a_example.in');


problem = optimproblem('ObjectiveSense','max');
positions = optimvar('positions',M,2,'LowerBound',0);
distances = optimvar('distances',N,M,'LowerBound',0);
z = optimvar('z',N,M,'LowerBound',0,'UpperBound',1,'Type','integer');
bonus = optimvar('bonus','LowerBound',0,'UpperBound',1,'Type','integer');

rangeConstr = optimconstr(N,M);
distanceConstr = optimconstr(N,M,4);
problem.Objective = 0;
for b=1:N
    for a=1:M
        rangeConstr(b,a) = distances(b,a)-antennas_features(1,a) <= (W+H)*(1-z(b,a));
        distanceConstr(b,a,1) = distances(b,a) >= +(buildings_features(1,b)-positions(a,1))+(buildings_features(2,b)-positions(a,2));
        distanceConstr(b,a,2) = distances(b,a) >= +(buildings_features(1,b)-positions(a,1))-(buildings_features(2,b)-positions(a,2));
        distanceConstr(b,a,3) = distances(b,a) >= -(buildings_features(1,b)-positions(a,1))+(buildings_features(2,b)-positions(a,2));
        distanceConstr(b,a,4) = distances(b,a) >= -(buildings_features(1,b)-positions(a,1))-(buildings_features(2,b)-positions(a,2));
        
        %problem.Objective = problem.Objective - buildings_features(3,b)*distances(b,a);
    end
end

problem.Objective = problem.Objective + buildings_features(4,:)*z*antennas_features(2,:)' + bonus*R;

problem.Constraints.rangeConstr = rangeConstr;
problem.Constraints.distanceConstr = distanceConstr;
problem.Constraints.boxConstr = positions <= repmat([W,H],M,1);
problem.Constraints.connectedConstr = sum(z,2) <= 1;
problem.Constraints.bonusConstr = N*bonus <= sum(z,'all');

showproblem(problem)

% x0.positions = rand(M,2).*repmat([W,H],M,1);
% x0.distances = zeros(N,M);
% x0.z = zeros(N,M);
% x0.bonus = 1;
[sol, score] = solve(problem);%,x0,'Solver','surrogateopt');