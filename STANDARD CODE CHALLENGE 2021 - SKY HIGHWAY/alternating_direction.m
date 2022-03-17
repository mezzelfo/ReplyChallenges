clear all
clc

global W H N M R buildings_features antennas_features

read_file('data_scenarios_b_mumbai.in');
%bound_above();
%[antennas_positions] = get_good_starting_pos();
%tic
%get_total_score(antennas_positions)
%toc
%return
%unique(antennas_features(1,:))
%hist(antennas_features(1,:))

fileID = fopen('good_starting_pos/b_mio.txt');
placed_antennas = fscanf(fileID,'%d',[1,1]);
good_start_pos = fscanf(fileID,'%d',[3,placed_antennas]);
antennas_positions(:,good_start_pos(1,:)+1) = good_start_pos(2:3,:);
fclose(fileID);

init_position = antennas_positions;
pippo_pluto = zeros(M,1);

%Get connection structure: given a building, get the associated antenna
connections = zeros(N,1);
score = 0;
for b = 1:N
    distances = vecnorm(antennas_positions - buildings_features(1:2,b),1);
    near_antennas = find(distances <= antennas_features(1,:));
    if any(near_antennas)
        scores = buildings_features(4,b)*antennas_features(2,near_antennas) - buildings_features(3,b)*distances(near_antennas);
        [max_val, max_idx] = max(scores);
        score = score + max_val;
        connections(b) = near_antennas(max_idx);
        %             if max_val > 0
        %                 connections(b) = near_antennas(max_idx);
        %             else
        %                 connections(b) = NaN;
        %             end
    else
        connections(b) = NaN;
    end
end

score
res = [sum(isnan(connections))/length(connections), score];
res

for iteration = 1:1
    
    %Optimize placement, keeping fixed connection structure
    opts1=  optimset('display','off');
    for a = 1:M
        near_buildings = find(connections == a);
        %         switch length(near_buildings)
        %             case 0
        %                 antennas_positions(:,a) = [randi(W); randi(H)];
        %             otherwise
        %                 near_buildings_position = buildings_features(1:2,near_buildings);
        %                 antennas_positions(:,a) = mean(near_buildings_position,2);
        %         end
        
        %         switch length(near_buildings)
        %             case 0
        %                 %Randomly place again this antenna
        %                 antennas_positions(:,a) = [randi(W); randi(H)];
        %             case 1
        %                 antennas_positions(:,a) = buildings_features(1:2,near_buildings);
        %             otherwise
        %                 near_buildings_latency = buildings_features(3,near_buildings);
        %                 if sum(near_buildings_latency) < 0.5
        %                     continue
        %                 end
        %                 near_buildings_position = buildings_features(1:2,near_buildings);
        %                 antenna_range = antennas_features(1,a);
        %                 %Solve linear optimization problem
        %                 prob = optimproblem('ObjectiveSense','minimize');
        %                 x = optimvar('x','LowerBound',0,'UpperBound',W);
        %                 y = optimvar('y','LowerBound',0,'UpperBound',H);
        %                 t = optimvar('t',2,length(near_buildings),'LowerBound',0);
        %                 deltas = repmat([x;y],[1,length(near_buildings)]) - near_buildings_position;
        %                 prob.Objective = dot(near_buildings_latency, sum(t,1));
        %                 prob.Constraints.abs1 = t >= deltas;
        %                 prob.Constraints.abs2 = t >= -deltas;
        %                 prob.Constraints.range = sum(t,1) <= antenna_range;
        %                 [sol,fval,EXITFLAG] = solve(prob,'Options',opts1);
        %                 assert(EXITFLAG > 0);
        %                 antennas_positions(:,a) = [sol.x; sol.y];
        %         end
        
        switch length(near_buildings)
            case 0
                %Randomly place again this antenna
                antennas_positions(:,a) = [randi(W); randi(H)];
                pippo_pluto(a) = 1;
            case 1
                antennas_positions(:,a) = buildings_features(1:2,near_buildings);
                pippo_pluto(a) = 2;
            otherwise
                near_buildings_latency = buildings_features(3,near_buildings);
                near_buildings_position = buildings_features(1:2,near_buildings);
                
                B = length(near_buildings);
                
                A = [-ones(B,1) zeros(B,1) -eye(B) zeros(B) zeros(B);
                    +ones(B,1) zeros(B,1) -eye(B) zeros(B) zeros(B);
                    zeros(B,1) -ones(B,1) zeros(B) -eye(B) zeros(B);
                    zeros(B,1) +ones(B,1) zeros(B) -eye(B) zeros(B);
                    zeros(B,1) zeros(B,1) zeros(B) zeros(B) eye(B);
                    ];
                
                b = [-near_buildings_position(1,:) near_buildings_position(1,:) -near_buildings_position(2,:) near_buildings_position(2,:) ones(1,B)*antennas_features(1,a)];
                
                Aeq = [zeros(B,1) zeros(B,1) -eye(B) -eye(B) eye(B)];
                beq = zeros(B,1);
                
                f = [0 0 zeros(1,B) zeros(1,B) ones(1,B)];
                
                lb = zeros(2+3*B,1);
                ub = [];
                
                [x,fval,exitflag,output] = linprog(f,A,b,Aeq,beq,lb,ub,opts1);
                
                assert(exitflag > 0)
                
                antennas_positions(:,a) = x(1:2);
        end
        
    end
    
    %Recalculate score
    connections = zeros(N,1);
    score = 0;
    for b = 1:N
        distances = vecnorm(antennas_positions - buildings_features(1:2,b),1);
        near_antennas = find(distances <= antennas_features(1,:));
        if any(near_antennas)
            scores = buildings_features(4,b)*antennas_features(2,near_antennas) - buildings_features(3,b)*distances(near_antennas);
            [max_val, max_idx] = max(scores);
            score = score + max_val;
            connections(b) = near_antennas(max_idx);
                        if max_val > 0
                            connections(b) = near_antennas(max_idx);
                        else
                            connections(b) = NaN;
                        end
        else
            connections(b) = NaN;
        end
    end
    
    res(iteration+1,:) = [sum(isnan(connections))/length(connections), score];
    res
end

res(end,2)/res(1,2)
plot(res(:,2))

deltas = vecnorm(init_position - antennas_positions, 1);
sum(pippo_pluto == 0)
sum(pippo_pluto == 1)
sum(pippo_pluto == 2)
max(deltas(pippo_pluto == 0))
max(deltas(pippo_pluto == 1))
max(deltas(pippo_pluto == 2))
% scatter(deltas, pippo_pluto)
% max(deltas(pippo_pluto == 0))
% antennas_features(1,pippo_pluto == 1)
% max(deltas(pippo_pluto == 2))