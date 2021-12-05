function [antennas_positions] = get_good_starting_pos()
global W H N M R buildings_features antennas_features

%% Random
%antennas_positions = [randi(W,[1,M]); randi(H,[1,M])];%Place randomly the antennas

%% Rearrangement Inequality
[~,sorted_antennas_idx] = sort(antennas_features(end,:),'descend');
[~,sorted_buildings_idx] = sort(buildings_features(end,:),'descend');
antennas_positions(:,sorted_antennas_idx) = buildings_features(1:2,sorted_buildings_idx(1:M));


%% Greedy
% building_latency_map = full(sparse(...
%     1+buildings_features(1,:),...
%     1+buildings_features(2,:),...
%     buildings_features(4,:),...
%     W,H));
% 
% building_connection_map = full(sparse(...
%     1+buildings_features(1,:),...
%     1+buildings_features(2,:),...
%     buildings_features(4,:),...
%     W,H));
% 
% 
% antennas_map = zeros(W,H);
% antennas_positions = NaN*ones(2,M);
% 
% [~,sorted_ant_idx] = sort(sqrt(antennas_features(1,:))+antennas_features(2,:),'descend');
% 
% for a = sorted_ant_idx
%     R = antennas_features(1,a);
%     L1_dist_ker = (abs(-R:R)+abs(-R:R)');
%     L1_reachable_ker = L1_dist_ker <= R;
%     L1_dist_ker = L1_dist_ker .* L1_reachable_ker;
% 
%     latency_conv = conv2(building_latency_map, L1_dist_ker,'same');
%     connection_conv = conv2(building_connection_map,L1_reachable_ker,'same');
%     score_map = connection_conv - latency_conv;
%     score_map(antennas_map > 0) = -Inf;
%     [max_val,max_idx] = max(score_map,[],'all');
% 
%     [r,c] = ind2sub(size(score_map),max_idx);
%     antennas_positions(:,a) = [r;c];
% 
%     antennas_map(max_idx) = a;
%     building_latency_map = building_latency_map .* (conv2(antennas_map,L1_reachable_ker,'same') == 0);
%     building_connection_map = building_connection_map .* (conv2(antennas_map,L1_reachable_ker,'same') == 0);
% end
% 
% [rows,cols,ant_idx] = find(antennas_map);
% 
% assert(length(ant_idx) == M)
% 
% antennas_positions = NaN*ones(2,M);
% antennas_positions(:,ant_idx) = [rows,cols]';

% hold on
% spy(sparse(...
%     1+buildings_features(1,:),...
%     1+buildings_features(2,:),...
%     ones([N,1]),...
%     W,H))
% spy(map_antennas,'ro')
% hold off
end