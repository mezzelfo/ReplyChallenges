global W H N M R buildings_features antennas_features

building_latency_map = full(sparse(...
    1+buildings_features(1,:),...
    1+buildings_features(2,:),...
    buildings_features(4,:),...
    W,H));

building_connection_map = full(sparse(...
    1+buildings_features(1,:),...
    1+buildings_features(2,:),...
    buildings_features(4,:),...
    W,H));

[unique_ants_feat,ia,ic] = unique(antennas_features','rows'); %C = A(ia,:) and A = C(ic,:)
a_counts = accumarray(ic,1);


bound_res = 0;
contribs = [];
for ua = 1:length(unique_ants_feat)
    R = unique_ants_feat(ua,1);
    
    L1_dist_ker = (abs(-R:R)+abs(-R:R)');
    L1_reachable_ker = L1_dist_ker <= R;
    L1_dist_ker = L1_dist_ker .* L1_reachable_ker;
    
    num = a_counts(ua);

    latency_conv = conv2(building_latency_map, L1_dist_ker,'same');
    connection_conv = conv2(building_connection_map,L1_reachable_ker,'same');
    score_map = connection_conv - latency_conv;
    best_val = max(score_map,[],'all');
    contribs(end+1) = num*best_val;
    bound_res =  bound_res + num*max(best_val,0);
    %candidate_positions = score_map >= best_val; %0.8*?
    %candidate_positions = conv2(candidate_positions,L1_reachable_ker,'same'); 
    %far_from_max_score_map = score_map -10^7*candidate_positions;
    %second_best_val = max(far_from_max_score_map,[],'all');
    %bound_res =  bound_res + num*max(second_best_val,0);
    if R > 10
        break
    end
end
bound_res