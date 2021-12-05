function [score, connections] = get_total_score(antennas_positions)
global W H N M R buildings_features antennas_features
connections = zeros(N,1);
score = 0;
for b = 1:N
    distances = sum(abs(antennas_positions - buildings_features(1:2,b)));
    near_antennas = distances <= antennas_features(1,:);
    if any(near_antennas)
        scores = buildings_features(4,b)*antennas_features(2,near_antennas) - buildings_features(3,b)*distances(near_antennas);
        [max_val, max_idx] = max(scores);
        score = score + max_val;
        connections(b) = near_antennas(max_idx);
    else
        connections(b) = NaN;
    end
end

if sum(isnan(connections)) == 0
    score = score + R;
end

end