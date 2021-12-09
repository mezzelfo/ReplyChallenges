close all
clear all
clc

% A NONE
% B 2078043619
% C NONE
% D 5247238794
% E 8109310667


%% ../data_scenarios_b_mumbai.in

% fout = importdata('c++/fout_B.csv');
% sum(fout > 2078043619)/length(fout)
% subplot(2,2,1)
% plot(fout)
% yline(2078043619,'r')
%
% fout = importdata('c++/fout_D.csv');
% sum(fout > 5247238794)/length(fout)
% subplot(2,2,2)
% plot(fout)
% yline(5247238794,'r')
%
% fout = importdata('c++/fout_E.csv');
% sum(fout > 8109310667)/length(fout)
% subplot(2,2,3)
% hold on
% plot(fout)
% yline(8109310667,'r')
% hold off

%%
% global W H N M R buildings_features antennas_features
% 
% choice = 'E';
% switch choice
%     case 'B'
%         fileID = fopen('c++/antennas_contrib_B.csv');
%         TARGET = 2078043619;
%         read_file('data_scenarios_b_mumbai.in');
%     case 'D'
%         fileID = fopen('c++/antennas_contrib_D.csv');
%         TARGET = 5247238794;
%         read_file('data_scenarios_d_polynesia.in');
%     case 'E'
%         fileID = fopen('c++/antennas_contrib_E.csv');
%         TARGET = 8109310667;
%         read_file('data_scenarios_e_sanfrancisco.in');
% end
% ant_contrib = fscanf(fileID,'%d,');
% fclose(fileID);
% 
% figure()
% plot(sum(ant_contrib)-cumsum(sort(ant_contrib,'descend')))
% yline(TARGET,'r')
% 
% mean(sum(ant_contrib)-cumsum(sort(ant_contrib,'descend')) > TARGET)
% sum(sum(ant_contrib)-cumsum(sort(ant_contrib,'descend')) > TARGET)

% figure()
% scatter3(antennas_features(1,:),antennas_features(2,:),ant_contrib'./(1+antennas_features(1,:).^2))
% xlabel('range')
% ylabel('speed')
% zlabel('contrib')

%%
fileID = fopen('c++/fmapout.csv');
map = reshape(fscanf(fileID,'%d,'),[400,400]);
fclose(fileID);
imagesc(map)