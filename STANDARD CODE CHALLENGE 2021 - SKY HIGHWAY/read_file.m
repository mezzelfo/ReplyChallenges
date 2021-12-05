function read_file(fname)
global W H N M R buildings_features antennas_features
fileID = fopen(fname);
dim = fscanf(fileID,'%d',[1,2]);
nums = fscanf(fileID,'%d',[1,3]);
W = dim(1);
H = dim(2);
N = nums(1); %Number of buildings
M = nums(2); %Number of antennas
R = nums(3); %Reward
buildings_features = fscanf(fileID,'%d',[4,N]);
antennas_features = fscanf(fileID,'%d',[2,M]);
fclose(fileID);
end