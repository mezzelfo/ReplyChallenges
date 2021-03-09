#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include "structures.hpp"


std::vector<CustomerHQ> hqVec;
int N, M, C, R;
int** map;
const std::map<char, int> symbolCost = {{'#', 999999999},{'~', 800},{'*', 200},{'+', 150},{'X', 120},{'_',100},{'H',70},{'T',50}};
const int delta[4][3] = {{0,+1,'D'},{0,-1,'U'},{+1,0,'R'},{-1,0,'L'}};
unsigned long score = 0;
        
void esplora_dintorni(int ox, int oy, int benza, Matrix* costmap)
{
    for(auto& d: delta)
    {
        int x = ox + d[0];
        int y = oy + d[1];
        if ((x<0) or (x>=N)) continue;
        if ((y<0) or (y>=M)) continue;
        if (benza - map[x][y] > (costmap->at(x,y)))
        {
            costmap->at(x,y) = benza - map[x][y];
            esplora_dintorni(x,y,benza-map[x][y],costmap);
        }
    }
}


int main(int argc, char const *argv[])
{
    if (argc != 2)
        throw std::runtime_error("Parametri non corretti. Usa ./a.out fileinput.txt");
    
    // Opening File
    std::ifstream finput;
    finput.open(argv[1]);
    if (not finput.is_open()) throw std::runtime_error("File input non trovato");

    // Reading first line. Names as per definition in pdf
    // N=map width   N=map height   C=num of Customer HQ   R=max buildable Reply
    finput >> N >> M >> C >> R;

    // Storing all the customer HQ in a vector
    hqVec.reserve(C);
    for(int i=0; i<C; i++)
    {
        CustomerHQ tmp;
        finput >> tmp;
        hqVec.push_back(tmp);
    }
    hqVec.shrink_to_fit();

    // Allocating the map
    map = new int*[N];
    for (int i = 0; i < N; i++) map[i] = new int[M];

    // Reading the map
    for(int row = 0; row < M; row++)
    {
        for(int col = 0; col < N; col++)
        {
            char c;
            finput >> c;
            map[col][row] = symbolCost.at(c);
        }
    }
    
    // Close input file
    finput.close();

    // Aggiorno il reward vero dei hqVec
    for(auto& hq : hqVec) hq.reward -= map[hq.x][hq.y];

    // Sort hqVec by reward. Highest reward first
    sort(hqVec.begin(),hqVec.end());

    for(auto& hq : hqVec)
    {
        //std::cout << hq.x << ' ' << hq.y << ' ' << hq.reward << '\n';
        for(auto& d : delta)
        {
            int x = hq.x + d[0];
            int y = hq.y + d[1];
            if ((x<0) or (x >= N)) continue;
            if ((y<0) or (y >= M)) continue;
        }
    }
    
    Matrix TotalCostMap(N,M);
    for (auto& hq : hqVec)
    {
        Matrix costmap(N,M);
        esplora_dintorni(hq.x,hq.y,hq.reward,&costmap);
        //std::cout<<hq.reward<<std::endl;
        //hq.positiveZone = costmap;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < M; j++)
                TotalCostMap.at(i,j) += costmap.at(i,j);
    }

    /*for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
                std::cout << TotalCostMap.at(i,j) << ' ';
        std::cout << '\n';
    }*/

    std::vector<std::pair<int,int>> positions;
    for(int r = 0; r < R; r++)
    {
        
        std::pair<int,int> pos = std::make_pair(0,0);
        int maxrew = 0;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < M; j++)
            {
                if (TotalCostMap.at(i,j) > maxrew)
                {
                    pos = std::make_pair(i,j);
                    maxrew = TotalCostMap.at(i,j);
                }
            }
        }
        std::cout << "Piazzo in ("<<pos.first<<", "<<pos.second<<")\n";
        TotalCostMap.at(pos.first,pos.second) = 0;

    }
            

    //std::cout << "Score: " << score << '\n';
    return 0;
}

