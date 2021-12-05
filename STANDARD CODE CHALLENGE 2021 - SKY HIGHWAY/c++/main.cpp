#include <iostream>
#include <vector>
#include <set>
#include <fstream>
#include <algorithm>

using namespace std;

struct Building
{
    int x, y, latency_weight, connection_speed;
    int connected_to = -1;
    int connection_score;
    Building(const int x, const int y, const int l, const int c)
        : x(x), y(y), latency_weight(l), connection_speed(c){};
};

struct Antenna
{
    int range, connection_speed, ID;
    int x, y;
    set<int> buildings_connected;
    Antenna(const int r, const int c, const int id)
        : range(r), connection_speed(c), ID(id){};
};

struct Matrix
{
    int *map;
    size_t W, H;

    void reserve(const size_t w, const size_t h)
    {
        W = w;
        H = h;
        map = new int[W * H];
        for (int i = 0; i < W*H; i++)
            map[i] = -1;        
    }

    const int get(const size_t x, const size_t y)
    {
        return map[x + W * y];
    }

    void set(const size_t x, const size_t y, int v)
    {
        map[x + W * y] = v;
    }

    ~Matrix()
    {
        delete [] map;
    }
};

int W, H, N, M, R;

vector<Building> buildings;
vector<Antenna> antennas;
Matrix buildings_map;

const int evaluate_position_optimistic(const int x, const int y, const Antenna& a)
{
    long long sum = 0;
    for (int xx = max(x - a.range,0); xx <= min(x + a.range,W-1); ++xx)
    {
        int dif = a.range - abs(x - xx);
        for (int yy = max(y - dif,0); yy <= min(y + dif,H-1); ++yy)
        {
            int build_idx = buildings_map.get(xx,yy);
            if (build_idx == -1)
                continue;
            const Building& b = buildings[build_idx];
            int dist = abs(x-xx)+abs(y-yy);
            int long contrib = b.connection_speed*a.connection_speed - b.latency_weight*dist;
            if (contrib > 0)
            {
                sum += contrib;
            }
        }
    }
    return sum;
}

int main(int argc, char const *argv[])
{
    ifstream file(argv[1], ios::in);
    file >> W >> H >> N >> M >> R;
    buildings.reserve(N);
    antennas.reserve(M);
    buildings_map.reserve(W,H);

    for (size_t i = 0; i < N; i++)
    {
        int x, y, l, c;
        file >> x >> y >> l >> c;
        buildings.emplace_back(x, y, l, c);
        if ((x > W) or (y > H))
        {
            cerr << x << " " << y << endl;
            cerr << W << " " << H << endl;
        }
        buildings_map.set(x,y,i);
    }
    for (size_t i = 0; i < M; i++)
    {
        int r, c;
        file >> r >> c;
        antennas.emplace_back(r, c, i);
    }

    file.close();

    
    long long bestsum = 0;
    long long secondbestsum = 0;
    for (const Antenna& a : antennas)
    {
        cout << a.ID << "/" << M << " -> (" << a.range << "," << a.connection_speed << ")"<<endl;
        long long best = -1;
        long long secondbest = -2;
        for (int x = 0; x < W; x++)
        {
            for (int y = 0; y < H; y++)
            {
                auto res = evaluate_position_optimistic(x,y,a);
                if (res > best)
                {
                    secondbest = best;
                    best = res;
                }
                else if ((res > secondbest)&(res != best))
                {
                    secondbest = res;
                }

                if (best == secondbest)
                {
                    cout << a.ID << endl 
                        << best << endl
                        << secondbest << endl
                        << x << " " << y << endl;
                    return -1;
                }
                
            }
        }
        bestsum += best;
        secondbestsum += secondbest;
    }
    
    cout << argv[1] << ":\t"<< "best sum" << bestsum << "\tsecond best sum" << secondbestsum <<  endl;
    return bestsum;
}
