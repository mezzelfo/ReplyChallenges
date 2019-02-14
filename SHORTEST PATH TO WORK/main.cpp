#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <array>
#include <map>
#include <utility>
#include <queue>
#include <math.h>

using namespace std;

typedef pair<int,int> Point;
typedef array<Point,3> Triangle;

ostream & operator << (ostream &out, const Point &P) {return out << "("<<P.first<<", "<<P.second<<")";}
istream & operator >> (istream &in,  Point &P) {return in >> P.first >> P.second;}
istream & operator >> (istream &in,  Triangle &T) {return in >> T[0] >> T[1] >> T[2];}

array<Point,12> points_near_triangle(Triangle& T)
{
    array<Point,12> v;
    for(int i = 0; i < 3; i++)
    {
        v[4*i+0] = T[i];
        v[4*i+0].first++;
        v[4*i+1] = T[i];
        v[4*i+1].first--;
        v[4*i+2] = T[i];
        v[4*i+2].second++;
        v[4*i+3] = T[i];
        v[4*i+3].second--;
    }
    return v;
}

int sign(const Point& p1, const Point& p2, const Point& p3)
{
  return (p1.first - p3.first) * (p2.second - p3.second) - (p2.first - p3.first) * (p1.second - p3.second);
}

bool is_point_in_triangle(const Point& P, const Triangle& T)
{
    bool b1, b2, b3;

    b1 = sign(P, T[0], T[1]) <= 0;
    b2 = sign(P, T[1], T[2]) <= 0;
    b3 = sign(P, T[2], T[0]) <= 0;

    return ((b1 == b2) && (b2 == b3));
}

int side(const Point& p, const Point& q, const Point& a, const Point& b)
{
    int z1 = (b.first - a.first) * (p.second - a.second) - (p.first - a.first) * (b.second - a.second);
    int z2 = (b.first - a.first) * (q.second - a.second) - (q.first - a.first) * (b.second - a.second);
    return z1 * z2;
}

bool are_points_visible(const Point& p0, const Point& p1, const vector<Triangle>& obsV)
{
    for(auto& tri : obsV)
    {
        Point t0 = tri[0];
        Point t1 = tri[1];
        Point t2 = tri[2];
       /* Check whether segment is outside one of the three half-planes
     * delimited by the triangle. */
    float f1 = side(p0, t2, t0, t1), f2 = side(p1, t2, t0, t1);
    float f3 = side(p0, t0, t1, t2), f4 = side(p1, t0, t1, t2);
    float f5 = side(p0, t1, t2, t0), f6 = side(p1, t1, t2, t0);
    /* Check whether triangle is totally inside one of the two half-planes
     * delimited by the segment. */
    float f7 = side(t0, t1, p0, p1);
    float f8 = side(t1, t2, p0, p1);

    /* If segment is strictly outside triangle, or triangle is strictly
     * apart from the line, we're not intersecting */
    if ((f1 < 0 && f2 < 0) || (f3 < 0 && f4 < 0) || (f5 < 0 && f6 < 0)
          || (f7 > 0 && f8 > 0)) continue;

    /* If segment is aligned with one of the edges, we're overlapping */
    if ((f1 == 0 && f2 == 0) || (f3 == 0 && f4 == 0) || (f5 == 0 && f6 == 0)) return 0;

    /* If segment is outside but not strictly, or triangle is apart but
     * not strictly, we're touching */
    if ((f1 <= 0 && f2 <= 0) || (f3 <= 0 && f4 <= 0) || (f5 <= 0 && f6 <= 0)
          || (f7 >= 0 && f8 >= 0)) return 0;

    /* If both segment points are strictly inside the triangle, we
     * are not intersecting either */
    if (f1 > 0 && f2 > 0 && f3 > 0 && f4 > 0 && f5 > 0 && f6 > 0) continue;

    /* Otherwise we're intersecting with at least one edge */
    return 0;
        

    }
    return 1;
}

inline double distance(const Point& P1, const Point& P2)
{
    return sqrt((P1.first-P2.first)*(P1.first-P2.first)+(P1.second-P2.second)*(P1.second-P2.second));;
}

int main(int argc, char const *argv[])
{
    ifstream inFile;
    if (argc != 2)
    {
        cerr << "Please use ./a.out <namefileinput>" << endl;
        exit(EXIT_FAILURE);
    }
    inFile.open(argv[1]);
    if (!inFile)
    {
        cerr << "Unable to open input file" << endl;
        exit(EXIT_FAILURE);
    }

    Point startPoint, endPoint;
    int obsCount;
    inFile >> startPoint >> endPoint;
    inFile >> obsCount;
    vector<Triangle> obsV;
    obsV.reserve(obsCount);
    for(int i = 0; i < obsCount; i++)
    {
        Triangle T;
        inFile >> T;
        obsV.push_back(T);
    }
    obsV.shrink_to_fit();
    inFile.close();

    for(auto& tri : obsV)
    {
        if (is_point_in_triangle(startPoint,tri) || is_point_in_triangle(endPoint,tri)) {
            cout << "IMPOSSIBLE" << endl;
            exit(EXIT_SUCCESS);
        }
    }

    set<Point> Vertices;
    Vertices.emplace(startPoint);
    Vertices.emplace(endPoint);
    for(auto& tri1 : obsV)
    {
        const auto v = points_near_triangle(tri1);
        array<bool,v.size()> needed;
        needed.fill(true);
        for(auto& tri2 : obsV)
            for(unsigned i = 0; i < v.size(); i++)
                if (needed[i] && is_point_in_triangle(v[i],tri2)) needed[i] = false;
        for(unsigned i = 0; i < v.size(); i++)
            if (needed[i]) Vertices.emplace(v[i]);
    }

    cout << Vertices.size() << endl;

    map<Point,double> heap;
    for(auto& P : Vertices) heap[P]=9999;
    heap[startPoint] = 0;


    cout << heap.begin()->first << endl;
    

    

    
/*
    exit(1);
    map<Point,set<pair<Point,double>>> graph;
    for(auto& P1 : Vertices)
        for(auto& P2 : Vertices)
            if (are_points_visible(P1,P2,obsV))
                graph[P1].emplace(P2,distance(P1,P2));
*/



    
        


    return 0;
}
