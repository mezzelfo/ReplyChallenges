#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <set>
#include <array>
#include <map>
#include <utility>
#include <queue>
#include <math.h>
#include <limits>
constexpr double INF = std::numeric_limits<double>::max();

using namespace std;

typedef pair<int,int> Point;
typedef array<Point,3> Triangle;

ostream & operator << (ostream &out, const Point &P) {return out << P.first<<" "<<P.second;}
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

int ccw (const Point& p0, const Point& p1, const Point& p2)
{
   int dx1,dx2,dy1,dy2;

   dx1 = p1.first-p0.first; 
   dy1 = p1.second-p0.second;
   dx2 = p2.first-p0.first; 
   dy2 = p2.second-p0.second;

   if (dx1*dy2 > dy1*dx2) return +1;
   if (dx1*dy2 < dy1*dx2) return -1;
   if ((dx1*dx2 < 0) || (dy1*dy2 < 0)) return -1; 
   if ((dx1*dx1+dy1*dy1) < (dx2*dx2+dy2*dy2)) return +1;
   return 0;   
}

int intersect(const Point& l1p1, const Point& l1p2, const Point& l2p1, const Point& l2p2)
{
   return ((ccw(l1p1,l1p2,l2p1)*ccw(l1p1,l1p2,l2p2)) <=0) &&
          ((ccw(l2p1,l2p2,l1p1)*ccw(l2p1,l2p2,l1p2)) <=0);
}


bool are_points_visible(const Point& p0, const Point& p1, const vector<Triangle>& obsV)
{
    for(auto& tri : obsV)
    {
        if (intersect(p0,p1,tri[0],tri[1])) return false;
        if (intersect(p0,p1,tri[0],tri[2])) return false;
        if (intersect(p0,p1,tri[2],tri[1])) return false;
    }
    return true;
}

inline double distance(const Point& P1, const Point& P2)
{
    return sqrt((P1.first-P2.first)*(P1.first-P2.first)+(P1.second-P2.second)*(P1.second-P2.second));;
}

list<Point> visibleFrom(const Point& u, const set<Point>& Vertices, const vector<Triangle>& obsV)
{
	list<Point> l;
	l.clear();
	for (auto& P : Vertices)
        if (are_points_visible(u,P,obsV))
            l.push_back(P);
	return l;
}

int main(int argc, char const *argv[])
{
    ifstream inFile;
    if (argc != 2)
        throw std::runtime_error("Please use ./a.out <namefileinput>");
    inFile.open(argv[1]);
    if (!inFile)
        throw std::runtime_error("Unable to open input file");

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

    //cout << "Number of nodes: "<<Vertices.size()<<endl;
    set< pair<double, Point> > setds;
    map<Point, double> dist;
    map<Point, Point> parent;
    setds.insert(make_pair(0.0, startPoint)); 
    for (auto& P : Vertices) dist[P] = INF;
    dist[startPoint] = 0;
    while (!setds.empty()) 
    {
    	Point u = setds.begin()->second;
    	if(u == endPoint)
    		break;
    	setds.erase(setds.begin());
    	auto vis = visibleFrom(u, Vertices, obsV);
        for (Point& v: vis) 
        {
            double weight = distance(u,v) + distance(v,endPoint) - distance(u,endPoint);
            if (dist[v] > dist[u] + weight) 
            { 
                if (dist[v] != INF) 
                    setds.erase(setds.find(make_pair(dist[v], v))); 
                dist[v] = dist[u] + weight; 
                setds.insert(make_pair(dist[v], v)); 
                parent[v] = u;
            }
        } 
    } 

    list<Point> shortestPath;
    Point p = endPoint;
    do
    {
        shortestPath.push_front(p);
        p = parent[p];
    }while(p != startPoint);
    shortestPath.push_front(startPoint);

    cout << shortestPath.size() << endl;
    for (auto& P: shortestPath)
        cout << P << endl;

    return 0;
}
