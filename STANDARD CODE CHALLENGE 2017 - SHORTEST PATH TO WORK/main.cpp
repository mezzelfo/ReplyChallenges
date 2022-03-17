#include <iostream>
#include <fstream>
#include <list>
#include <set>
#include <map>
#include <string>
#include <assert.h>
#include <math.h>
#include <algorithm>
#include "IndexedHeap.hpp"

#include <limits>
constexpr double INF = std::numeric_limits<double>::max();

using namespace std;

typedef pair<int, int> Point;
typedef array<Point, 3> Triangle;
//typedef pair<Point,Point> BoundingBox;

ostream &operator<<(ostream &out, const Point &P) { return out << P.first << " " << P.second; }
istream &operator>>(istream &in, Point &P) { return in >> P.first >> P.second; }
istream &operator>>(istream &in, Triangle &T) { return in >> T[0] >> T[1] >> T[2]; }

array<Point, 12> points_near_triangle(Triangle &T)
{
    array<Point, 12> v;
    for (int i = 0; i < 3; i++)
    {
        v[4 * i + 0] = T[i];
        v[4 * i + 0].first++;
        v[4 * i + 1] = T[i];
        v[4 * i + 1].first--;
        v[4 * i + 2] = T[i];
        v[4 * i + 2].second++;
        v[4 * i + 3] = T[i];
        v[4 * i + 3].second--;
    }
    return v;
}

int sign(const Point &p, const Point &A, const Point &B)
{
    // Ritorna la distanza con segno rispetto alla retta orientata da A a B.
    // Se è positiva allora il punto è a sinistra della retta
    return (p.first - B.first) * (A.second - B.second) - (p.second - B.second) * (A.first - B.first);
}

bool is_point_in_triangle(const Point &P, const Triangle &T)
{
    bool b1, b2, b3;

    b1 = sign(P, T[0], T[1]) <= 0;
    b2 = sign(P, T[1], T[2]) <= 0;
    b3 = sign(P, T[2], T[0]) <= 0;

    return ((b1 == b2) && (b2 == b3));
}

int ccw(const Point &p0, const Point &p1, const Point &p2)
{
    int dx1, dx2, dy1, dy2;

    dx1 = p1.first - p0.first;
    dy1 = p1.second - p0.second;
    dx2 = p2.first - p0.first;
    dy2 = p2.second - p0.second;

    if (dx1 * dy2 > dy1 * dx2)
        return +1;
    if (dx1 * dy2 < dy1 * dx2)
        return -1;
    if ((dx1 * dx2 < 0) || (dy1 * dy2 < 0))
        return -1;
    if ((dx1 * dx1 + dy1 * dy1) < (dx2 * dx2 + dy2 * dy2))
        return +1;
    return 0;
    return 0;
    return 0;
    return 0;
    return 0;
}

int intersect(const Point &l1p1, const Point &l1p2, const Point &l2p1, const Point &l2p2)
{
    return ((ccw(l1p1, l1p2, l2p1) * ccw(l1p1, l1p2, l2p2)) <= 0) &&
           ((ccw(l2p1, l2p2, l1p1) * ccw(l2p1, l2p2, l1p2)) <= 0);
}

// BoundingBox getBoundingBox(const Point &p0, const Point &p1)
// {
//     BoundingBox bb;
//     bb.first.first = min(p0.first, p1.first);
//     bb.first.second = max(p0.second, p1.second);
//     bb.second.first = max(p0.first, p1.first);
//     bb.second.second = min(p0.second, p1.second);
//     return bb;
// }

// BoundingBox getBoundingBox(const Triangle &T)
// {
//     BoundingBox bb;
//     bb.first.first = std::min({T[0].first, T[1].first, T[2].first});
//     bb.first.second = std::max({T[0].second, T[1].second, T[2].second});
//     bb.second.first = std::max({T[0].first, T[1].first, T[2].first});
//     bb.second.second = std::min({T[0].second, T[1].second, T[2].second});
//     return bb;
// }

// bool BoundingBoxesOverlap(const BoundingBox& B1, const BoundingBox& B2)
// {
//     auto xMin1 = B1.first.first;
//     auto yMin1 = B1.second.second;
//     auto xMax1 = B1.second.first;
//     auto yMax1 = B1.first.second;

//     auto xMin2 = B2.first.first;
//     auto yMin2 = B2.second.second;
//     auto xMax2 = B2.second.first;
//     auto yMax2 = B2.first.second;

//     if ((xMin1 < xMax2 || xMax1 > xMin2) && (yMin1 < yMax2 || yMax1 > yMin2))
//         return false;
//     return true;

// }

bool are_points_visible(const Point &p0, const Point &p1, const vector<Triangle> &obsV)
{
    if ((p0.first == p1.first) and (p0.second == p1.second))
        return false;
        
    for (auto &tri : obsV)
    {
        // Estendendo i lati di un triangolo a rette si ottiene una partizione del piano in 7 parti
        // Se i due punti stanno nella stessa parte sicuramente si vedono
        // Se i due punti stanno in due parti contigue sicuramente si vedono

        // bool b1_0 = sign(p0, tri[0], tri[1]) <= 0;
        // bool b2_0 = sign(p0, tri[1], tri[2]) <= 0;
        // bool b3_0 = sign(p0, tri[2], tri[0]) <= 0;

        // bool b1_1 = sign(p1, tri[0], tri[1]) <= 0;
        // bool b2_1 = sign(p1, tri[1], tri[2]) <= 0;
        // bool b3_1 = sign(p1, tri[2], tri[0]) <= 0;

        // unsigned crossings = (b1_0 != b1_1) + (b2_0 != b2_1) + (b3_0 != b3_1);

        // if (crossings >= 2)
        // {
        //     if (intersect(p0, p1, tri[0], tri[1]))
        //         return false;
        //     if (intersect(p0, p1, tri[0], tri[2]))
        //         return false;
        //     if (intersect(p0, p1, tri[2], tri[1]))
        //         return false;
        // }

        // int b0 = sign(tri[0], p0, p1);
        // int b1 = sign(tri[1], p0, p1);
        // int b2 = sign(tri[2], p0, p1);
        // if ((b0 == 0) or (b1 == 0) or (b2 == 0))
        // {
        //     cout << p0 << endl << p1 << endl
        //     << tri[0] << "\t" << b0 << endl
        //     << tri[1] << "\t" << b1 << endl
        //     << tri[2] << "\t" << b2 << endl
        //     << endl << flush;
        //     exit(-1);
        // }

        // int b0 = sign(tri[0], p0, p1);
        // int b1 = sign(tri[1], p0, p1);
        // int b2 = sign(tri[2], p0, p1);

        // if ((b0 == 0) or (b1 == 0) or (b2 == 0))
        // {

        // }
        // else
        // {
        //     if ((b0 != b1) or (b0 != b2) or (b1 != b2))
        //     {
        //         cout << p0 << endl << p1 << endl
        //         << tri[0] << "\t" << b0 << endl
        //         << tri[1] << "\t" << b1 << endl
        //         << tri[2] << "\t" << b2 << endl
        //         << endl << flush;
        //         exit(-1);
        //     }
        // }

        // if (!BoundingBoxesOverlap(getBoundingBox(p0,p1), getBoundingBox(tri)))
        //    continue; 
        
        

        if (intersect(p0, p1, tri[0], tri[1]))
            return false;
        if (intersect(p0, p1, tri[0], tri[2]))
            return false;
        if (intersect(p0, p1, tri[2], tri[1]))
            return false;
    }
    return true;
}

inline double distance(const Point &P1, const Point &P2)
{
    return sqrt((P1.first - P2.first) * (P1.first - P2.first) + (P1.second - P2.second) * (P1.second - P2.second));
    ;
}

list<Point> visibleFrom(const Point &u, const set<Point> &Vertices, const vector<Triangle> &obsV)
{
    list<Point> l;
    l.clear();
    for (auto &P : Vertices)
        if (are_points_visible(u, P, obsV))
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
    for (int i = 0; i < obsCount; i++)
    {
        Triangle T;
        inFile >> T;
        obsV.push_back(T);
    }
    obsV.shrink_to_fit();
    inFile.close();

    for (auto &tri : obsV)
    {
        if (is_point_in_triangle(startPoint, tri) || is_point_in_triangle(endPoint, tri))
        {
            cout << "IMPOSSIBLE" << endl;
            exit(EXIT_SUCCESS);
        }
    }

    set<Point> Vertices;
    Vertices.emplace(startPoint);
    Vertices.emplace(endPoint);
    for (auto &tri1 : obsV)
        for (auto &p : points_near_triangle(tri1))
        {
            bool obscured = false;
            for (auto &tri2 : obsV)
            {
                if (is_point_in_triangle(p, tri2))
                {
                    obscured = true;
                    break;
                }
            }
            if (!obscured)
                Vertices.emplace(p);
        }

    cout << "Number of nodes: " << Vertices.size() << endl
         << flush;

    IndexedHeap<Point, double> heap(Vertices);
    map<Point, Point> parent;
    map<Point, double> gScore;

    heap.insertOrUpdate(startPoint, distance(startPoint, endPoint));
    gScore[startPoint] = 0;

    while (!heap.empty())
    {
        Point current = heap.extract();
        if (current == endPoint)
        {
            cout << "Found\n";
            break;
        }
        for (Point &neighbor : visibleFrom(current, Vertices, obsV))
        {
            auto w = gScore.at(current) + distance(current, neighbor);
            auto search = gScore.find(neighbor);
            if ((search == gScore.end()) or (w < search->second))
            {
                parent[neighbor] = current;
                gScore[neighbor] = w;
                heap.insertOrUpdate(neighbor, w + distance(neighbor, endPoint));
            }
        }
    }
    cout << "Finished\n"
         << flush;

    list<Point> shortestPath;
    Point p = endPoint;
    do
    {
        shortestPath.push_front(p);
        p = parent[p];
    } while (p != startPoint);
    shortestPath.push_front(startPoint);

    cout << shortestPath.size() << endl;
    for (auto &P : shortestPath)
        cout << P << endl;

    return 0;
}
