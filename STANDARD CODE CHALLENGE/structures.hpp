#ifndef STRUCTURES_HPP
#define STRUCTURES_HPP

#include <fstream>
#include <array>

struct CustomerHQ
{
    int x,y,reward;
};

struct Reply
{
    unsigned x,y;
};

std::istream& operator >> (std::istream& in, CustomerHQ& c) { return (in >> c.x >> c.y >> c.reward); }
bool operator < (const CustomerHQ& a, const CustomerHQ& b) { return (a.reward > b.reward); }


#endif

