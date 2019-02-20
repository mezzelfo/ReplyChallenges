#ifndef TOOL_HPP
#define TOOL_HPP
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
#include <limits>
typedef std::string Service;
typedef std::string Country;
typedef long long int num;


struct Region
{
    std::string name;
    num availablePackages;
    double packageUnitCost;
    std::map<Service,num> packagesPerServices;
    std::map<Country,num> latencyPerCountry;
    friend std::istream& operator>>(std::ifstream& is, Region& r);
    bool operator<(const Region& other) const
    {
        return name < other.name;
    }
};
struct Provider
{
    std::string name;
    std::vector<Region> regions;
    friend std::istream& operator>>(std::ifstream& is, Provider& p);
};
struct Project
{
    num basePenality;
    Country country;
    std::map<Service,num> unitsNeededPerServices;
    num total_units_needed();
    bool is_possible(std::map<Service,num> tot);
    friend std::istream& operator>>(std::ifstream& is, Project& p);
};

template<class T> T read_from_file(std::ifstream& input)
{
    T tmp;
    input >> tmp;
    if(!input)
        throw std::runtime_error("Read bad value from file");
    return tmp;
}

double RegionPenalty(Project& P, Region& R, num num_uses);

#endif
