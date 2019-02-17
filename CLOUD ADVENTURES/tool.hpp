#ifndef TOOL_HPP
#define TOOL_HPP
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <map>
typedef std::string Service;
typedef std::string Country;

struct Region
{
    std::string name;
    int availablePackages;
    float packageUnitCost;
    std::map<Service,int> packagesPerServices;
    std::map<Country,int> latencyPerCountry;
    friend std::istream& operator>>(std::ifstream& is, Region& r);
};
struct Provider
{
    std::string name;
    std::vector<Region> regions;
    friend std::istream& operator>>(std::ifstream& is, Provider& p);
};
struct Project
{
    int basePenality;
    Country country;
    std::map<Service,int> unitsNeededPerServices;
    long long total_units_needed();
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

#endif
