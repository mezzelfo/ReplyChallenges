#ifndef TOOL_HPP
#define TOOL_HPP
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
typedef std::string Service;
typedef std::string Country;

class Region
{
    std::string name;
    int availablePackages;
    float packageUnitCost;
    std::map<Service,int> packagesPerServices;
    std::map<Country,int> latencyPerCountry;
    friend std::istream& operator>>(std::ifstream& is, Region& r);
};
class Provider
{
    std::string name;
    std::vector<Region> regions;
    friend std::istream& operator>>(std::ifstream& is, Provider& p);
};
class Project
{
    int basePenality;
    Country country;
    std::map<Service,int> unitsNeededPerServices;
    friend std::istream& operator>>(std::ifstream& is, Project& p);
};

template<class T> T read_from_file(std::ifstream& input)
{
    T tmp;
    input >> tmp;
    return tmp;
}

#endif
