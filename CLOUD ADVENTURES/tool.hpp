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
    unsigned availablePackages;
    double packageUnitCost;
    std::map<Service,unsigned> packagesPerServices;
    std::map<Country,unsigned> latencyPerCountry;
};
class Provider
{
    std::string name;
    std::vector<Region> regions;
};
class Project
{
    unsigned basePenality;
    Country country;
    std::map<Service,unsigned> unitsNeededPerServices;
};

#endif
