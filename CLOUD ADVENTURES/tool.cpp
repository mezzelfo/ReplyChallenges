#include "tool.hpp"

extern std::vector<Service> services;
extern std::vector<Country> countries;

std::istream& operator>>(std::ifstream& is, Region& r)
{
    r.name = read_from_file<std::string>(is);
    r.availablePackages = read_from_file<int>(is);
    r.packageUnitCost = read_from_file<float>(is);
    for(Service& s : services)
        r.packagesPerServices[s] = read_from_file<int>(is);
    for(Country& c: countries)
        r.latencyPerCountry[c] = read_from_file<int>(is);
    return is;
}

std::istream& operator>>(std::ifstream& is, Provider& p)
{
    p.name = read_from_file<std::string>(is);
    p.regions.reserve(read_from_file<int>(is));
    for(size_t i = 0; i < p.regions.capacity(); i++)
        p.regions.push_back(read_from_file<Region>(is)); 
    p.regions.shrink_to_fit();   
    return is;
}

std::istream& operator>>(std::ifstream& is, Project& p)
{
    p.basePenality = read_from_file<int>(is);
    p.country = read_from_file<Country>(is);
    for(Service& s : services)
        p.unitsNeededPerServices[s] = read_from_file<int>(is);

    return is;    
}

long long Project::total_units_needed()
{
    long long x = 0;
    for(Service& s : services)
        x += unitsNeededPerServices[s];
    return x;
}