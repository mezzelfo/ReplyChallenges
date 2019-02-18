#include "tool.hpp"

extern std::vector<Service> services;
extern std::vector<Country> countries;

std::istream& operator>>(std::ifstream& is, Region& r)
{
    r.name = read_from_file<std::string>(is);
    r.availablePackages = read_from_file<num>(is);
    r.packageUnitCost = read_from_file<float>(is);
    for(Service& s : services)
        r.packagesPerServices[s] = read_from_file<num>(is);
    for(Country& c: countries)
        r.latencyPerCountry[c] = read_from_file<num>(is);
    return is;
}

std::istream& operator>>(std::ifstream& is, Provider& p)
{
    p.name = read_from_file<std::string>(is);
    p.regions.reserve(read_from_file<num>(is));
    for(size_t i = 0; i < p.regions.capacity(); i++)
    {
        p.regions.push_back(read_from_file<Region>(is));
        p.regions[i].name = p.name + "::" + p.regions[i].name;
    }
    return is;
}

std::istream& operator>>(std::ifstream& is, Project& p)
{
    p.basePenality = read_from_file<num>(is);
    p.country = read_from_file<Country>(is);
    for(Service& s : services)
        p.unitsNeededPerServices[s] = read_from_file<num>(is);

    return is;    
}

num Project::total_units_needed()
{
    num x = 0;
    for(Service& s : services)
        x += unitsNeededPerServices[s];
    return x;
}

bool Project::is_possible(std::map<Service,num> tot)
{
    for(Service& S: services)
        if(tot[S] < unitsNeededPerServices[S])
            return false;
    return true;
}

double RegionPenalty(Project& P, Region& R, num num_uses)
{
    num total_requirements = 0;
    for (auto& S : services)
        total_requirements += P.unitsNeededPerServices[S];
    if (!total_requirements)
    {
        return 10000 * num_uses;
        throw std::runtime_error("Nope"+std::to_string(total_requirements));
    }
    num future_requirements = 0;
    num overassign = 0;
    for (auto& S : services)
    {
        if (P.unitsNeededPerServices[S] > R.packagesPerServices[S])
            future_requirements += P.unitsNeededPerServices[S] - R.packagesPerServices[S];
        else
            overassign += R.packagesPerServices[S] - P.unitsNeededPerServices[S];
    }
    double perc = 1 - 0.3 * future_requirements / total_requirements;
    if (perc < 1e-6)
        return std::numeric_limits<double>::max();
    return R.packageUnitCost * R.latencyPerCountry[P.country] / perc + 10000 * num_uses + overassign * 10;
}