#include <iostream>
#include <fstream>
#include <vector>
#include "tool.hpp"

std::vector<Provider> providers;
std::vector<Service> services;
std::vector<Country> countries;
std::vector<Project> projects;

int main(int argc, char const *argv[])
{
    std::ifstream inFile;

    if (argc != 2)
    {
        std::cerr << "Please use ./a.out <namefileinput>" << std::endl;
        exit(EXIT_FAILURE);
    }
    inFile.open(argv[1]);
    if (!inFile)
    {
        std::cerr << "Unable to open input file" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    providers.reserve(read_from_file<int>(inFile));
    services.reserve(read_from_file<int>(inFile));
    countries.reserve(read_from_file<int>(inFile));
    projects.reserve(read_from_file<int>(inFile));

    for(size_t i = 0; i < services.capacity(); i++)
        services.push_back(read_from_file<Service>(inFile));
    services.shrink_to_fit();
    for(size_t i = 0; i < countries.capacity(); i++)
        countries.push_back(read_from_file<Country>(inFile));
    countries.shrink_to_fit();
    for(size_t i = 0; i < providers.capacity(); i++)
        providers.push_back(read_from_file<Provider>(inFile));
    providers.shrink_to_fit();
    for(size_t i = 0; i < projects.capacity(); i++)
        projects.push_back(read_from_file<Project>(inFile));
    projects.shrink_to_fit();

    std::cout   << "Read " << providers.size() << " providers\n"
                << "Read " << services.size() << " services\n"
                << "Read " << countries.size() << " countries\n"
                << "Read " << projects.size() << " projects" << std::endl;

    std::map<Service,int> totalUnitesNeededPerServices;
    std::map<Service,int> totalUnitesAvailablePerServices;
    for(Project& P : projects)
        for(Service& S: services)
            totalUnitesNeededPerServices[S] += P.unitsNeededPerServices[S];

    for(Provider& P : providers)
        for(Region& R: P.regions)
            for(Service& S: services)
                totalUnitesAvailablePerServices[S] += R.availablePackages * R.packagesPerServices[S];
    return 0;
}
