#include "tool.hpp"

std::vector<Provider> providers;
std::vector<Service> services;
std::vector<Country> countries;
std::vector<Project> projects;

int main(int argc, char const *argv[])
{
    std::ifstream inFile;

    if (argc != 2)
        throw std::runtime_error("Please use ./a.out <namefileinput>");
    inFile.open(argv[1]);
    if (!inFile)
        throw std::runtime_error("Unable to open input file");
    
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

    std::sort(std::begin(projects), std::end(projects), [](Project a, Project b) {return a.total_units_needed() < b.total_units_needed(); });

    

    return 0;
}
