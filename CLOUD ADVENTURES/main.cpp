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
    
    providers.reserve(read_from_file<num>(inFile));
    services.reserve(read_from_file<num>(inFile));
    countries.reserve(read_from_file<num>(inFile));
    projects.reserve(read_from_file<num>(inFile));

    for(size_t i = 0; i < services.capacity(); i++)
        services.push_back(read_from_file<Service>(inFile));
    for(size_t i = 0; i < countries.capacity(); i++)
        countries.push_back(read_from_file<Country>(inFile));
    for(size_t i = 0; i < providers.capacity(); i++)
        providers.push_back(read_from_file<Provider>(inFile));
    for(size_t i = 0; i < projects.capacity(); i++)
        projects.push_back(read_from_file<Project>(inFile));

    std::sort(std::begin(projects), std::end(projects), [](Project a, Project b) {return a.total_units_needed() < b.total_units_needed(); });

    std::map<Service,num> availability;
    for(Provider& P : providers)
        for(Region& R: P.regions)
            for(Service& S: services)
                availability[S] += R.availablePackages * R.packagesPerServices[S];

    std::map<int, std::map<Region, num> > used_vec;
    std::vector<std::string> outputs;
    for (Project& P : projects) outputs.push_back("");
    int failed = 0;
    int done = 0;

    for(size_t i = 0; i<projects.size(); i++)
    {
        std::cerr << "Failed: " << failed << " done: " << done << "\r";
        auto& used = used_vec[i];
        Project& I = projects[i];

        if (I.is_possible(availability))
        {
            while (true)
            {
                Region& regionChosen = providers[0].regions[0];
                double minCost = std::numeric_limits<double>::max();
                for (Provider& J : providers)
                {
                    for (Region& K : J.regions)
                    {
                        num num_uses = 0;
                        if (used.count(K))
                            num_uses = used.at(K);
                        if (!K.availablePackages)
                            continue;
                        double pen = RegionPenalty(I, K, num_uses);
                        if (pen < minCost)
                        {
                            regionChosen = K;
                            minCost = pen;
                        }
                    }
                }
                // unsatisfiable project
                if (minCost == std::numeric_limits<double>::max())
                {
                    failed++;
                    break;
                }
                used[regionChosen]++;
                bool is_done = true;
                for (Service& S: services)
                {
                    availability[S] -= regionChosen.packagesPerServices[S];
                    if (I.unitsNeededPerServices[S] <= regionChosen.packagesPerServices[S])
                    {
                        I.unitsNeededPerServices[S] = 0;
                        continue;
                    }
                    I.unitsNeededPerServices[S] -= regionChosen.packagesPerServices[S];
                    is_done = false;
                }
                regionChosen.availablePackages--;
                if (is_done)
                    break;
            }
            done++;
        }
        else
            continue;        
    }

    for(size_t i = 0; i<projects.size(); i++)
    {
        std::cerr << "Failed: " << failed << " done: " << done << "\r";
        auto& used = used_vec[i];
        Project& I = projects[i];

        if (not I.is_possible(availability))
        {
            while (true)
            {
                Region& regionChosen = providers[0].regions[0];
                double minCost = std::numeric_limits<double>::max();
                for (Provider& J : providers)
                {
                    for (Region& K : J.regions)
                    {
                        num num_uses = 0;
                        if (used.count(K))
                            num_uses = used.at(K);
                        if (!K.availablePackages)
                            continue;
                        double pen = RegionPenalty(I, K, num_uses);
                        if (pen < minCost)
                        {
                            regionChosen = K;
                            minCost = pen;
                        }
                    }
                }
                // unsatisfiable project
                if (minCost == std::numeric_limits<double>::max())
                {
                    failed++;
                    break;
                }
                used[regionChosen]++;
                bool is_done = true;

                for (Service& S: services)
                {
                    availability[S] -= regionChosen.packagesPerServices[S];
                    if (I.unitsNeededPerServices[S] <= regionChosen.packagesPerServices[S])
                    {
                        I.unitsNeededPerServices[S] = 0;
                        continue;
                    }
                    I.unitsNeededPerServices[S] -= regionChosen.packagesPerServices[S];
                    is_done = false;
                }
                regionChosen.availablePackages--;
                if (is_done)
                    break;
            }
            done++;
        }
        for (auto v : used)
        {
            outputs[i] +=   v.first.name + " " +
                            std::to_string(v.second) + " ";
        }
    }

    for (const auto& s : outputs)
        std::cout << s << std::endl;
    return 0;
}
