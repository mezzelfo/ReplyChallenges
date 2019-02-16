#include <iostream>
#include <fstream>
#include <vector>
#include "tool.hpp"
using namespace std;

int main(int argc, char const *argv[])
{
    ifstream inFile;
    if (argc != 2)
    {
        cerr << "Please use ./a.out <namefileinput>" << endl;
        exit(EXIT_FAILURE);
    }
    inFile.open(argv[1]);
    if (!inFile)
    {
        cerr << "Unable to open input file" << endl;
        exit(EXIT_FAILURE);
    }

    vector<Provider> providers;
    vector<Service> services;
    vector<Country> countries;
    vector<Project> projects;

    return 0;
}
