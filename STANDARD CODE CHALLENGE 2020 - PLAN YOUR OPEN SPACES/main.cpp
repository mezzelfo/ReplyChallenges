#include <bits/stdc++.h>
using namespace std;

map<string, int> skillMap;
map<string, int> companyMap;

struct replier
{
    int company;
    int bonus;
    vector<int> skills;
};

vector<replier> developers;
vector<replier> managers;

int totpotential(replier& r1, replier& r2)
{
    int s1 = r1.skills.size();
    int s2 = r2.skills.size();
    int bonuspotential = r1.bonus * r2.bonus * (r1.company == r2.company);
    if ((s1 == 0) or (s2 == 0))
        return bonuspotential;
    int common_skills = 0;
    int index1 = 0, index2 = 0;
    while (1)
    {
        if (r1.skills[index1] == r2.skills[index2])
        {
            common_skills++;
            index1++;
            index2++;
        }
        else if (r1.skills[index1] >= r2.skills[index2])
            index1++;
        else
            index2++;

        if ((index1 >= s1) || (index2 >= s2))
            break;
    }
    int distinct_skills = s1 + s2 - 2 * common_skills;
    return common_skills * distinct_skills + bonuspotential;
}

int main(int argc, char const *argv[])
{
    ifstream infile("f_glitch.txt");
    int W, H, D, M;
    string company, skill;
    int bonus, numskills;
    infile >> W >> H >> ws;

    for (int i = 0; i < H; i++)
    {
        string line;
        getline(infile, line);
    }
    infile >> D >> ws;
    for (int i = 0; i < D; i++)
    {
        replier dev;
        infile >> company >> bonus >> numskills;
        if (companyMap.find(company) == companyMap.end())
            companyMap.emplace(company, companyMap.size());
        dev.company = companyMap[company];
        dev.bonus = bonus;
        for (int j = 0; j < numskills; j++)
        {
            infile >> skill;
            if (skillMap.find(skill) == skillMap.end())
                skillMap.emplace(skill, skillMap.size());
            dev.skills.push_back(skillMap[skill]);
        }
        dev.skills.shrink_to_fit();
        sort(dev.skills.begin(), dev.skills.end());
        developers.push_back(dev);
        infile >> ws;
    }
    infile >> M >> ws;
    for (int i = 0; i < M; i++)
    {
        replier man;
        infile >> company >> bonus >> ws;
        if (companyMap.find(company) == companyMap.end())
            companyMap.emplace(company, companyMap.size());
        man.company = companyMap[company];
        man.bonus = bonus;
        managers.push_back(man);
    }
    cout << companyMap.size() << endl
         << skillMap.size() << endl;

    developers.shrink_to_fit();
    managers.shrink_to_fit();
    cout << developers.size() << endl;

    ofstream outfile("out_f_glitch.txt");
    for(int i=0; i<developers.size(); i++)
    {
        outfile << i << ":";
        for(int j=i+1; j<developers.size(); j++)
        {
            int pot = totpotential(developers[i],developers[j]);
            if (pot > 0)
            {
                outfile << "(" << j << "," << pot << "), ";
            }
        }
        if (i%500 == 0) cout << i << endl;
    }
    outfile.flush();
    return 0;
}
