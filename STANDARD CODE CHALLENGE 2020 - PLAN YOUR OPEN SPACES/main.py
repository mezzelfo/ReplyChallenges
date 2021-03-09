from os import listdir
from bitarray import bitarray

class fileReader():
    def __init__(self, fileobj):
        lines = fileobj.readlines()
        W, H = [int(a) for a in lines[0].split(' ')]
        floormap = lines[1:H+1]
        D = int(lines[H+1])
        developers = [l.strip().split(' ') for l in lines[H+2:H+D+2]]
        managers = [l.strip().split(' ') for l in lines[H+D+3:]]

        allskills = set.union(*[set(d[3:]) for d in developers])
        allskills = {v:i for (i,v) in enumerate(allskills)}
        allcompanies = set([d[0] for d in developers]).union(set([m[0] for m in managers]))
        allcompanies = {v:i for (i,v) in enumerate(allcompanies)}

        print('\t',len(allskills),len(allcompanies))

        self.developers = []
        for d in developers:
            a = bitarray(len(allskills))
            a.setall(False)
            for s in d[3:]:
                a[allskills[s]] = True
            self.developers.append(
                {
                    'company': allcompanies[d[0]],
                    'bonus':d[1],
                    'skillsNum':a.count(),
                    'skills':a#set([allskills[s] for s in d[3:]])
                }
            )
        managers = [{
            'company': allcompanies[m[0]],
            'bonus':m[1],
            'skills':None
        } for m in managers]
        assert len(managers) == int(lines[H+D+2])

        self.floormap = floormap
        #self.developers = developers
        self.managers = managers
        print('parsed')

    def workingPotential(cls, dev1, dev2):
        intersection = (dev1['skills'] & dev2['skills']).count()
        return intersection*(dev1['skillsNum']+dev2['skillsNum']-2*intersection)
        #intersection = len(dev1['skills'].intersection(dev2['skills']))
        #return intersection*(len(dev1['skills'])+len(dev2['skills'])-2*intersection)

    def bonusPotential(cls, rep1, rep2):
        if rep1['company'] == rep2['company']:
            return int(rep1['bonus'])*int(rep2['bonus'])
        return 0

    def totalPotential(cls, rep1, rep2):
        #if rep1['skills'] == None or rep2['skills'] == None:
        #    return cls.bonusPotential(rep1,rep2)
        return cls.bonusPotential(rep1, rep2)+cls.workingPotential(rep1,rep2)

for filename in sorted(listdir()):
    if '.txt' in filename:
        with open(filename, 'r') as inputfile:
            print(filename)
            f = fileReader(inputfile)
            print('\t','Repliers:',len(f.developers)+len(f.managers))

filename = 'd_maelstrom.txt'
with open(filename, 'r') as inputfile:
    f = fileReader(inputfile)
    #totskills = set.union(*[d['skills'] for d in f.developers])
    #print(filename,len(f.developers)+len(f.managers),len(totskills))

D = len(f.developers)
for i in range(0,D):
    for j in range(i+1,D):
        rep1 = f.developers[i]
        rep2 = f.developers[j]
        f.totalPotential(rep1, rep2)

# for rep1 in f.developers:
#     for rep2 in f.managers:
#         f.totalPotential(rep1, rep2)

# for rep1 in f.developers:
#     for rep2 in f.managers:
#         f.totalPotential(rep1, rep2)
print('done')
