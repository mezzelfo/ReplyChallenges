with open('input.txt','r') as f:
	testCaseCount = int(f.readline())
	for testCase in range(testCaseCount):
		stringLengths = [int(k) for k in f.readline().split(' ')]
		viruslLength = int(f.readline())
		fileStrings = [f.readline().strip() for _ in range(4)]
		shortest = min(fileStrings, key = lambda s: len(s))
		for offset in range(len(shortest)-viruslLength+1):
			virus = shortest[offset:offset+viruslLength]
			pos = [s.find(virus) for s in fileStrings]
			if all([p != -1 for p in pos]):
				print('Case #'+str(testCase+1)+':',pos[0],pos[1],pos[2],pos[3])
				break
