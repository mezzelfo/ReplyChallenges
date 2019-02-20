with open('input','r') as f:
    testCaseCount = int(f.readline())
    for testCase in range(testCaseCount):
        friendsCount = int(f.readline())
        frientsRating = [int(k) for k in f.readline().split(' ')]
        positiveFrientsRating = [k for k in frientsRating if k >= 0]
        print('Case #'+str(testCase+1)+':',sum(positiveFrientsRating))
