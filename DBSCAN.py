from auxi import *
class Dbscan(object):
    def __init__(self,data):
        self.x = data
        self.m,self.n = np.shape(data)
        self.distm = np.zeros((self.n,self.n))
        self.k = 0
        self.eps = 10
        self.minpts = 4
        self.label = np.zeros((1,self.n))
        self.setcorept = []
        self.ptnn = {}


    def getcorept(self):
        x = self.x

        for i in range(self.n):
            cnt = 0
            subset = []
            for j in range(self.n):
                t = x[:,i] - x[:,j]
                t = t.reshape((2,1))
                #print np.shape(t)
                #assert np.shape(t) == (2,1)
                self.distm[i][j] = np.sqrt(np.square(t[0][0]) + np.square(t[1][0]))
                if self.distm[i][j] <= self.eps:
                    cnt += 1
                    subset.append(j)
            if cnt >= self.minpts:
                self.setcorept.append(i)
                self.ptnn[i] = subset
                #print cnt
        print 'len of setcore: ',len(self.setcorept)

    def genonecu(self):
        corenum = len(self.setcorept)
        idx = np.random.randint(0,corenum)
        coreque = []
        coreque.append(self.setcorept.pop(idx))

        while len(coreque) != 0:
            corept = coreque.pop(0)
            # get the points in the neighborhood
            subset = self.ptnn[corept]
            for pt in subset:
                # ensure core point and not visited
                if pt in self.setcorept:
                    coreque.append(pt)
                    self.setcorept.remove(pt)
                self.label[0][pt] = self.k

    def loopgencu(self):
        while len(self.setcorept) != 0:
            self.k += 1
            print 'cu num: ',self.k
            self.genonecu()

    def start(self):
        self.getcorept()
        self.loopgencu()


if __name__ == '__main__':
    auxi = Auxi()
    #x,y,line = auxi.gaussiondata(1)
    x,y,line = auxi.sindata()
    dbscan = Dbscan(x)
    dbscan.start()
    print dbscan.label
    auxi.plotres(x,dbscan.label[0],line)



