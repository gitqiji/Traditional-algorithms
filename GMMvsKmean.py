from auxi import *
import pdb

class GMM(object):
    def __init__(self,sample,z):
        print 'This is GMM algorithm'
        self.x = sample
        self.m,self.n = np.shape(sample)
        self.y = np.random.randint(0,3,size=(1,self.n))
        #alpha = np.random.randint(1,8,size=(1,3))
        #self.alpha = alpha * 1.0 / np.sum(alpha,axis=1)
        self.alpha = np.zeros((1,z)) + 1.0 / z
        c1 = np.random.randint(0,50)
        c2 = np.random.randint(50,100)
        c3 = np.random.randint(100,150)
        c = [c1,c2,c3]
        self.z = z
        self.miu = sample[:,np.random.randint(0,150,size=(1,self.z))[0][:]]
        print type(self.miu)
        print self.miu

    def initsigma(self):
        self.sigma = np.zeros((self.z,2,2))
        for i in range(self.z):
            t = self.x - np.reshape(self.miu[:,i],(2,1))
            self.sigma[i] = np.dot(t,t.T) / self.n

    def getpzx(self,sam, miu, sigma):
        sigma_1 = np.array(np.matrix(sigma).I)
        t = sam - miu
        #print t,np.shape(t)

        up = np.exp(-np.dot(np.dot(t.T,sigma_1),t))
        down = np.sqrt(np.linalg.det(sigma))
        #down = np.sqrt(pow(2 * np.pi,self.m) * np.linalg.det(sigma))
        #print up,'\n',down
        res = up / down

        return res

    def getexpect(self,sam):
        pzx = np.zeros((1,self.z))
        for i in range(self.z):
            pzx[0][i] = self.getpzx(sam,self.miu[:,i],self.sigma[i]) * self.alpha[0][i]

        return pzx

    def oneitem(self):
        miuup = np.zeros((2,self.z))
        miudown = np.zeros((1,self.z))
        sigmaup = np.zeros((self.z,2,2))
        Qzm = np.zeros((self.n,self.z))

        for i in range(self.n):
            sam = self.x[:,i]
            pzx = self.getexpect(sam)
            Qz = pzx / np.sum(pzx,axis=1)
            sam = np.reshape(sam,(2,1))
            miuup += np.dot(sam,Qz)
            miudown += Qz
            Qzm[i] = Qz
        #print Pzm
        #pdb.set_trace()

        # M step
        for j in range(self.z):
            #print '0 -- ',miuup
            miuup[:,j] = miuup[:,j] / miudown[0][j]
            #print '1 -- ',miuup
            #pdb.set_trace()
            for i in range(self.n):
                sam = self.x[:,i]
                sam = np.reshape(sam,(2,1))
                sigmaup[j] += np.dot((sam - miuup[:,j]),(sam - miuup[:,j]).T) * Qzm[i][j]
            sigmaup[j] = sigmaup[j] / miudown[0][j]

        self.alpha = miudown / self.n
        print self.alpha
        #print miuup

        return miuup, sigmaup

    def train(self):
        self.initsigma()
        item = 1000
        for i in range(item):
            miu, sigma = self.oneitem()
            chv = np.sum(np.abs(self.miu - miu),axis=0)
            #print chv.shape
            chv = np.sum(chv,axis=0)
            if chv < 0.00000001:
                break
            else:
                print 'item: ',i
                self.miu = miu
                self.sigma = sigma

    def test(self):
        y = np.zeros((1,self.n))
        for i in range(self.n):
            pzx = self.getexpect(self.x[:,i])
            y[0][i] = np.argmax(pzx[0])
            print pzx
        return y

class Kmean(GMM):
    def oneitem(self):
        y = np.random.randint(0,1,size=(1,self.n))
        samsum = np.zeros((2,self.z))
        for i in range(self.n):
            sam = self.x[:,i]
            t = self.miu - np.reshape(sam,(2,1))
            t = np.square(t)
            t = np.sum(t,axis=0)
            y[0][i] = np.argmin(t)

            samsum[:,y[0][i]] += sam
        print self.miu,'\n',y

        for i in range(self.z):
            if i in y[0]:
                t = y[0][y[0] == i]
                samsum[:,i] = samsum[:,i] * 1.0 / len(t)
        return samsum,y

    def train(self):
        item = 100
        for i in range(item):
            miu,y = self.oneitem()
            chv = np.sum(np.abs(self.miu - miu))
            if chv < 0.00000001:
                return y
            else:
                self.miu = miu
                print 'item: ',i

if __name__ == '__main__':
    auxi = Auxi()
    x,y,line = auxi.gaussiondata(3)
    z = len(list(set(y[0])))

    gmm = GMM(x,z)
    #
    kmn = Kmean(x,z)
    py = kmn.train()
    auxi.name = kmn.__class__.__name__
    auxi.plotres(x,py[0],line)

    gmm.miu = kmn.miu
    print gmm.miu

    #pdb.set_trace()
    gmm.train()
    py = gmm.test()
    auxi.name = gmm.__class__.__name__
    auxi.plotres(x,py[0],line)



