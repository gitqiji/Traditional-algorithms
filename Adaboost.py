from auxi import *

class Adaboost(object):

    def __init__(self,sample,label):
        print 'adaboost'
        self.sample = sample
        self.y = label
        self.m,self.n = np.shape(sample)
        print self.m,self.n
        self.w = np.zeros((1,self.n)) + 1.0 / self.n
        self.item = 200
        self.finalpre = 0
        self.shift = ['gt','lt']
        self.classifier = []
        self.viewem = []

    def getpre(self,arr, thd, flag):
        if flag == 'gt':
            return np.where(arr >= thd, 1, -1)
        else:
            return np.where(arr >= thd, -1, 1)

    def getalpha(self,err):
        temp = (1 - err) / max(err,0.000001)
        alpha = 0.5*np.log2(temp)
        return alpha

    def weakclassify(self,w):
        x = self.sample #* w
        minerr = self.n
        print 'initerr: ',minerr
        for j in range(self.m):
            for k in range(self.n):
                for sml in self.shift:
                    pre = self.getpre(x[j,:],x[j,k],sml)
                    res = np.where(pre == self.y[0], 0, 1)
                    err = np.sum(res*w)
                    if err < minerr:
                        minerr = err
                        dim = [j,k]
                        thed = x[j,k]
                        curres = res
                        curpre = pre
                        cursml = sml

        print 'minerr: ',minerr,'state: ',np.sum(curres)
        print 'w = ',w

        if minerr < 0.5:
            alpha = self.getalpha(minerr)
            self.finalpre = self.finalpre + alpha * curpre
            finalpre = np.sign(self.finalpre)
            #print 'len of finalpre: ',len(finalpre)
            res = np.where(finalpre == self.y[0],0,1)
            self.viewem.append((dim,curpre,finalpre))
            node = (dim,thed,cursml,alpha)
            self.classifier.append(node)
            print self.y[0]
            print finalpre
            if np.sum(res) == 0:
                return -1
            res = np.where(curpre == self.y[0], -1, 1)
            nw = w * np.exp(alpha * res)
            nw = nw/np.sum(nw)
            return nw
        else:
            return -1

    def train(self):
        w = self.w[0]
        for i in range(self.item):
            #print w
            nw = self.weakclassify(w)
            if type(nw) is np.ndarray :
                w = nw
            else:
                break
    def show(self):
        print len(self.classifier)
        #print self.classifier[-1]
    def test(self,pnt):
        pre = 0
        for em in self.classifier:
            dim = em[0]
            thd = em[1]
            sml = em[2]
            alpha = em[3]
            sam = pnt[dim[0]]
            res = self.getpre(sam,thd,sml)
            print np.shape(res)
            pre = pre + alpha * res
        return np.sign(pre)

if __name__ == '__main__':
    auxi = Auxi()
    x,y,line = auxi.nonline()

    cls = Adaboost(x,y)
    cls.train()
    cls.show()
    auxi.plotorigin(x,line)
    decline = []
    for em in cls.viewem:
        #print em
        dim = em[0]
        finalpre = em[2]
        if dim[0] == 0:
            subline = [(x[dim[0],dim[1]],0),(x[dim[0],dim[1]],500)] # (x[0,dim[1]],x[1,dim[1]])
        else:
            subline = [(0,x[dim[0],dim[1]]),(500,x[dim[0],dim[1]])]
        decline.append(subline)
        print subline
        auxi.plotres(x,finalpre,line,decline)
    pnt = np.random.randint(100,400,size=(2,20))
    print 'testpnt:\n',pnt
    pre = cls.test(pnt)
    print 'testval: \n',pre
    auxi.plotres(pnt,pre,line)
