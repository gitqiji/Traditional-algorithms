from adaboost import *


class Gbdt(Adaboost):
    def __init__(self,sample,label):
        super(Gbdt,self).__init__(sample,label)
        self.x = sample
        self.y = label
        self.tree = []
        self.cury = label
        self.diry = label
        self.m,self.n = np.shape(sample)
        yave = np.sum(label) * 1.0 / self.n
        self.finalpre = np.zeros((1,self.n)) + yave
        print 'GBDT'

    def getpre(self, x, thd):
        return np.where(x >= thd, 1, 0)

    def getloss(self,ys,pre):
        Ls = np.log(1 + np.exp(ys*pre*(-2)))
        return Ls

    def cacvar(self, y, pre):
        lb = y[0][pre == 1]
        rb = y[0][pre == 0]
        if len(lb) == 0:
            lave = 0
        else:
            lave = np.sum(lb) * 1.0 / len(lb)
        if len(rb) == 0:
            rave = 0
        else:
            rave = np.sum(rb) * 1.0 / len(rb)

        variance = np.sum(pow((lb - lave), 2)) + np.sum(pow((rb - rave), 2))

        return variance, lave, rave

    def dectree(self, x, y):
        minvar = self.n
        for j in range(self.m):
            for i in range(self.n):
                pre = self.getpre(x[j,:],x[j,i])
                varc,lave,rave = self.cacvar(y,pre)

                if varc < minvar:
                    curpre = pre
                    curlave = lave
                    currave = rave

                    minvar = varc
                    dim = [j,i]
                    thd = x[j,i]
                    print varc

        node = [dim,thd]
        return node, curpre, curlave, currave

    def buildtree(self):
        ny = self.y - self.finalpre
        x = self.x
        cpy = np.zeros_like(self.y)
        item = 120

        for i in range(item):
            print 'item: ',i
            print np.round(self.finalpre[0])
            node, curpre, curlave, currave = self.dectree(x, ny)

            cpy[0][curpre == 1] = curlave; cpy[0][curpre == 0] = currave
            self.finalpre[0] = self.finalpre[0] + cpy[0]
            rsdl = self.y - self.finalpre
            self.tree.append(node)
            ny = rsdl

'''
2 3 5 6 initial predict value 4 4 4 4
initial residual -2 -1 1 2
min variance to get new predict -1.5 -1.5 1.5 1.5
update final predict 2.5 2.5 5.5 5.5

update residual -0.5 0.5 -0.5 0.5
current predict with min variance -0.5 0.5 -0.5 0.5
update final predict 2 3 5 6
'''

if __name__ == '__main__':
    auxi = Auxi()
    x,y,line = auxi.nonline(0)
    gb = Gbdt(x,y)
    gb.buildtree()
