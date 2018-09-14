from auxi import *
class Dectree(object):
    def __init__(self,sample,label):
        print 'decision tree'
        self.shift = ['gt','lt']
        self.feaname = ['x','y']
        self.sample = sample
        self.y = label
        self.m,self.n = np.shape(sample)

    def getentropy(self,prob):
        eny = -np.sum(prob * np.log2(prob))
        return eny

    def getcondeny0(self,pbd,eny):
        print 'get condition entropy'
        condeny = np.sum(pbd * eny)
        return condeny

    def getpre(self,arr,thd,flag):
        if flag == 'gt':
            return np.where(arr >= thd, 1, 0)
        else:
            return np.where(arr >= thd, 0, 1)

    def getcondeny(self, y, pre):
        lb = y[0][pre == 1]
        rb = y[0][pre == 0]
        pbd = np.zeros((1,2))
        pbd[0][0] = len(lb) * 1.0 / len(y[0]) #self.n
        pbd[0][1] = len(rb) * 1.0 / len(y[0]) #self.n
        probl = self.getprob(lb)
        probr = self.getprob(rb)
        condenyl = pbd[0][0] * self.getentropy(probl)
        condenyr = pbd[0][1] * self.getentropy(probr)
        condeny = condenyl + condenyr
        return condeny

    def getinfogain(self,dset,condeny):
        prob = self.getprob(dset)
        eny = self.getentropy(prob)
        infogain = eny - condeny
        return infogain

    def cacnum(self,dset):
        dct = {};tl = dset.tolist()
        for em in set(tl):
            dct.update({em:tl.count(em)})
        return dct

    def getprob(self,y):
        idx = 0
        dct = self.cacnum(y)
        prob = np.zeros((1,len(dct)))
        for tag in dct:
            prob[0][idx] = dct[tag] * 1.0 / len(y)
            idx = idx + 1
        prob[0][prob[0] == 0] = 0.0001
        return prob

    def selectfea(self, x, y):
        maxgain = 0
        m, n = np.shape(x)
        for j in range(m):
            for i in range(n):
                for sml in self.shift:
                    pre = self.getpre(x[j,:],x[j,i],sml)
                    condeny = self.getcondeny(y, pre)
                    infogain = self.getinfogain(y[0],condeny)
                    if maxgain < infogain:
                        maxgain = infogain
                        print maxgain
                        dim = [j,i]
                        thd = x[j,i]
                        cursml = sml
                        curpre = pre
        return dim, thd, cursml, curpre

    def splitdata(self, y, pre):
        # pre only represent mark that distinct two sub sets, 1 is left subset and 0 is right subset
        lb = y[0][pre == 1]
        rb = y[0][pre == 0]
        print lb,'\n',rb

        if self.ispure(lb):
            mark = 1; vau = lb[0]
        elif self.ispure(rb):
            mark = 0; vau = rb[0]
        else:
            lvau, lerr = self.majority(lb)
            rvau, rerr = self.majority(rb)
            mark = 1; vau = lvau
            if lerr > rerr:
                mark = 0; vau = rvau

        return mark, vau

    def ispure(self, dset):
        flag = False
        ds = np.sum(dset)
        if ds == dset[0]*len(dset):
            flag = True
        return flag

    def majority(self,dset):
        dct = self.cacnum(dset)
        sz = np.sum(dct.values())
        mx = max(dct,key=dct.get)
        err = 1 - dct[mx] * 1.0 / sz
        return mx, err

    def buildtree(self, x, y):
        # {(x,sml,thd):1,(x,sml,thd):{(y,sml,thd):1,(y,sml,thd):{}}

        tree = {}
        dim, thd, sml, curpre = self.selectfea(x, y)

        mark, vau = self.splitdata(y, curpre)
        subdata = np.delete(x,np.where(curpre == mark),1)
        suby = np.delete(y,np.where(curpre == mark),1)
        # This condition select the right part of the thd, so the sml should be <=
        if sml == 'gt' and mark == 0:
            sml = 'lt'
        # This condition select the left part of the thd, so the sml should be >=
        elif sml == 'lt' and mark == 0:
            sml = 'gt'
        node = (self.feaname[dim[0]],sml,thd)
        tree[node] = vau
        print tree,type(vau)

        # contruct another branch
        sml = self.shift[abs(self.shift.index(sml) - 1)]
        node = (self.feaname[dim[0]],sml,thd)

        if self.ispure(suby[0]):
            print type(suby[0][0])
            tree[node] = suby[0][0]
            return tree
        else:
            #print node
            tree[node] = self.buildtree(subdata, suby)

        return tree

    def getdecline(self, tree):
        decline = []
        for k in tree:
            if type(tree[k]) is not dict:
                if k[0] == 'x':
                    subline = [(k[2],0),(k[2],500)]
                else:
                    subline = [(0,k[2]),(500,k[2])]
                decline.append(subline)
        for k in tree:
            if type(tree[k]) is dict:
                decline.extend(self.getdecline(tree[k]))
        return decline

    def test(self, x, tree):
        #nl = []
        sm = np.random.randint(1,2,size=(3,0))
        for k in tree:
            if type(tree[k]) is not dict:
                # ensure dec dim
                dim = [0 if k[0] == 'x' else 1]
                # ensure dec sml
                inx = [np.where(x[dim[0]] >= k[2]) if k[1] == 'gt' else np.where(x[dim[0]] <= k[2])]
                #print inx[0][0]
                t = np.vstack((x[0,inx[0][0]],x[1,inx[0][0]]))
                y = np.random.randint(1,2,size=(1,len(inx[0][0])))
                y[0] = tree[k]
                t = np.vstack((t,y))
                sm = np.hstack((sm,t))
                #
                x = np.delete(x,inx[0][0],1)

        for k in tree:
            if type(tree[k]) is dict:
                #nl.extend(self.test(x,tree[k]))
                sm = np.hstack((sm,self.test(x,tree[k])))

        return sm
# ID3 will preference the attribute with more values, because it usually has larger entropy and larger info gain
# C4.5 will divides splitting information and select the attribute with greatest rate of info gain
# For splitting information, the prob should be the proportion of split value for every attribute,
# then calculate entropy using the formula
if __name__ == '__main__':
    auxi = Auxi()
    x,y,line = auxi.svmdata(0)
    dectree = Dectree(x,y)
    tree = dectree.buildtree(x,y)
    print tree
    decline = dectree.getdecline(tree)
    #x,y,line = auxi.svmdata(0)
    #x = np.random.randint(10,300,size=(2,50))
    sm = dectree.test(x,tree)
    print decline,'\n',sm

    tx = np.delete(sm,2,0)
    ty = np.delete(sm,[0,1],0)
    #print ty
    auxi.plotres(tx,ty[0],line,decline)




