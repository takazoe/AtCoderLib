import io, sys

_INPUT = """\
2
1 2
1 2 3
aaaz
"""
sys.stdin = io.StringIO(_INPUT)



#=======================
#     ライブラリ
#=======================


def II(): return int(input()) # a = 1
def MII(): return map(int, input().split()) # a,b,c = 1 2 3
def LMII(): return list(map(int, input().split())) # A = [1,2,3]
def LMIIS(n): return [LMII() for _ in range(n)]
def PLUS(x,y): return [[x+1,y],[x,y+1],[x-1,y],[x,y-1]]
def YES(): print('Yes'), exit()
def NO(): print('No'), exit()

# settings
import sys
sys.setrecursionlimit(10**8) # 再起回数の設定


from collections import defaultdict
class UnionFind():
    """
    Union Find木クラス
    参考 https://note.nkmk.me/python-union-find/

    Attributes
    --------------------
    n : int
        要素数
    root : list
        木の要素数
        0未満であればそのノードが根であり、添字の値が要素数
    rank : list
        木の深さ
    """

    def __init__(self, n):
        
        """
        parent[i] : i番目の親 , 根の場合は負数(絶対値はその根の持つ要素数)
        :param n: 要素数
        """
        self.n = n                  # 要素数
        self.parents = [-1] * n
 
    def find(self, x):
        """
        要素xが属するグループの根の要素番号を返す
        :return:int xの根の要素番号
        """
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]
 
    def union(self, x, y):
        """
        要素xが属するグループと要素yが属するグループとを併合する
        """
        x = self.find(x)        # 要素xの根
        y = self.find(y)        # 要素yの根
        if x == y:
            return
 
        # Union by Size
        # サイズが大きいほうに小さいほうを加える
        if self.parents[x] > self.parents[y]:   # サイズが負数で格納されているため、この場合yのほうが大きい場合
            x, y = y, x     # xとy入れ替え
 
        # xが大きい状態
        self.parents[x] += self.parents[y]
        self.parents[y] = x
 
    def size(self, x):
        """
        要素xが属するグループのサイズ（要素数）を返す
        """
        return -self.parents[self.find(x)]  # 負数で管理している為　-付ける
 
    def same(self, x, y):
        """
        要素x, yが同じグループに属するかどうかを返す
        :return:True→同じグループ
        """
        return self.find(x) == self.find(y)
 
    def members(self, x):
        """
        要素xが属するグループに属する要素をリストで返す
        O(NlogN)
        """
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]
 
    def roots(self):
        """
        すべての根の要素をリストで返す
        O(n)
        """
        return [i for i, x in enumerate(self.parents) if x < 0]
 
    def group_count(self):
        """
        グループの数を返す　0が使われない頂点の場合0が孤立した根になっていることに注意！！！！
        O(n)
        """
        return len(self.roots())
 
    def all_group_members(self):
        """
        {ルート要素: [そのグループに含まれる要素のリスト], ...}のdefaultdictを返す
        O(NlogN)
        """
        group_members = defaultdict(list)
        for member in range(self.n):
            group_members[self.find(member)].append(member)
        return group_members
 
    def __str__(self):
        """
        print()での表示用
        ルート要素: [そのグループに含まれる要素のリスト]を文字列で返す
        O(NlogN)
        """
        return '\n'.join(f'{r}: {m}' for r, m in self.all_group_members().items())
 
# 上のクラスを文字列やタプルなどでも使えるように
# 上のと比べてInitのオーダーが辞書を作成する為が3倍　n=4,000,000付近で2s超えるので注意！！！！
class UnionFindLabel(UnionFind):
    def __init__(self, labels):
        """
        parent[i] : i番目の親の要素番号 , 根の場合は負数(絶対値はその根の持つ要素数)
        d[labelsの要素]: 割り当てられた要素番号
        d_inv[割り当てられた要素番号]: labelsの要素
        :param labels: リスト型(一応文字列でも) 要素は辞書のキーとして使えるオブジェクト str/int/tupleなど listはNG
        """
        assert len(labels) == len(set(labels))      # リスト(labels)の中に同じ要素があった場合エラー
 
        self.n = len(labels)                                # 要素数
        self.parents = [-1] * self.n
        self.d = {x: i for i, x in enumerate(labels)}
        self.d_inv = {i: x for i, x in enumerate(labels)}
 
    def find_label(self, x):
        """
        :param x: labelsの要素
        :return: labels要素xが属するグループの根のlabelsの要素を返す
        find が様々な場所で使っているのでオーバーライドできない為別名で定義
        """
        return self.d_inv[super().find(self.d[x])]
 
    def union(self, x, y):
        """
        labels要素xが属するグループとlabels要素yが属するグループとを併合する
        """
        super().union(self.d[x], self.d[y])
 
    def size(self, x):
        """
        labels要素xが属するグループのサイズ（要素数）を返す
        :param x:
        :return:
        """
        return super().size(self.d[x])
 
    def same(self, x, y):
        """
        labels要素x, labels要素yが同じグループに属するかどうかを返す
        :return:True→同じグループ
        """
        return super().same(self.d[x], self.d[y])
 
    def members(self, x):
        """
        labels要素xが属するグループに属するlabels要素をリストで返す
        O(NlogN)
        """
        root = self.find(self.d[x])
        return [self.d_inv[i] for i in range(self.n) if self.find(i) == root]
 
    def roots(self):
        """
        すべての根のlabels要素をリストで返す
        O(n)
        """
        return [self.d_inv[i] for i, x in enumerate(self.parents) if x < 0]
 
    def all_group_members(self):
        """
        {labelsルート要素: [そのグループに含まれるlabels要素のリスト], ...}のdefaultdictを返す
        O(NlogN)
        """
        group_members = defaultdict(list)
        for member in range(self.n):
            group_members[self.d_inv[self.find(member)]].append(self.d_inv[member])
        return group_members


class MultiSet:
    """
    BIT一本で実装されたMultiset
    参考 : https://qiita.com/toast-uz/items/a63f2d57ec7321186f12
    """

    # n: サイズ、compress: 座圧対象list-likeを指定(nは無効)
    # multi: マルチセットか通常のOrderedSetか
    def __init__(self, n=0, *, compress=[], multi=True):
        self.multi = multi
        self.inv_compress = sorted(set(compress)) if len(compress) > 0 else [i for i in range(n)]
        self.compress = {k: v for v, k in enumerate(self.inv_compress)}
        self.counter_all = 0
        self.counter = [0] * len(self.inv_compress)
        self.bit = BIT(len(self.inv_compress))

    def add(self, x, n=1):     # O(log n)
        if not self.multi and n != 1: raise KeyError(n)
        x = self.compress[x]
        count = self.counter[x]
        if count == 0 or self.multi:  # multiなら複数カウントできる
            self.bit.add(x + 1, n)
            self.counter_all += n
            self.counter[x] += n

    def remove(self, x, n=1):  # O(log n)
        if not self.multi and n != 1: raise KeyError(n)
        x = self.compress[x]
        count = self.bit.get(x + 1)
        if count < n: raise KeyError(x)
        self.bit.add(x + 1, -n)
        self.counter_all -= n
        self.counter[x] -= n

    def __repr__(self):
        return f'MultiSet {{{(", ".join(map(str, list(self))))}}}'

    def __len__(self):         # oprator len: O(1)
        return self.counter_all

    def count(self, x):        # O(1)
        return self.counter[self.compress[x]]

    def __getitem__(self, i):  # operator []: O(log n)
        if i < 0: i += len(self)
        x = self.bit.lower_bound(i + 1)
        if x > self.bit.n: raise IndexError('list index out of range')
        return self.inv_compress[x - 1]

    def __contains__(self, x): # operator in: O(1)
        return self.bit.get(self.compress.get(x, self.bit.n) + 1, 0) > 0

    def bisect_left(self, x):  # O(log n)
        return self.bit.sum(bisect.bisect_left(self.inv_compress, x))

    def bisect_right(self, x): # O(log n)
        return self.bit.sum(bisect.bisect_right(self.inv_compress, x))

    @property
    def max(self):              # O(1)
        return self[-1]

    @property
    def min(self):              # O(1)
        return self[0]


class BIT:
    """
    BIT
    参考 : https://qiita.com/toast-uz/items/bf6f142bace86c525532#13-bit%E3%81%A8%E8%BB%A2%E5%80%92%E6%95%B0
    """

    def __init__(self, n):
        self.n = len(n) if isinstance(n, list) else n
        self.size = 1 << (self.n - 1).bit_length()
        if isinstance(n, list):  # nは1-indexedなリスト
            a = [0]
            for p in n: a.append(p + a[-1])
            a += [a[-1]] * (self.size - self.n)
            self.d = [a[p] - a[p - (p & -p)] for p in range(self.size + 1)]
        else:                    # nは大きさ
            self.d = [0] * (self.size + 1)

    def __repr__(self):
        p = self.size
        res = []
        while p > 0:
            res2 = []
            for r in range(p, self.size + 1, p * 2):
                l = r - (r & -r) + 1
                res2.append(f'[{l}, {r}]:{self.d[r]}')
            res.append(' '.join(res2))
            p >>= 1
        res.append(f'{[self.sum(p + 1) - self.sum(p) for p in range(self.size)]}')
        return '\n'.join(res)

    def add(self, p, x):  # O(log(n)), 点pにxを加算
        assert p > 0
        while p <= self.size:
            self.d[p] += x
            p += p & -p

    def get(self, p, default=None):     # O(log(n))
        assert p > 0
        return self.sum(p) - self.sum(p - 1) if 1 <= p <= self.n or default is None else default

    def sum(self, p):     # O(log(n)), 閉区間[1, p]の累積和
        assert p >= 0
        res = 0
        while p > 0:
            res += self.d[p]
            p -= p & -p
        return res

    def lower_bound(self, x):   # O(log(n)), x <= 閉区間[1, p]の累積和 となる最小のp
        if x <= 0: return 0
        p, r = 0, self.size
        while r > 0:
            if p + r <= self.n and self.d[p + r] < x:
                x -= self.d[p + r]
                p += r
            r >>= 1
        return p + 1
 


#=======================
#     コード
#=======================
import bisect

# input
a = int(input())
a,b = map(int, input().split())
A = list(map(int, input().split()))

# main
ans = A

# output
print(ans)