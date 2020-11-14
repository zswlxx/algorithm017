学习笔记

字典树 Trie

树 二叉树 多叉树 深度优先DFS 广度优先BFS 二叉搜索树（子树关系，非儿子父亲关系）
二叉搜索树（1.左子树所有节点都要小于根节点，右子树所有节点都要大于根节点，且任何子树都满足这样的特性；2.是个升序序列（前中后序遍历））
二叉搜索树查询效率高

字典树，即Trie树，又称单词查找树或键树，是一种树形结构。典型的英语是用于统计和排序大量的字符串（但不仅限于字符串）,所以经常被搜索引擎用于文本词频统计。
优点是最大限度地减少无谓的字符串比较，查询效率比哈希表高。

Trie不是二叉树，是多叉树，基本性质：1. 节点本身不存完整单词；2.从根结点到某一结点，路径上经过的字符连接起来，为该结点对应的字符串；3.每个结点的所有子节点路径代表的字符都不相同。
核心思想：用空间换时间，利用字符串的公共前缀来降低查询时间的开销以达到提高效率的目的。

# 实现trie 
class Trie(object):

    def __init__(self):
        self.root = {}
        self.end_of_word = "#"

    def insert(self, word):
        node = self.root
        for char in word:
            node = node.setdefault(char, {})
        node[self.end_of_word] = self.end_of_word

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return self.end_of_word in node

    def startsWith(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True

初级搜索的优化方式：不重复（Fibonacci）、剪枝（生成括号问题）、搜索方向：DFS、BFS
高级优化：双向搜索、启发式搜索
做搜索问题时脑子里面要有一棵树（零钱置换状态树）

双向BFS 单词接龙 多练

模板一定要滚瓜烂熟默写下来（抄10遍）

List O(n)  HashSet O(1)

启发式搜索
估价函数
启发式函数：h(n)，它用来评价哪些结点最有希望的是一个我们要找的结点，h(n)会返回一个非负实数，也可以认为是从结点n的目标结点路径的估计成本。
启发式函数是一个告知搜索方向的方法。它提供了一种明智的方法来猜测哪个邻居结点会导向一个目标。
A * 模板
def AsterSearch(graph, start, end):

    pq = collections.priority_queue() #优先级 -> 估价函数
    pq.append([start])
    visited.add(start)

    while pq:
        node = pq.pop()
        visited.add(node)

        process(node)
        nodes = generate_related_nodes(node)
    unvisited = [node for node in nodes if node not in visited]
        pq.push(unvisited)

二叉树遍历
前序（Pre-order）: 根-左-右
中序（In-order）: 左-根-右
后序（Post-order）: 左-右-根

保证性能的关键：
1. 保证二维维度！ -> 左右子树结点平衡（recursively）
2. Balanced

AVL树
平衡二叉树
Balance Factor(平衡因子)：
是它的左子树的高度减去它的右子树的高度（有时相反）
每个节点存balance factor = {-1, 0, 1}
通过旋转操作来进行平衡（四种）左旋、右旋、左右旋、右左旋

Red-black Tree
红黑树是一种近似平衡的二叉搜索树，它能确保任何一个结点的左右子树的高度差小于两倍。具体来说，红黑树是满足如下条件的二叉搜索树：
1. 每个结点要么是红色，要么是黑色
2.根节点是黑色
3.每个叶结点（NIL结点空结点）是黑色的。
4.不能有相邻接的两个红色结点
5.从任一结点到其每个叶子的所有路径都包含形同数目的黑色结点。


