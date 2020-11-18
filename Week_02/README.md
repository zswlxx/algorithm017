学习笔记

哈希表(Hash table)，根据关键码值(key value)而直接进行访问的数据结构，通过把关键码值映射到表中的一个位置来访问记录，来加快查询速度。使用的映射参数叫做散列函数(Hash Function)，存放记录的数组叫做哈希表或散列表。常用于用户信息存储，缓存，键值对存储。如果函数选的不好可能会出现哈希重复。
哈希表时间复杂度 查询、插入、删除一般都是O(1)，；最坏情况下是Size太小或者函数选的不好，造成很多冲突，进而退化为链表，时间复杂度为O(n)。
Map：key-value键值对，key不重复
Set: 不重复元素的集合
Q: 有效的字母异位词
A: 1. 暴力法，sort, sorted_str 相等？ O(NlogN)  N表示字符串长度
2. hash, map --> 统计每个字符的频次，如果每个字符频次一样，肯定符合要求。
Q: 字母异位词分组
A: 
Q: 两数之和
A: 哈希表做法：a, b --> a + b == target --> for each a: check (target - a ) exists in nums
看下官方的题解
国际站Python：O(n) 
class Solution(object):
    def twoSum(self, nums, target):
        if len(nums) <= 1:
            return False
        buff_dict = {}
        for i in range(len(nums)):
            if nums[i] in buff_dict:
                return [buff_dict[nums[i]], i]
            else:
                buff_dict[target - nums[i]] = i

注意：养成收藏优秀代码的习惯
做题四件套：
1. clarification   把题目过一遍
2. possible  solutions --> optimal (time & space) 找最优
3. code  
4. test cases 测试样例

Linked List是特殊化的Tree, Tree是特殊化的Graph
单链表的next指针如果有多个，就演化为树。
树和图的区别就是有没有环，有环就是图。

class TreeNode:
  def __int__(self, val):
    self.val = val
    self.left, self.right = None, None

二叉树遍历主要有三种：(1) 前序遍历(Pre-order): 根-左-右; (2)中序(In-order): 左-根-右; (3)后序(Post-order): 左-右-根
树的各种操作要拥抱递归。
def preorder(self, root):
    if root:
        self.traverse_path.append(root.val)
        self.preorder(root.left)
        self.preorder(root.right)

def inorder(self, root):
    if root:
        self.inorder(root.left)
        self.traverse_path.append(root.val)
        self.inorder(root.right)

def postorder(self, root):
    if root:
        self.postorder(root.left)
        self.postorder(root.right)
        self.traverse_path.append(root.val)

二叉搜索树(Binary Search Tree)也叫二叉排序树、有序二叉树(Ordered Binary Tree)、排序二叉树(Sorted Binary Tree)，是指一棵空树或具有下列性质的二叉树：
1.左子树上所有结点的值均小于它的根结点的值；
2.右子树上所有结点的值均大于它的根结点的值；
3.以此类推：左右子树也分别为二叉查找树(这就是重复性)。

中序遍历也是升序排列
二叉搜索树常见操作： 查询logn、 插入新结点（创建）logn、删除logn。
树的解法一般都是递归(代码本身，树的定义没有所谓的后继结果或者便于循环的结构，而更多的是左结点右结点，访问子树经常更好的一种方式是直接对它的左结点再调相同的遍历函数)
自己画图尝试三种遍历方式
傻递归锅不在递归，而不是没有做缓存。例如Fibonacci数列

二叉树三种遍历方式自己写写

递归的Python代码模板
def recursion(level, param1, param2, ...)
    # recursion terminator
    if level > MAX_LEVEl:
        process_result
        return

    # process logic in current level
    process(level, data...)

    # drill down
    self.recursion(level +1, p1, ...)

    # reverse the current level status if needed

递归的思维要点：1.不要人肉递归(最大误区)；2.找到最近最简方法，将其拆解成可重复解决的问题（重复子问题）; 3. 数学归纳法思维

堆Heap: 可以迅速找到一堆数中的最大或最小值得数据结构。常见的有二叉堆，斐波那契堆。根节点最大的堆叫做大顶堆或大根堆；根节点最小的堆叫做小顶堆或小根堆。

大顶堆常见的操作： find-max：O(1);  delete-max: O(logN); insert(create): O(logN) or O(1)

二叉堆是通过完全二叉树(根全满，注意：不是二叉搜索树)来实现，二叉堆(大顶)，是一棵完全树；数中任意节点的值总是>=其子节点的值。 
二叉堆实现细节：一般都是通过“数组”来实现；
二叉堆，假设“第一个元素，根节点，顶堆元素”在数组中的索引为0的话，则父节点和子节点的位置关系如下：
(1) 索引为i的左孩子的索引是(2 * i + 1)；
(2) 索引为i的右孩子的索引是(2 * i + 2);
(3) 索引为i的父结点的索引是floor((i-1)/2) 取整;
找最大值O(1)  
(再回顾下图形) Insert 插入操作 O(logn)： 1. 新元素一律先插入到堆得尾部； 2. 依次向上调整（heapify up）整个堆得结构（一直到根即可）
(再回顾下图形) Delete Max删除堆顶操作： 1. 将堆尾元素替换到顶部（即堆顶被替换删除掉） 2.依次从根部向下（heapify down）调整整个堆的结构（与较大的比较，一直到堆尾即可）

二叉堆是一种常见且简单(优先队列 Priority Queue)的实现，并不是最优的实现。

Q: 最小的k个数
A: 1. sort: NlogN  2. heap: NlogK  3. quick-sort

Q: 滑动窗口最大值
A: 1. 双端队列 2. heap 优先队列 大顶堆 

Q: 前K个高频元素 （特别重要）
A: 统计每个元素的频次，可以使用哈希表（不知道数的范围），也可以用数组（知道数的范围，可以使用下标）
(一般看到logn就是二叉堆，就是用堆或者用二叉搜索树或者二分查找或者是排序这样的手段)

----- 图不做重点 ------
图：属性 Graph(V，E)
V - vertex: 点 （1. 度 - 入度和出度，连了多少边； 2. 点与点之间：连通与否）
E - edge: 边 （1.有向和无向（单行线）； 2.权重（边长））

图的表示方法：临接矩阵、邻接表
无向无权图、有向无权图、无向有权图

基于图的算法： DFS BFS
和树最大区别是 DFS和BFS必须加上 visited = set()  

图的高级算法(leetcode)： 连通图个数、拓扑排序、 最短路径、最小生成树
