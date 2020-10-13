学习笔记

哈希表(Hash table)，根据关键码值(key value)而直接进行访问的数据结构，通过把关键码值映射到表中的一个位置来访问记录，来加快查询速度。使用的映射参数叫做散列函数(Hash Function)，存放记录的数组叫做哈希表或散列表。常用于用户信息存储，缓存，键值对存储。如果函数选的不好可能会出现哈希重复。


Linked List是特殊化的Tree, Tree是特殊化的Graph

class TreeNode:
  def __int__(self, val):
    self.val = val
    self.left, self.right = None, None

二叉树遍历主要有三种：(1) 前序遍历(Pre-order): 根-左-右; (2)中序(In-order): 左-根-右; (3)后序(Post-order): 左-右-根

二叉搜索树(Binary Search Tree)也叫二叉排序树、有序二叉树(Ordered Binary Tree)、排序二叉树(Sorted Binary Tree)，是指一棵空树或具有下列性质的二叉树：
1.左子树上所有结点的值均小于它的根结点的值；
2.右子树上所有结点的值均大于它的根结点的值；
3.以此类推：左右子树也分别为二叉查找树。

中序遍历也是升序排列

树的解法一般都是遍历

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

递归的思维要点：1.不要人肉递归(最大误区)；2.找到最近最间方法，将其拆解成可重复解决的问题（重复子问题）; 3. 数学归纳法思维

堆Heap: 可以迅速找到一堆数中的最大或最小值得数据结构。常见的有二叉堆，斐波那契堆。根节点最大的堆叫做大顶堆或大根堆；根节点最小的堆叫做小顶堆或小根堆。

大顶堆常见的操作： find-max：O(1);  delete-max: O(logN); insert(create): O(logN) or O(1)

二叉堆是通过完全二叉树来实现，大顶，是一棵完全树；数中任意节点的值总是>=其子节点的值。 一般都是通过“数组”来实现；二叉堆是一种常见且简单的实现，并不是最优的实现。


