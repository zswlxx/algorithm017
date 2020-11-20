学习笔记


搜索-遍历
1.每个节点都要访问一次
2.每个节点仅仅要访问一次
3.对于节点的访问顺序不限
- 深度优先：depth first search
- 广度优先：breadth first search

DFS用递归程序简单，系统自动维护一个栈；也可以用循环方式。
# 递归写法
def dfs(node):
    if node in visited:
        # already visited
        return

    visited.add(node)

    # process current node
    # ... # logic here
    dfs(node.left)
    dfs(node.right)
    # 如果是二叉树，就是左孩子有孩子；如果是图，就是它的联通的相邻结点

# 非递归 (需要手动维护一个栈)
def DFS(self, tree):

    if tree.root is None:
        return []

    visited, stack = [], [tree.root]

    while stack:
        node = stack.pop()
        visited.add(node)

        process(node)
        nodes = generate_related_nodes(node)
        stack.push(nodes)

    # other processing work
    ...
# 多叉树
visited = set()

def dfs(node, visited):
    if node in visited: # terminator
        # already visited
        return

    visited.add(node)
    # process current node here.
    ...
    for next_node in node.children():
        if next_node in visited:
            dfs(next_node, visited)

    # 状态树和访问状态树熟悉（树和图）
    DFS  BFS 用对称二叉树对比下

def BFS(graph, start, end):
    visited = set()
    queue = []
    queue.append([start])

    while queue:
        node = queue.pop()
        visited.add(node)

        process(node)
        nodes = generate_related_nodes(node)
        queue.push(nodes)

    # other processing work
    ...

Q: 二叉树的层次遍历
A: 1. BFS  2. DFS (level)
看下官方题解 深度优先和广度优先都写一遍

Q: 最小基因变化
A: 看下题解，对应下括号问题

Q: 括号生成
A: DFS  左括号右括号就是状态树  也试试BFS

Q: 岛屿数量
A: DFS  BFS  并查集
以DFS为例: 用一个嵌套循环，循环数组里面的每一个元素，如果碰到是1，说明岛屿的数量至少找到了一个，所以岛屿数量+1；
然后把和1左右上下相邻的所有的点且相邻的无限递归下去，也就是说和所有1相连接的其他的1全部打掉，夷为平地变成0,
继续往下走知道循环到最末尾，整个地图被夷为平地打成0，说明我们把所有的岛全部都统计完了。而岛的数量就在我们每次碰到1的时候累加的时候就已经加出来了。


贪心算法
贪心算法是一种在每一步选择中都采取在当前状态下最好或最优（即最有利）的选择，从而希望导致结果是全局最好或者最优的算法。

贪心算法与动态规划的不同在于它对每个子问题的解决方案都做出选择，不能回退。
动态规划会保存以前的运算结果，并根据以前的结果对当前的进行选择，右回退功能。

[贪心： 当下做局部最优判断；回溯： 能够回退； 动态规划： 最优判断+回退]

遇到选择最优、最近、最好的问题，都可以选择贪心算法
贪心算法可以解决一些最优化问题：如求图中的最小生成树、求哈夫曼编码等。
一旦一个问题可以通过贪心算法来解决，那么贪心法一般是解决这个问题的最好方法。
由于贪心法的高效性以及其所求得的答案比较接近最优结果，贪心法可以用作辅助算法或者直接解决一些要求结果不特别精确的问题。

贪心算法有条件（例如，备选有整除关系）

问题能够分解成子问题来解决，子问题的最优解能递推到最终问题的最优解。这种子问题最优解成为最优子结构。

Q: 分发饼干
A: 参考江五渣解法

Q: 最佳股票买卖时间
A: 1.最典型的做法是动态规划
2.贪心算法参考官方题解（一次遍历）

Q: 跳跃游戏
A: 1.穷举暴力搜索，按层递归 时间复杂度是指数级
2. 时间复杂度n^2  参考官方题解
3. 贪心算法 
# java  O(n) 从后往前
class Solution{
    public bollean canJump(int[] nums) {
        if (nums == null) {
            return false;
        }
        int endReachable = nums.length - 1;
        for (int i = nums.length - 1; i >= 0; i--) {
            if (nums[i] + i >= endReachable) {
                endReachable = i;
            }
        }
        return endReachable == 0;
    }
}

二分查找最关键的三个前提
1.目标函数单调性（单调递增或递减）
2.存在上下界（bounded）
3.能通过索引访问(index accessible)
代码模板形成肌肉式记忆
left, right = 0, len(array) - 1
while left <= right:
  mid = (left + right) / 2
  if array[mid] == target:
    # find the target
    break or return result
  elif array[mid] < target:
    left = mid + 1
  else:
    right = mid - 1

Q: 求平方根
A: 1. 二分查找 y = x^2, (x > 0): 抛物线，在y轴右侧单调递增；上下界
# java
class Solution {
    public int mySqrt(int x) {
        if (x == 0 || x == 1)
            return x;
        long left = 1, right =x;
        long mid = 1;
        while (left <= right) {
            mid = left + (right - left) / 2;
            if (mid * mid > x){
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return (int) right;
    }
}
// 方法2： 牛顿迭代法（学习下 ）
class Solution(object):
    def mySqrt(self, x):
        r = x
        while r * r > x:
            r = (r + x/r) / 2
        return r

# 五毒神掌； 四步做题
Q: 搜索旋转排序数组
A：1. 暴力：还原 O(logN) -> 升序 -> 二分： O(logN) (写、总结)
2. 正解： 二分查找 参考breezean的解法 参考jimmy00745解释
a. 单调
b. 边界
c. index 
