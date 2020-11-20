学习笔记
前序知识回顾：
树的解法一般都是遍历
1. 节点和树本身数据结构的定义就是用递归的方式进行的
2. 不仅是树本身、二叉树以及搜索二叉树，在定义数据结构和算法特性的时候也是有所谓的重复性（自相似性）

递归本质就类似于循环，只不过是通过循环体调用自己来进行所谓的循环

递归Recursion
最简单的递归例子 阶乘n!
def Factorial(n):
    if n <= 1:
        return 1
    return n * Factorial(n-1)

递归的Python代码模板
def recursion(level, param1, param2, ...)
    # recursion terminator 递归终止条件
    if level > MAX_LEVEl:
        process_result
        return

    # process logic in current level  处理当前层逻辑
    process(level, data...)

    # drill down  下探到下一层
    self.recursion(level +1, p1, ...)

    # reverse the current level status if needed   清理当前层

递归的思维要点：1.不要人肉递归(最大误区)；2.找到最近最简方法，将其拆解成可重复解决的问题（重复子问题）; 3. 数学归纳法思维
数学归纳法：最开始最简单的条件是成立的，且第二点能证明当n成立的时候可以推导出n+1也成立
Q: 爬楼梯
A: 见第一周课程   mutual exclusive, complete exhaustive

Q:括号生成
A:递归  抽象成2n个格子
递归模板写下来
// terminator 
left == n && right ==n
// process current logic: left, right
// drill down 
left 随时可以加，只要别超标（n）
right 必须之前有left ，且left个数 > right
// reverse states
# python
def generateParenthesis(self, n):
    def generate(p, left, right, parens=[]):
        if left:     generate(p + '(', left-1, right)
        if right > left: generate(p + ')', left, right-1)
        if not right: parens += p,
        return parens
    return generate('', n, n)

爱上看别人代码，这个地方还可以这样写，抄下来，练会
Q: 验证二叉搜索树
A: 二叉搜索树BST --> 中序遍历是递增的
推荐看官方题解代码

Q: 二叉树的最大深度
A: 找出重复性，最大深度只可能来自两个地方：左子树深度+根1 或 右子树深度+根1，哪个最大就是哪个 
左子树/右子树深度怎么求，递归调用即可
拒绝人肉递归

本期的作业要看看官方代码， 二叉树最近公共祖先（要会做）


思路：不管是递归、分治、还是回溯，或者其他的办法，最后本质就是找重复性以及分解问题，和最后组合每个子问题的结果

分治代码模板

def divide_conquer(problem, param1, param2, ...):
  # recuesion terminator
  if problem is None:
    print_result
    return
  # prepare data
  data = prepare_data(problem)
  subproblems = split_problem(problem, data)
  # conquer subproblems
  subresult1 = self.divide_conquer(subproblems[0], p1, ...)
  subresult2 = self.divide_conquer(subproblems[1], p1, ...)
  subresult3 = self.divide_conquer(subproblems[2], p1, ...)
  ...
  # process and generate the final result
  result = process_result(subresult1, subresult2, subresult3, ...)
  # revert the current level states

回溯
回溯法采用试错的思想，它尝试分步的去解决一个问题。在分步解决问题的过程中，当它通过尝试发现现有的分步答案不能得到有效正确的解答的时候，它将取消上一步甚至是上几步的计算，再通过其他的可能的分步解答再次尝试寻找问题的答案。
回溯法通常用最简单的递归方法来实现，再反复重复上述的步骤后肯恩出现两种情况:
1. 找到一个可能存在的正确的答案
2. 在尝试了所有的可能的分步方法后宣告该问题没有答案
在最坏的情况下，回溯法会导致一次复杂度为指数时间的计算。

回溯最典型的问题是八皇后和数独问题

Q: 实现求x的n次方函数 （非常高频）
A: 1. 暴力法 for i: 0 -> n { result ****= x  }
2. 分治
template: 1. terminator 2. process (split your big problem) 3. drill down(subproblems) , merge(subresult) 4. reverse states

x^n --> 2^10 : 2^5 -> (2^2)*2
pow(x, n):
    subproblem: subresult = pow(x, n/2)

merge:
if n % 2 == 1 {
    // odd
    result = subresult * subresult * x;
} else {
    // even
    result = subresult * subresult
}
注意边界条件！！比较阴险

Q: 子集
A: 

# python 
class Solution(object):
    def subsets(self, nums):
        subsets = [[]]

        for num in nums:
            newsets = []
            for subset in subsets:
                new_subset = subset + [num]
                newsets.append(new_subset)
            subsets.extend(newsets)

        result newssets

推荐参考 powcai的解法

Q: 电话号码的字母组合
A: 回溯 看官方题解（python代码）

Q: 八皇后问题
A: 回溯 liweiwei1419动画 和 官方动画
# python
class Solution(object):
    def solveNQueens(self, n):
        if n < 1: return []

        self.result = []
        # 之前的皇后所攻击的位置
        self.cols = set() 
        self.pie = set() 
        self.na = set()

        self.DFS(n, 0, [])
        return self._generate_result(n)

    def DFS(self, n, row, cur_state):
        # recursion terminator 最后一层
        if row >= n:
            self.result.append(cur_state)
            return

        # current level ! Do it !
        for col in range(n):  # 遍历列 column
            if col in self.cols or row + col in self.pie or row - col in self.na:
                # go die !
                continue

            # update the flags
            self.cols.add(col)
            self.pie.add(row + col)
            self.na.add(row - col)
            # 下探
            self.DFS(n, row + 1, cur_state + [col])
            
            # reverse states
            self.cols.remove(col)
            self.pie.remove(row + col)
            self.na.remove(row - col)

    def _generate_result(self, n):
        board = []
        for res in self.result:
            for i in res:
                board.append("." * i + "Q" + "." * (n -i -1))

        return [board[i: i + n] for i in range(0, len(board), n)]

# python
def solveNQueens(self, n):
    def DFS(queens, xy_dif, xy_sum):
        p = len(queens)
        if p == n:
            result.append(queens)
            return None
        for q in range(n):
            if q not in queens and p-q not in xy_dif and p+q not in xy_sum:
                DFS(queens+[q], xy_dif+[p-q], xy_sum+[p+q])

    result = []
    DFS([], [], [])
    return [["." * i + "Q" + "." * (n-i-1) for i in sol] for sol in result]


