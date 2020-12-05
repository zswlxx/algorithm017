学习笔记

递归模板 python版

def recursion(level, param1, param2, ...):
    # recursion terminator
    if level > MAX_LEVEL:
        process_result
        return

    # process logic in current level
    process(level, data ...)

    # drill down
    self.recursion(level + 1, p1, ...)

    # reverse the current level status if needed

分治代码模板 python版
def divide_conquer(problem, param1, param2, ...):
    # recursion terminator
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

1. 人肉递归低效、很累
2. 找到最近最简方法，将其拆解为可重复解决的问题（面试一般都有解）
3. 数学归纳法思维（抵制人肉递归的诱惑）
本质：寻找重复性
人肉递归可以将递归状态树画出来

动态规划：动态递推，用一种递归方法进行分治
动态规划和递归或者分治没有根本上的区别（关键看有误最优子结构）
共性：找到重复子问题
差异性：最优子结构、中途可以淘汰次优解

Fibonacci数列代码 时间复杂度O(n)

int fib(int n, int[] memeo) {
    if (n <= 1) {
        return n;
    }

    if (memo[n] == 0) { //砍掉重复的分支
        memo[n] = fib(n - 1) + fib( n - 2);
    }
    return memo[n];

}

right clean code

自顶向下，自底向上（熟练使用） 

动态规划关键点：
1.最优子结构 opt[n] = best_of(opt[n-1], opt[n-2], ...)
2.储存中间状态：opt[i]
3.递推公式（状态转移方程或DP方程）
Fib: opt[i] = opt[n-1] + opt[n-2]
二维路径： opt[i,j] = opt[i+i][j] + opt[i][j+1] (且判断a[i,j]是否空地)


Q:不同路径（不用判断障碍物）
DP方程：dp[i][j]= dp[i-1][j] + dp[i][j-1]
# Java
class Solution:
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < n; i++) dp[0][i] = 1;
        for (int i = 0; i < m; i++) dp[i][0] = 1;
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1]
    }

# Java 2
class Solution {
    public int uniquePaths(imt m, int n) {
        int[] cur = new int[n];
        Arrays.fill(cur, 1);
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                cur[j] += cur[j-1];
            }
        }
        return cur[n-1]
    }
}

Q: 不同路径2
A: 参考powcai解法 和官方题解的评论
国际站 tusizi JAVA解法

Q: 最长公共子序列(顺序不可颠倒)
A: 1. 暴力枚举生成子序列，类似括号问题 O(2^n)
   2. 找重复性，化解为Fib
字符串问题：
0. s1 = ""  //最长子序列是空 
   s2 = ""
1. s1 = ""  //最长子序列是空（只要存在一个是空）
   s2 = 任意字符串
2. s1 = "A"  // 是否A存在S2，存在则最长子序列为1，不存在未空
   s2 = 任意
3. s1 = "......A"  // 最后一个字母相同，则至少存在一个，其实可以转化为子问题，求S1和S2的前面的任意字符串的最长子序列的值再加1
   s2 =  ".....A"
   字符串的变化的问题，特别是两个字符串之间的变化，最后要做成一个二维数组，行和列分别就是两个不同的字符串
   S1 = "ABAZDC"
   S2 = "BACBAD"
   // -1 表示最后一个字符，也可以谢伟n-1，也即 S1.length-1或者S2.length-1
   if S1[-1] != S2[-1]: LCS[s1, s2] = Max(LCS[s1-1,s2], LCS[s1, s2-1]) //DP方程 去掉前者一个字符和后者比或者后者一个字符和前者比
   LCS[s1, s2] = Max(LCS[s1-1,s2], LCS[s1, s2-1], LCS[s1-1, s2-1])
   if S1[-1] = S2[-1]: LCS[s1, s2] = LCS[s1-1, s2-1] + 1  //DP方程
   LCS[s1, s2] = Max(LCS[s1-1, s2], LCS[s1, s2-1], LCS[s1-1, s2-1], LCS[s1-1][s2-1] + 1) // 最后一个肯定是最大
 # Python
class Solution(object):
    def longestCommonSubsequence(self, text1, text2):
        if not text1 or not text2:
            return 0
        m = len(text1)
        n = len(text2)
        dp = [[0]*(n+1) for _ in range(m+1)  //初始化数组，开始全部是0
        for i in range(1, m+1):  // 递推，两层循环
            for j in range(1, n+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

        只要DP方程列出来，程序并不复杂，关键在于：1. 大家的经验和习惯，多练； 2. 数组优势要定义m+1,有时候定位n+1，以及初始化要初始化好，同时下标的位置别越界，同时最后的结果是在m和n的这个位置
        多写几遍，把感觉找到
动态规划思维小结：
1.打破自己的思维惯性，形成机器思维
2.理解复杂逻辑的关键
3.也是职业进阶的要点要领

Q:爬楼梯问题
A: DP方程：F(n) = F(n-1) + F(n-2)
注意该问题的扩展问题

Q: 三角形最小路径和
A: 1. brute-force ，递归，n层： left or right: O(2^n)
   2. DP步骤  a. 重复性（分治） b. 定义状态数组 c. DP方程
这个问题类似于之前的不同路径，只不过路径是二维的矩阵，也就是比较工整的二维数组
而三角形就相当于切掉了一个对角线的一个三角形的数组，转换的子问题: problem(i,j) = min(sub(i+1, j), sub(i+1, j+1)) + a[i, j] 
dp[i][j] = triangle[i][j] + min(dp[i+1][j], dp[i+1][j+1])
注意看下国际站： caikehe 题解的优化过程
# Python
class Solution(object):
    def mininumTotal(triangle):
        mini, M = triangle[-1], len(triangle)
        for i in range(M - 2， -1， -1):
            for j in range(len(triangle[i])):
                mini[j] = triangle[i][j] + min(mini[j], mini[j+1])
        return mini[0]

# python
class Solution:
    def mininumTotal(self, triangle):
        dp = triangle //相当于把triangle的值初始化到dp
        for i in range(len(triangle)-2, -1, -1):
            for j in range(len(triangle[i])):
                dp[i][j] += min(dp[i+1][j], dp[i+1][j+1])
        print(triangle[0][0])
        return dp[0][0]

Q: 最大子序列和
A: 1. 暴力：枚举 O(2^n)
   2. DP: a. 分治（子问题） max_sum(i) = Max(max_sub(i-1), 0) + a[i]
   b.状态数组定义  f[i]
   c.DP方程: f[i] = Max(f[i-1], 0) + a[i]
# Python
class Solution(object):
    """
        1. DP问题，公式为：dp[i] = max(nums[i], nums[i] + dp[i - 1])
        2. 最大子序和 = 当前元素自身最大， 或者 包含之前后最大
    """
    def maxSubArray(nums):
        dp = nums
        for i in  range(1, len(nums)):
            # nums[i-1]代表dp[i - 1]
            dp[i] = max(nums[i], nums[i] + dp[i - 1])
            # dp[i] = max(nums[i] + 0, nums[i] + dp[i - 1])
            # dp[i] = max(0, dp[i - 1]) + nums[i] 这个是上步的提取nums[i]后简化的
            # nums[i] = max(0, nums[i - 1]) + nums[i]  这个是复用nums

        return max(dp)

该题目提升： 152题 乘积最大子序列
# Python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        mi = max = res = nums[0]
        for i in range(1, len(nums)):
            if nums[i] < 0: mi, ma = ma, mi
            ma = max(ma * nums[i], nums[i])
            mi = min(mi * nums[i], nums[i])
            res = max(res, ma)
        return res

Q: 322 零钱兑换
A: 
1. 暴力： 递归： 指数级复杂度 把所有的不同的路径都给它求出来（递归状态树）
2. BFS 
3. DP  a. subproblem
       b. DP array：f(n) = min{f(n-k), for k in [1, 2, 5]} + 1  其中1 表示至少再加上一个硬币，这个硬币的面值就是k。为什么是n-k, 是因为现在给的是一个数组，而不是常数，k就是在这个数组里面取所有的值。 
       c. DP方程 
这个题可以看下官方题解，国际站的python代码。
# python   时间复杂度O(amount * coins.length) 空间复杂度 O(amount)
class Solution(object):
    def coinChange(self, coins, amount):
        MAX = float('inf')
        dp = [0] + [MAX] * amount

        for i in xrange(1, amount + 1):
            dp[i] = min([dp[i - c] if i - c >= 0 else MAX for c in coins]) + 1

        return [dp[amount], -1][dp[amount] == MAX]
假设因为有最少这个样子，所以这里的话有最佳的子结构
这个问题再泛化一点，不是最少的硬币个数，问有多少不同的组合方式，就等同于上楼梯问题
# 硬币兑换状态树：
根11 -1 -2 -5分别转化为子问题：10 9 6的子问题，不断往下走，叶子节点就是到0的话表示刚好凑对了一组解。
还有一种情况是叶子节点是负，表示凑不到，就可以停掉。
树的深度表示用的硬币的个数，在第几层就表示用了几个硬币，
演变为在这颗状态树里面找到节点为0的结点，且是层次最低的，就是深度要是最小的。
也就是广度优先，一层层往下遍历，直到第一次碰到数值为0的结点，那么当前的层就是我们要的答案，遍历的深度层数就是最小使用的硬币数，边即为硬币面值。

在写代码时，一定要转换成机器思维，在提升自己认知的时候，一定要注意，要找到一个所谓的自相似的办法，也就是说有重复性的办法，然后把它进行化繁为简，同时逻辑上是简洁的，且能够严谨证明的。

Q:打家劫舍
A: DP  a[i]: 0..i 能偷到max value, 第i个房子可偷可不偷   0:不偷 1:偷
DP方程： a[i] = Math.max(a[i - 1, nums[i] + a[i - 2]]);
# python
class Solution(object):
    def rob(self, nums):
    pre = 0
    now = 0
    for i in nums:
        pre, now = now, max(pre + i, now)
    return now
# Java 
class Solution {
    public int rob(int[] nums) {
        int n = news.length;
        if ( n <= 1 ) return n == 0 ? 0 : nums[0];

        int[] dp = new int[n];
        dp[0] = nums[0]
        dp[1] = Math.max(nums[0], nums[1]);

        for ( int i = 2; i < n; i++ ) 
            dp[i] = Math.max(dp[i - 1], nums[i] + dp[i - 2]);

        return dp[n - 1];
    }
}
打家劫舍2 参考Krahets题解和讨论里面Ant的代码 注意不偷最后一个房子，应该是n-2,而不是n-1


