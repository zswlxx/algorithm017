学习笔记
# python递归模板
public void recur(int level, int param) {
    // terminator
    if (level > MAX_LEVEL) {
        // process result
        return;
    }

    // process current logic
    process(level, param);

    // drill down
    recur(level: level + 1, newParam);

    //restore current status

}

# python分治代码模板
def divide_conquer(problem, param1, param2, ...):
    # recursion terminator 递归终止条件
    if problem is None:
        print_result
        return

    # prepare data 准备数据和拆分数据
    data = prepare_data(problem)
    subproblems = split_problem(problem, data)

    # conquer subproblems 调分治函数递归求解
    subresult1 = self.divide_conquer(subproblems[0], p1, ...)
    subresult2 = self.divide_conquer(subproblems[1], p2, ...)
    subresult3 = self.divide_conquer(subproblems[2], p3, ...)
    ...

    # process and generate the final result  合并结果
    result = process_result(subresult1, subresult2, subresult3, ...)

    # revert the current level states  返回

总结：
1. 人肉递归低效、很累
2. 找到最近最简方法，将其拆解成可重复解决的问题
3. 数学归纳法思维
本质：寻找重复性-> 计算机指令集（面试一般不超30行）

分治+记忆化缓存就过渡到动态规划

动态规划：
1. 将一个复杂的问题分解成为各个简单的子问题
2. 分治 + 最优子结构
3. 顺推形式： 动态递推（从下往上推）

DP顺推模板（嵌套循环），两个复杂点
function DP():
    dp = [][] # 二维情况，复杂一，DP状态定义需要经验，同事需要把现实的问题定义成一个数组，里面保存状态，数组可能是一维、二维、三维。

    for i = 0 .. M {
        for j = 0 .. N {
            dp[i][j] = _Function(dp[i'][j']...) # 状态转移方程，复杂二，最简化的就是Fibonacci数列，更多的情况如：1. 求一个最小值；2.累加累减；3.有一层小的循环，从之前的k个状态里面找出它的最值。
        }
    }

    return dp[M][N];

关键点：
动态规划和递归或者分治没有根本上的区别（关键看有无最优的子结构）
拥有共性： 找到重复子问题

差异性： 动态规划用来处理有所谓中间的重复性以及所谓的最优子结构、在中途可以淘汰次优解。

常见的的DP题目和它的状态转移方程（多多回忆之前做过的DP问题）
1. 爬楼梯（老生常谈中的老生常谈）
递归公式：
f(n)=f(n - 1)+f(n - 2), f(1)=1, f(0)=0
（1）爬楼梯问题本质上可以转换为Fibonacci问题 （2）爬楼梯问题和硬币置换问题有异曲同工之处。
def f(n): // 最朴素 O(2^n)
    if n <= 1: return 1
    return f(n - 1) + f(n - 2)

def f(n): // 分治 + 记忆化搜索 O(n)
    if n <= 1: return 1
    if n not in mem:
        mem[n] = f(n - 1) + f(n - 2)
    return mem[n]

def f(n): // 把递归转换为顺推，变为for循环 O(n)
    dp = [1] * (n + 1)  内存的优化
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

def f(n): // O(n), O(1)
    x, y = 1, 1
    for i in range(1, n):
        y, x = x + y, y
    return y

2. 不同路径(没有障碍物，有障碍物需要补充下)
递推公式：f(x, y) = f(x - 1, y) + f(x, y - 1)
def f(x, y): // 分治 指数级
    if x <= 0 or y <= 0: return 0
    if x == 1 and y == 1: return 1
    return f(x - 1, y) + f(x, y - 1)

def f(x, y): // 加缓存 O(mn), O(mn)
    if x <= 0 or y <= 0: return 0
    if x == 1 and y == 1: return 1
    if (x, y) not in mem:
        mem[(x, y)] = f(x - 1, y) + f(x, y - 1)
    return mem[(x, y)]

def f(x, y): //转递推  O(mn), O(mn)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    dp[1][1] = 1
    for i in range(1, y + 1):
        for j in range(1, x + 1):
            dp[i][j] = dp[i - 1][j] + dp[j][i - 1]
    return dp[y][x]

3. 打家劫舍
dp[i] 状态的定义： max $ of robbing A[0 -> i]
dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
# 多加一个维度
dp[i][0]状态定义： max $ of robbing A[0 -> i] 且没偷 nums[i]
dp[i][1]状态定义： max $ of robbing A[0 -> i] 且偷了 nums[i]

dp[i][0] = max(dp[i - 1][0], dp[i - 1][1]);
dp[i][1] = dp[i - 1][0] + nums[i];

4. 最小路径和
dp[i][j] 状态的定义： minPath(A[1 -> i][1 -> j])
dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + A[i][j]

5. 买卖股票的最佳时机（系列）参考labuladong一个方法团灭6道股票问题
dp[i][k][0 or 1] (0 <= i <= n-1, 1 <= k <= K)
i 为天数 k 为最多交易次数 [0, 1] 为是否持有股票
总状态数： n * K * 2 种状态
for 0 <= i < n:
    for 1 <= k <= K:
        for s in {0, 1}:
            dp[i][k][s] = max(buy, sell, rest)

dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
              max( 选择 rest,      选择 sell  )
解释：今天我没有持有股票，有两种可能：
- 我昨天就没有持有， 然后今天选择rest, 所以我今天还是没有持有；
- 我昨天持有股票， 但是我今天sell了，所以我今天没有持有股票。

dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
              max( 选择 rest,      选择 buy  )
解释：今天我持有着股票，有两种可能：
- 我昨天就持有股票，然后今天选择rest, 所以我今天还持有着股票；
- 我昨天本没有持有，但今天我选择buy, 所以今天我就持有股票了。

初始状态：
dp[-1][k][0] = dp[i][0][0] = 0
dp[-1][k][1] = dp[i][0][1] = -infinity

状态转移方程：
dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])

复杂度来源：
1. 状态拥有更多的维度（二维、三维、或者更多、需要压缩）
2. 状态方程更加复杂

本质：内功、 逻辑思维、数学

# 爬楼梯问题改进：
1. 1、2、3  (a[0]=1; a[1]=2; a[2]=3)
for (int i = 2; i < n; ++i)
    a[i] = a[i-1] + a[i-2] + a[i-3];
2. x1, x2, ..., xm步(数组里面是可以走的步数)
for (int i = 2; i < n; ++i)
    for (int j = 0; j < m; ++j)
        a[i] += a[i - x[j]];
3. 前后不能走相同的步伐
a[i][k]  (i 表示上到第几级台阶；k 表示当前这一步走的是第几步)
for (int i = 2; i < n; ++i)
    for (int j = 0; j < m; ++j)
        for (int k = 0; k < m; ++k)
            a[i][x[j]] += a[i - x[j][x[k]]];  // 这个不对，需要考虑假设k走了3步。

# 编辑距离：
1. BFS , two-ended BFS , word1长度m, word2长度n 长度更长的单词只可能长度减少，长度少的单词只可能长度增加，最后词的长度的变化范围在m和n之间，长度向中间逼近
2. DP，注意看官方题解的表格 和 powcai的解法代码(-1,-1也即n1,n2)
dp[0..i][0..j]  // 如何定义？ word1[0:i] 与 word2[0:j] 之间的编辑距离

w1: ....x (i)
w2: ...x (j)
edit_dist(w1, w2) = edit_dist(w1[0:i-1], w2[0:j-1])
等价于 edit_dist(i, j) = edit_dist(i-1, j-1) if w1[i] == w2[j] // 分治

w1: ....x (i)
w2: ...y (j)
// w1[i] != w2[j]  增加和删除是同理可得
edit_dist(i, j) = min(edit_dist(i - 1, j - 1) + 1, // 打掉x y
                  edit_dist(i - 1, j) + 1,  // 打掉x 
                  edit_dist(i, j - 1) + 1  // 打掉y)

如果word1[i]与word2[j]相同，显然dp[i][j] = dp[i-1][j-1]
如果word1[i]与word2[j]不同，那么dp[i][j]可以通过
    1. 在dp[i-1][j-1]的基础上做replace操作达到目的
    2. 在dp[i-1][j]的基础上做insert操作达到目的
    3. 在dp[i][j-1]的基础上做delete操作达到目的
取三者最小情况即可（2和3没有绝对的）

Python中的字符串是不可变的 immutable是线程安全的,每次改变string时其实都是创建了一个新的string。

# 给定指定字符串，找出它的第一个不重复的字符，并返回索引
1. brute-force: O(n^2)
i 枚举所有字符
    j 枚举 i 后面的所有字符 // 找重复
2. 统计字符出现多少次 map(hashmap O(1), treemap二叉搜索树  O(logN))
   整体 O(N)     or    O(NlogN)
3. hash table (用字母的对应的数组，字母对应的下标来统计)
可以参考官方的题解

# 字符串转换整数
一定要思考下每一步干啥，如空格去掉，注意符号
# Python 
class Solution(object):

    def myAtoi(self, s):

        if len(s) == 0: return 0
        ls = list(s.strip())

        sign = -1 if ls[0] == '-' else 1

        if ls[0] in ['-', '+'] : del ls[0]

        ret, i = 0, 0

        while i < len(ls) and ls[i].isdigit() :
            ret = ret * 10 + ord(ls[i]) - ord('0')
            i += 1

        return max(-2 ** 31, min(sign * ret, 2 ** 31 - 1))

# 找出最长公共前缀
1. 纯暴力 O(M * n^2)  M为单词平均长度
2. 两层循环，排列，遍历列
3. Trie
看下官方题解

# 翻转字符串
头指针不断加加，尾指针不断减减，交换两个数
# Java
class Solution {
    public void reverseString(char[] s) {
        if (s == null) return;

        for (int i = 0, j = s.length - 1; i < j; ++i, --j) {
            char tmp = s[i]; s[i] = s[j]; s[j] = tmp;
        }
        // 以下只是一个练习的示例，与本题无关，要熟记, 嵌套遍历数组
        for (int i = 0; i < a.length - 1; ++i)
            for (int j = i+ 1; j < a.length - 1; ++i)
    }
}

# 翻转字符串中的逐个单词
1. split, reverse, join
2. reverse整个string, 然后再单独reverse每个单词

最长子串、子序列
1. 最长子序列
dp[i][j] = dp[i-1][j-1] + 1 (if s1[i-1] == s2[j-1])
else dp[i][j] = max(dp[i-1][j], dp[i][j-1]
# python
class Solution(object):
    def longestCommonSubsequence(self, text1, text2):
        if not text1 or not ext2:
            return 0
        m = len(text1)
        n = len(text2)
        dp = [[0] * (n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

2. 最长子串
dp[i][j] = dp[i-1][j-1] + 1 (if s1[i-1] == s2[j-1])
else dp[i][j] = 0
与上面的代码类似，else修改下
3. 编辑距离
见上节介绍和powcai思路

4. 最长回文
1. 暴力,嵌套循环，枚举i, j (起点 终点)，判断是否是回文 O(n^3)
2. 中间向两边扩张法 O(n^2)    参考windliang解法4扩展中心
3. 动态规划
首先定义P(i,j):
P(i,j) = true s[i,j]是回文串 
         false s[i,j]不是回文串
接下来
P(i,j)=(P(i+1,j-1) && S[i]==S[j])

5. 正则匹配
参考labuladong题解 和 吴彦祖的题解

6. 不同的子序列
1.暴力递归
2.动态规划
dp[i][j]代表T前i字符串可以由s前j字符串组成最多个数。
动态方程：
当S[j] == T[i], dp[i][j] = dp[i-1][j-1] + dp[i][j-1]
当S[j] != T[i], dp[i][j] = dp[i][j-1]
参考powcai的解法

字符串匹配算法
1.暴力法（brute force）
2.Rabin-Karp算法
3.KMP算法

