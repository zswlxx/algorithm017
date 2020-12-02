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

1.打破自己的思维惯性，形成机器思维
2.理解复杂逻辑的关键
3.也是职业进阶的要点要领




