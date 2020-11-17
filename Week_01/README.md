学习笔记


时间复杂度：常见的有 O(1) O(logn)  O(n) O(n^2)  O(n^3) O(2^n)  O(n!) 在判断时只关注最高复杂度的运算，注意加法和乘法，递归注意画出递归树来计算

Array数组：向数组中插入元素时，需要移动n-i个元素；删除下标为i的元素时，需移动n-i-1个元素。
Array数组时间复杂度：prepend O(1)  append O(1) lookup O(1) insert O(n) delete O(n)
注意：正常情况下数组的prepend操作的时间复杂度是O(n)，但是可以进行特殊优化到O(1)。采用的方式是申请稍大一些的内存空间，然后在数组最开始预留一部分空间，然后prepend的操作则是把头下标前移一个位置即可。
Linked List链表：链表有单向、双向、回环。如LRU Cache。
Linked List时间复杂度：prepend O(1) append O(1) lookup O(n) insert O(1) delete O(1)
Skip List跳表：主要是建立多级索引。只能用于元素有序的情况。所以跳表对标的是平衡树（AVL Tree）和二分查找，是一种 插入/删除/搜索 都是O(log n)的数据结构。
最大的优势是原理简单、容易实现、方便扩展、效率更高。Redis、LevelDB使用。
如何给有序的链表加速？一维数据结构加速经常采用的方式就是升维变成二维（多一层附加信息）; 以空间换时间。
跳表查询的时间复杂度分析：n/2、n/4、n/8、第k级索引结点的个数就是n/(2^k)。
假设索引有h级，最高级的索引有2个结点。n/(2^h)=2,从而求得h=log2(n)-1。
在跳表中查询任意数据的时间复杂度就是O(logn),增加和删除时间复杂度也是O(logn)
空间复杂度是O(n)

练习步骤： 
1. 5-10分钟：读题和思考
2. 有思路：自己开始做和写代码；不然马上看题解
3. 默写背诵、熟练
4. 然后开始自己写（闭卷）

Q: 移动0
A: 1. loop, count zeros  
2. 开新数组，loop (题目要求不允许) 
3. 直接在数组中执行index （一位数组坐标变换 双指针 再开一个下标j，记录到非0元素的位置）
# Java
class Solution {
    public void moveZeroes(int[] nums) {
        int j = 0
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] != 0) {
                nums[j] = nums[i];
                if (i != j) {
                    nums[i] = 0;
                }
                j++;
            }
        }
    }
}
# python
# in-place
def moveZeroes(self, nums):
    zreo = 0 # records the position of "0"
    for i in xrange(len(nums)):
        if nums[i] != 0:
            nums[i], nums[zero] = nums[zero], nums[i]
            zero += 1
代码提交(AC)完了仅是完成了50%, 剩下的50%是feedback
五遍刷题法（五毒神掌）
首先看官方的题解；要多看代码，觉得别人的代码写的好，可以学习并改造。
最大的误区是刷题只刷一遍；核心思想是升维和空间换时间
题目地址去掉-cn，进入国际站，看投票最高的答案。
该题另外看下国际站的滚雪球讲解。

Q: 盛水最多的容器
A：1. 枚举：left bar x， right bar y, (x-y)* x *height_diff 因为是嵌套所以时间复杂度为O(n^2) 
遍历
class Solution {
    public int maxArea(int[] a) {
        int max = 0;
        for (int i = 0; i < a.length - 1; ++i) {
            for (int j = i + 1; j < a.length; ++j) {
                int area = (j - i) * Math.min(a[i], a[j]);
                max = Math.max(max, area);
            }
        }
        return max;
    }
}
2. 时间复杂度O(n), 左右边界 i, j, 向中间收敛
解析：如果左右边界选到最两边，宽度是最宽的，虽有优势，但高度就不一定是最高的。
因为现在宽度是最大了，想要找最高的棒子，往中间收即可。如果它的高度不及外面的棒子，那就不用再看了。
因为你的高度都不及我，你在里面所以你的宽度肯定也不及我，那你的面积肯定也不及我，那就不用看了，每次收敛只关注棒子最高的。
左边变量i,右边变量j，每次往中间开始收敛，然后计算它的高度，求一个高度相对高的棒子，然后算它的面积，最后i和j在中间相遇，就可以得到结果。
由简到难，由普通到高级
栈：先入后出，后入先出。
class Solution {
    public int maxArea(int[] a) {
        int max = 0;
        for(int i = 0, j = a.length - 1; i < j; ){
            int minHeight = a[i] < a[j] ? a[i ++] : a[j --];
            int area = (j - i +1) * minHeight;
            max = Math.max(max, area);
        }
        return max;
    }
}
Q: 爬楼梯问题
A: 1. 懵逼的时候：暴力？ 基本情况？
找 最近 重复子问题 计算机只认识  if else,  for  while, recursion
数学归纳法得出 f(n) = f(n-1) + f(n-2) ;  Fibonacci
# python
class Solution(object):
    def climbStairs(self, n):
        if (n <= 2): return n
        f1, f2, f3 = 1, 2, 3
        for i in range(3, n+1):
            f3 = f1 + f2
            f1 = f2
            f2 = f3
        return f3
所有的问题归结到一起 就是找重复性
Q: 两数之和
A: a, b --> a + b == target  两层循环判断是否存在
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int[] a = new int[2];
        int numsSize = nums.length;
        for (int i = 0; i < numsSize - 1; i++) {
            for (int j = i+1; j < numsSize; j++) {
                if (nums[i] + nums[j] == target) {
                    a[0] = i;
                    a[1] = j;
                    return a;
                }
            }
        }
        return new int[0];
    }
 }
Q: 三数之和(一定要滚瓜烂熟)
A: a + b = -c (target不止一个)
1. 暴力求解: 需要三重循环，主要排重
注意三个变量 i < nums.length-2 , j < nums.length-1, k < nums.length
2. 暴力 + Hash : 两重循环
3. 双指针夹逼法（关注灵魂画手图解）
注意： 双指针有两种情况1.快慢指针 2.在排好序的数组头尾两边
# python
def threeSum(self, nums):
    res = []
    nums.sort()
    for i in xrange(len(nums)-2):
        for i > 0 and nums[i] == nums[i-1]:
            continue
        l, r = i+1, len(nums)-1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l += 1
            elif s > 0:
                r -= 1
            else:
                res.append((nums[i], nums[l], nums[r]))
                while l < r and nums[l] == nums[l+1]:
                    l += 1
                while l < r and nums[r] == nums[r-1]:
                    r -= 1
                l += 1; r -= 1
    return res

Q:环形链表
A: 1. 暴力法，遍历链表，时间和空间都为O(n)，遍历过程中使用hash或set记录下访问过的所有节点，再看接下来的元素有没有出现在hash或set中，如果出现，则表示又走回原来的老节点了，就说明有环了。
2. 快慢指针（针对环形链表的套路），快指针每次动两步，慢指针每次一步，如果有重叠则有环。
所有Linked List题目解法非常固定，主要就是熟能生巧，没有很多算法的东西，关键是要熟悉怎么把next指针换过来换过去，和把prev指针换过来换过去，要熟悉的办法就是多做，没有任何巧妙的地方。
五遍刷题法，在面试前要把高频的Linked List刷一遍。很令人崩溃的地方是：算法虽然很简单很直接，但是代码一不小心就写的非常复杂。
栈Stack： FILO 先入后出，添加、删除皆为O(1)，查询O(n) 
队列Queue：先进先出，添加、删除皆为O(1),查询O(n)。

双端队列Deque(Double-End Queue)：两边都可Push和Pop，插入、删除为O(1),查询O(n)，因为元素没有顺序可言。
优先队列Priority Queue：根据一定规则排序,插入O(1),取出O(logn)按照元素优先级取出。底层具体实现的数据结构较为多样和复杂：heap堆、bst二叉搜索树、treap.
python工程代码直接使用collections.deque  。

如果一个东西，它具有所谓的最近相关性的话（洋葱一层一层），就可以用栈来解决。先来后到用队列。

Q: 有效的括号
A: 1. 暴力: 不断replace匹配的括号->""  替换为空,时间复杂度O(n^2)。a. ()[]{}  b. (([{}]))) 
2. Stack：所有括号问题，要想到栈，如果是左括号就压到栈里面去，如果是右括号就和栈顶元素进行匹配，匹配上就正负抵消，那栈顶元素移出栈，不然就是不合法，一直这么操作下去，直到最后整个栈为空说明完全匹配了。
且在每一步不会发生栈顶元素和扫描到的左括号或右括号匹配不上的情况。
# python
class Solution:
    def isValid(self, s: str) -> bool:
        dic = {'{': '}', '[': ']', '(': ')', '?': '?'}
        stack = ['?']
        for c in s:
            if c in dic: stack.append(c)
            elif dic[stack.pop()] != c: return False
        return len(stack) == 1
# python
class Solution:
    def isValid(self, s):
        stack = []
        dict = {"]":"[", "}":"{", ")":"("}
        for char in s:
            if char in dict.values():
                stack.append(char)
            elif char in dict.keys():
                if stack == [] or dict[char] != stack.pop():
                    return False
            else:
                return False
        return stack == []

Q: 最小栈
A: 使用两个栈，一个维护出入关系，一个维护最小的栈。
用栈实现队列，用队列实现栈，都是通过两个栈进行处理。

Q: 柱状图中最大的矩形
A: 1. 暴力求解，复杂度O(n^3),用两个变量:i枚举左边，j枚举右边,把左右边界求出来之后，再找左右边界里面最大的面积，就是最矮的高度是多少，然后算出它的左右边界的长度，再找i和j的区域里面的最小高度，然后算出面积，每次记得更新最大区域面积max area，最后要的结果就是在max area里取。用两个指针写两个循环
for i -> 0, n-2
    for j -> i+1, n-1
        (i, j) -> 最小高度, area
        update max-area
2. 暴力法2，只枚举棒子的高度，0，n-1,说明这根棒子作为它的高度，这个时候就要找到左边界（第一次比它小的）和右边界（右边棒子比它小），面积就是高度 * （有边界-左边界+1?）
for i -> 0, n-1:
    找到left bound, right bound
    area = height[i] * (right - left)
    update max-area
3. stack: 维护一个栈，从小到大排列。（看官方动画）

Q: 滑动窗口最大值  （只要遇到滑动窗口就用队列处理）
A: 1. 暴力 两层循环，时间复杂度O(nk)
2. deque (sliding window) 双端队列
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        res = []
        if nums != []:
            tmp_max = max(nums[0:k])
            for i in range(len(nums)-(k-1)):
                tmp_max = max(nums[i:i+k])
                res.append(tmp_max)
        return res
