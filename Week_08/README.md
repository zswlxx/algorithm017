学习笔记

位运算
机器里的数字表示方式和存储格式就是二进制
位运算符：  左移 << 0011 => 0110 ; 右移  >> 0110 => 0011
按位或(只要有一个二进制位是1就是1) 0011 | 1011 => 1011
按位与(只要有一个二进制位是0就是0) 0011 & 1011 => 0011
按位取反(0变1，1变0) ~ 0011 => 1100
按位异或（相同为零不同为一） 0011 ^ 1011 => 1000
异或 XOR 的一些操作：
x ^ 0 = x
x ^ 1s = ~x 注意 1s = ~0  , 1s表示全1
x ^ (-x) = 1s
x ^ x = 0
c = a ^ b => a ^ c = b, b ^ c = a //交换两个数
a ^ b ^ c = a ^ (b ^ c) = (a ^ b) ^ c // associative

指定位置的位运算
将x最右边的n位清零：x & (~0 << n)
获取x的第n位值（0或者1）： （x >> n & 1
获取x的第n位的幂值： x & (1 << n)
仅将第n位置为1: x | (1 << n)
仅将第n位置为0: x & (~ (1 << n))
将x最高位至第n位(含)清零: x & ((1 << n)-1)

实战位运算要点
判断奇偶：x % 2 == 1 --> (x & 1) == 1 ;  x % 2 == 0 --> (x & 1) == 0
x >> 1 --> x / 2 即 x = x / 2; --> x = x >> 1; mid = (left + right) / 2 ; --> mid = (left + right) >> 1;
x = x & (x - 1) 清零最低位的1
x & -x => 得到最低位的1
x & ~x => 0

Q: 位1的个数
A: 1. for loop:  0 --> 32
2. %2,  /2
3. &1 , x = x >> 1; (32次)
4. while (x > 0) { count ++; x = x & (x - 1); } （循环x次）
看下官方题解 和 江五渣题解

Q: 2的幂数
A: 有且仅有一位二进制位是1
# python
class Solution(object):
    def isPowerOfTwo(self, n):
        return n != 0 and (n & (n - 1)) == 0

Q: 颠倒二进制位
A: int --> "010101" string --> reverse --> int   这种严重不推荐，慢并且不符合计算机运算方式
int --> 位运算 -->  
参考 Kimi的JavaScript的解法

Q: N皇后II
A: 现在的要求是要非常熟练地可以用数组来进行判重 
推荐国际站上的python解法
现在使用位运算进行加速，位运算取代的就是这三个判重的数组，用三个int的二进制位表示相应的列撇捺有没有被占据
# python 位运算终极解决方案
def totalNQueens(self, n):  # 入口函数
    if n < 1: return []
    self.count = 0
    self.DFS(n, 0, 0, 0, 0)
    return self.count

def DFS(self, n, row, cols, pie, na):
    # recursion terminator
    if row >= n:
        self.count += 1
        return

    bits = (~(cols | pie | na)) & ((1 << n) - 1) #得到当前所有的空位

    while bits:
        p = bits & -bits # 取到最低位的1
        bits = bits & (bits - 1) # 表示在p位置上放入皇后
        # 下探
        self.DFS(n, row + 1, cols | p, (pie | p) << 1, (na | p) >> 1)
        # 不需要revert cols, pie, na的状态 没有改变他们的值

# Java 位运算最佳
class Solution {
    private int size;
    private int count;

    private void solve(int row, int ld, int rd) {
        if (row == size) {
            count++;
            return ;
        }
        int pos = size & (~(row | ld | rd));
        while (pos != 0) {
            int p = pos & (-pos);
            pos -= p; // pos &= pos -1;
            solve(row | p, (ld | p) << 1, (rd | p) >> 1);
        }
    }

    public int totalNQueens(int n) {
        count = 0;
        size = (1 << n) - 1;
        solve(0, 0, 0);
        return count;
    }
}

Q: 比特位计数
A: 参考官方题解
# C++ 最佳解法
vector<int> countBits(int num) {
    vector<int> bits(num+1, 0);
    for (int i = 1; i <= nums; i++) {
        bits[i] += bits[i & (i - 1)] + 1;
    }
    return bits;
}

布隆过滤器Bloom Filter
回顾 HashTable + 拉链存储重要元素 （不仅需要使用哈希函数获取index，还要把要存的元素全部都放在哈希表里面，这是一个没有误差的数据结构，且有多少个元素，每个元素有多大，那么所有的这些元素需要占的内存空间在哈希表里面都要找相应的内存的大小给存进来）

Bloom Filter vs Hash Table
一个很长的二进制向量和一系列随机映射函数。布隆过滤器可以用于检索一个元素是否在一个集合中。
优点是空间效率和查询时间都远远超过一般的算法；
缺点是有一定的误识别率和删除困难。
当布隆过滤器把元素全部都插入完了之后，对于测试元素就是对于新来的一个元素，要来验证它是否存在的话，当它验证这个元素所对应的二进制位是1的时候，我们只能说它可能存在在布隆过滤器里面，但是当这个元素所对应的二进制位只要有一个不为1，那么我们可以百分之百肯定它不在。

# 布隆过滤器python简单实现
for bitarray import bitarray
import mmh3

class BloomFilter:
    def __int__(self, size, hash_num):
        self.size = size
        self.hash_num = hash_num
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def add(self, s):
        for seed in range(self.hash_num):
            result = mmh3.hash(s, seed) % self.size
            self.bit_array[result] = 1

    def lookup(self, s):
        for seed in range(self.hash_num):
            result = mmh3.hash(s, seed) % self.size
            if self.bit_array[result] == 0:
                return "Nope"
        return "Probably"

    bf = BloomFilter(500000, 7)
    bf.add("dantezhao")
    print (bf.lookup("dantezhao"))
    print (bf.lookup("yyj"))

LRU Cache
两个元素：大小、替换策略
实现方式：Hash Table + Double LinkedList 查询、修改、删除时间复杂度O(1)

Q: LRU Cache实现
A: 看官方题解 （最正统的是自己实现，哈希表和双端队列）

# LRU Cache Python实现
class LRUCache(object):

    def __init__(self, capacity):
        self.dic = collections.OrderedDict()
        self.remain = capacity

    def get(self, key):
        if key not in self.dic:
            return -1
        v = self.dic.pop(key)
        self.dic[key] = v # key as the newest one
        return v

    def put(self, key, value):
        if key in self.dic:
            self.dic.pop(key)
        else:
            if self.remian > 0:
                self.remian -= 1
            else:   # self.dic is full
                self.dic.popitem(last = False)
            self.dic[key] = value

多多练习整个链表的操作

排序算法
1. 比较类排序：
通过比较来决定元素间的相对次序，由于其时间复杂度不能突破O(nlogn),因此也称为非线性四件比较类排序
2. 非比较类排序：
不通过比较来决定元素间的相对次序，它可以突破基于比较排序的时间下界，以线性时间运行，因此也称为线性时间非比较类排序。
缺点是一般只能用于整型相关的数据类型，也就是说对于一些比如说字符串的排序或者是对象之间的排序就无能为力了，同时它一般需要辅助用额外的内存空间。

比较类排序：
交换排序：冒泡排序、快速排序
插入排序：简单插入排序、希尔排序
选择排序：简单选择排序、堆排序
归并排序：二路归并排序、多路归并排序

非比较排序：
计数排序
桶排序
基数排序

初级排序 O(n^2)
1. 选择排序（Selection Sort）
每次找最小值，然后放到待排序数组的起始位置。
2. 插入排序 (Insertion Sort)
从前到后逐步构建有序序列；对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。
3. 冒泡排序（Bulle Sort, 在实际中基本不常用）
嵌套循环，每次查看相邻的元素，如果逆序，则交换。

高级排序O(N * LogN)
1. 快速排序（Quick Sort）重点
数组取标杆pivot,将小元素放pivot左边，大元素放右侧，然后依次对右边和右边的子数组继续快排，以达到整个序列有序。
注意： 正常情况下数组的prepend操作的时间复杂度是O(n)，但是可以进行特殊优化到O(1)。采用的方式是申请稍大一些的内存空间，然后在数组最开始预留一部分空间，然后prepend的操作则是把头下标前移一个位置即可。
2. 归并排序（Merge Sort）重点  - 分治
1. 把长度为n的输入序列分成两个长度为n/2的子序列
2. 对这两个子序列分别采用归并排序；
3. 将两个排序好的子序列合并成一个最终的排序序列

归并和快排具有相似性，但步骤顺序相反
归并：先排序左右子数组，然后合并两个有序子数组
快排：先调配出左右子数组，然后对于左右子数组进行排序

3. 堆排序 （Heap Sort）- 堆插入O(logN),取最大/最小值O(1)
1. 数组元素依次建立小顶堆
2. 依次取堆顶元素，并删除


# 快排 Java实现
public static void quickSort(int[] array, int begin, int end) {
    if (end <= begin) return;
    int pivot = partition(array, begin, end);
    quickSort(array, begin, privot - 1);
    quickSort(array, pivot + 1, end);
}
static int partition(int[] a, int begin, int end) {
    // pivot: 标杆位置, counter: 小于pivot的元素的个数
    int pivot = end, counter = begin;
    for (int i = begin; i < end; i++) {
        if (a[i] < a[pivot]) {
            int temp = a[counter]; a[counter] = a[i]; a[i] = temp;
            counter++;
        }
    }
    int temp = a[pivot]; a[pivot] = a[counter]; a[counter] = temp;
    return counter;
}

调用方式：quickSort(a, 0, a.length - 1)

# 归并排序 Java实现 当成模板记忆
public static void mergeSort(int[] array, int left, int right) {
    if (right <= left) return;
    int mid = (left + right) >> 1; // (left + right) / 2

    mergeSort(array, left, mid);
    mergeSort(array, mid + 1, right);
    merge(array, left, mid, right);
}

public static void merge(int[] arr, int left, int mid, int right) {
    int[] temp = new int[right - left + 1]; //中间数组
    int i = left, j = mid + 1, k = 0;
    // 两个数组有序的合并在一起，它永远是这种while三段式
    while (i <= mid && j <= right) {
        temp[k++] = arr[i] <= arr[j] ? arr[i++] : arr[j++];
    }

    while (i <= mid)   temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];

    for (int p = 0; p < temp.length; p++) {
        arr[left + p] = temp[p];
    }
    // 也可以用System.arraycopy(a, start1, b, start2, length)
}

堆排序代码 C++实现，建议直接调用priority_queue
void head_sort(int a[], int len) {
    priority_queue<int, vector<int>, greater<int> > q;

    for (int i = 0; i < len; i++) {
        q.push(a[i]);
    }
    for (int i = 0; i < len; i++) {
        a[i] = q.pop();
    }
}

特殊排序 - O(n) 代码实现不做要求
1. 计数排序（Counting Sort）
计数排序要求输入的数据必须是有确定范围的正数。将输入的数据值转化为键存储在额外开辟的数组空间中；然后依次把计数大于1的填充回原数组。
2. 桶排序（Bucket Sort）
假设输入数据服从均匀分布，将数据分到有限数量的桶里，每个桶再分别排序（有可能再使用别的排序算法或是以递归方式继续使用桶排序进行排）
3. 计数排序（Radix Sort）不支持浮点数
按照低位先排序，然后收集；再按照高位排序，然后再收集；依次类推，直到最高位。有时候有些属性是有优先级顺序的，先按低优先级排序，再按高优先级排序。

Q: 有效的字母异位词
A: 看官方题解

Q: 合并区间
A: 推荐吴彦祖解法 或官方解法的方法二排序

Q: 翻转对 (对应逆序对 重点)
A: 1. 暴力：两个嵌套循环 一个枚举起点，一个它后面那个数  O(n^2)
2. merge-sort  O(nlogn) 推荐  看下国际站解法zhugejunwei写的最好
3. 树状数组


