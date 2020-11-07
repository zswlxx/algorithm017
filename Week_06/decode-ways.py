class Solution:
    def numDecoding(self, s: str) -> int:
        size = len(s)
        if size == 0:
            return 0

        dp = [0 for _ in range(size)]

        if s[0] == '0':
            rerturn 0
        dp[0] = 1

        for i in range(1, size):
            if s[i] != '0':
                dp[i] = dp[i -1]

            num = 10 * (ord(s[i - 1]) -ord('0')) + (ord(s[i]) - ord('0'))

            if 10 <= num <= 26:
                if i == 1:
                    dp[i] += 1
                else:
                    dp[i] += dp[i - 2]
        return dp[size - 1]
