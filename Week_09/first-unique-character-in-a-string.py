class Solution:
    def firstUniqChar(self, s: str) -> int:
        count = collections.Counter(s)

        for idx, cha in enumerate(s):
            if count[ch] == 1:
                return idx
        return -1
