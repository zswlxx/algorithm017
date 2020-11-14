class Trie:

    def __init__(self):
        self.lookup = {}

    def insert(self, word: str) -> None:
        tree = self.lookup
        for a in word:
            if a not in tree:
                tree[a] = {}
            tree = tree[a]
        tree["#"] = "#"

    def search(self, word: str) -> bool:
        tree = self.lookup
        for a in word:
            if a not in tree:
                return False
            tree = tree[a]
        if "#" in tree:
            return True
        return False

    def startWith(self, prefix: str) -> bool:
        tree = self.lookup
        for a in prefix:
            if a not in tree:
                return False
            tree = tree[a]
        return True
