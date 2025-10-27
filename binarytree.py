import matplotlib.pyplot as plt
from collections import deque

class BinarySearchTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def insert(self, value):
        if value < self.value:
            if self.left is None:
                self.left = BinarySearchTree(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = BinarySearchTree(value)
            else:
                self.right.insert(value)

    def search(self, value):
        if value == self.value:
            return True
        elif value < self.value and self.left:
            return self.left.search(value)
        elif value > self.value and self.right:
            return self.right.search(value)
        return False

    def delete(self, value):
        if value < self.value:
            if self.left:
                self.left = self.left.delete(value)
        elif value > self.value:
            if self.right:
                self.right = self.right.delete(value)
        else:
            # 삭제 대상 노드
            if not self.left:
                return self.right
            elif not self.right:
                return self.left

            # 오른쪽 서브트리에서 최소값 찾기
            temp = self.right
            while temp.left:
                temp = temp.left
            self.value = temp.value
            self.right = self.right.delete(temp.value)
        return self

    # inorder 출력
    def print_tree(self):
        if self.left:
            self.left.print_tree()
        print(self.value, end=" ")
        if self.right:
            self.right.print_tree()

    # 시각화 함수
    def visualize(self):
        positions = {}
        edges = []
        x_counter = [0]

        # 중위순회로 x좌표 배정
        def inorder(node, depth=0):
            if node is None:
                return
            inorder(node.left, depth + 1)
            positions[node] = (x_counter[0], -depth)
            x_counter[0] += 1
            inorder(node.right, depth + 1)

        inorder(self)

        # 간선 연결
        queue = deque([self])
        while queue:
            node = queue.popleft()
            if node.left:
                edges.append((node, node.left))
                queue.append(node.left)
            if node.right:
                edges.append((node, node.right))
                queue.append(node.right)

        # 그림 그리기
        plt.figure(figsize=(8, 5))
        for p, c in edges:
            x1, y1 = positions[p]
            x2, y2 = positions[c]
            plt.plot([x1, x2], [y1, y2])

        for node, (x, y) in positions.items():
            plt.scatter([x], [y], s=500, edgecolors="black", facecolors="white")
            plt.text(x, y, str(node.value), ha="center", va="center", fontsize=10)

        plt.title("Binary Search Tree Visualization")
        plt.axis("off")
        plt.show()


# 테스트
bst = BinarySearchTree(10)
bst.insert(6)
bst.insert(14)
bst.insert(3)
bst.insert(8)
bst.insert(12)
bst.insert(15)

bst.print_tree()
print("\n--- 시각화 ---")
bst.visualize()
