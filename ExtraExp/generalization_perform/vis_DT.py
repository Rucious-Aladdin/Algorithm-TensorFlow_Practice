from graphviz import Digraph
from decisionTree import DecisionTreeClassifier
import numpy as np
import sys
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\dataset")
from mnist import load_mnist
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    
    def fit(self, X, y):
        print("Building decision tree...")
        self.tree = self._build_tree(X, y, depth=0)
        print("Decision tree built.")
    
    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))
        
        # 종료 조건: 노드의 깊이가 max_depth에 도달하거나
        # 데이터 포인트 수가 min_samples_split보다 작을 때
        if (self.max_depth is not None and depth == self.max_depth) or \
           (num_samples < self.min_samples_split):
            print(f"Reached leaf node: depth={depth}, samples={num_samples}")
            return self._create_leaf_node(y)
        
        # 더 이상 분할할 수 없는 경우, 리프 노드 생성
        if num_classes == 1:
            print(f"Reached leaf node: depth={depth}, samples={num_samples}")
            return self._create_leaf_node(y)
        
        # 최적의 분할을 찾기 위해 속성과 분할 기준을 선택
        best_split_feature, best_split_threshold = self._find_best_split(X, y)
        
        # 분할이 유의미하지 않을 때, 리프 노드 생성
        if best_split_feature is None:
            print(f"Reached leaf node: depth={depth}, samples={num_samples}")
            return self._create_leaf_node(y)
        
        # 최적의 분할을 통해 노드 생성
        left_indices = X[:, best_split_feature] < best_split_threshold
        right_indices = ~left_indices
        print(f"Split node: depth={depth}, samples={num_samples}, feature={best_split_feature}, threshold={best_split_threshold}")
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return (best_split_feature, best_split_threshold, left_subtree, right_subtree)
    
    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        if num_samples <= 1:
            return None, None
        
        # 현재 노드의 Entropy 계산
        class_counts = np.bincount(y)
        node_entropy = self._calculate_entropy(class_counts)
        
        best_entropy = float('inf')
        best_split_feature = None
        best_split_threshold = None
        
        for feature in range(num_features):
            unique_values = np.unique(X[:, feature])
            
            for threshold in unique_values:
                left_indices = X[:, feature] < threshold
                right_indices = ~left_indices
                
                if np.sum(left_indices) < self.min_samples_leaf or np.sum(right_indices) < self.min_samples_leaf:
                    continue
                
                left_class_counts = np.bincount(y[left_indices])
                right_class_counts = np.bincount(y[right_indices])
                
                # 왼쪽과 오른쪽 자식 노드의 entropy 계산
                left_entropy = self._calculate_entropy(left_class_counts)
                right_entropy = self._calculate_entropy(right_class_counts)
                
                # 가중 평균 entropy 계산
                weighted_entropy = (np.sum(left_indices) / num_samples) * left_entropy + \
                                   (np.sum(right_indices) / num_samples) * right_entropy
                
                # 분할이 더 나은 경우 최적의 분할로 업데이트
                if weighted_entropy < best_entropy:
                    best_entropy = weighted_entropy
                    best_split_feature = feature
                    best_split_threshold = threshold
        
        return best_split_feature, best_split_threshold
    
    def _calculate_entropy(self, class_counts):
        # Entropy 계산
        class_probabilities = class_counts / np.sum(class_counts)
        entropy = -np.sum(class_probabilities * np.log2(class_probabilities + 1e-10))
        return entropy
    
    def _create_leaf_node(self, y):
        # 리프 노드 생성: 가장 빈도가 높은 클래스 선택
        class_counts = np.bincount(y)
        most_common_class = np.argmax(class_counts)
        return most_common_class
    
    def predict(self, X):
        if self.tree is None:
            raise ValueError("The model has not been trained yet.")
        
        return np.array([self._predict_tree(x) for x in X])
    
    def _predict_tree(self, x):
        node = self.tree
        while isinstance(node, tuple):
            feature, threshold, left_subtree, right_subtree = node
            if x[feature] < threshold:
                node = left_subtree
            else:
                node = right_subtree
        return node
    
    # 모델 저장
    def save_model(model, filename):
        with open(filename, 'wb') as file:
            pickle.dump(model, file)

    # 모델 로드
    def load_model(filename):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model

    def visualize_tree(self, filename):
        if self.tree is None:
            raise ValueError("The model has not been trained yet.")

        dot = Digraph(comment='Decision Tree')
        self._visualize_tree(self.tree, dot)
        dot.render(filename, view=True)

    def _visualize_tree(self, node, dot, parent_node=None, split_feature=None):
        if isinstance(node, tuple):
            feature, threshold, left_subtree, right_subtree = node
            node_id = str(id(node))

            if split_feature is not None:
                edge_label = f"{split_feature} <= {threshold}"
            else:
                edge_label = None

            dot.node(node_id, label=f"Feature {feature}\nThreshold {threshold}")
            
            if parent_node is not None and edge_label is not None:
                dot.edge(parent_node, node_id, label=edge_label)
            
            self._visualize_tree(left_subtree, dot, node_id, f"Feature {feature}")
            self._visualize_tree(right_subtree, dot, node_id, f"Feature {feature}")
        else:
            class_label = f"Class {node}"
            dot.node(str(id(node)), label=class_label)
            dot.edge(parent_node, str(id(node)), label=split_feature)
            



model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=2)
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=False)
x_train = x_train[:100]
t_train = t_train[:100]
x_test = x_test
t_test = t_test
"""
model.fit(x_train, t_train)
model.visualize_tree("decision_tree_visualization")
"""
with open("decisionTree4.pkl", "rb") as f:
    loaded_obj = pickle.load(f)

loaded_obj.visualize_tree("DT")
