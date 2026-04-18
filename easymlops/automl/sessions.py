"""
会话管理,管理会话的上下文信息
"""
import copy
import uuid


class LLMSessionManager(object):
    def __init__(self):
        self.context = {}  # id->state
        self.dependency_top2down = {}  # 父节点与子节点依赖关系
        self.dependency_down2top = {}  # 子节点与父节点依赖关系
        self.queue = []

    def register_state(self, pid, state):
        sid = self.add_state(state)
        if pid is not None:
            self.dependency_down2top[sid] = pid  # 每个子节点只有一个父节点
            if self.dependency_top2down.get(pid) is None:
                self.dependency_top2down[pid] = []
            self.dependency_top2down[pid].append(sid)
        return sid

    def add_state(self, state):
        sid = str(uuid.uuid4()).split("-")[0]
        self.queue.append(sid)
        self.context[sid] = copy.deepcopy(state)
        return sid

    def get_state(self, sid):
        return copy.deepcopy(self.context.get(sid))

    def get_state_in_queue(self, index):
        return copy.deepcopy(self.context.get(self.queue[index]))

    def plot_val_loss_in_queue(self, index):
        import matplotlib.pyplot as plt
        plt.plot(self.get_state_in_queue(index)["val_evaluate_value"][1:])

    def get_val_loss_in_queue(self, index):
        return self.get_state_in_queue(index)["val_evaluate_value"][1:]

    def get_parent_id(self, sid):
        return self.dependency_down2top.get(sid)

    def get_child_id(self, pid):
        return self.dependency_top2down.get(pid)

    def plot_dag_state(self):
        import networkx as nx
        dg = nx.DiGraph()
        for k, vs in self.dependency_top2down.items():
            for v in vs:
                dg.add_edge(k, v)
        nx.draw(dg, with_labels=True)
