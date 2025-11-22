# Rill - AI Agent Framework

Rill 是一个用于构建 AI Agent 工作流的 Python 框架。

## 安装

### 从 GitHub 安装

```bash
pip install git+https://github.com/zhixiangxue/rill-ai.git@main
```

### 从本地安装（开发模式）

```bash
git clone https://github.com/zhixiangxue/rill-ai.git
cd rill-ai
pip install -e .
```

## 快速开始

```python
from rill import Flow, FlowState, node, END

class MyFlow(Flow):
    @node(start=True, goto=END)
    async def process(self, inputs):
        # 你的处理逻辑
        self.state.result = "done"
        return {"result": "done"}

# 运行流程
flow = MyFlow()
await flow.run()
```

## 主要特性

- 基于装饰器的节点定义
- 支持动态路由（DYNAMIC）
- 循环控制（max_loop）
- 节点执行统计
- 基于 Pydantic 的状态管理

## 依赖

- Python >= 3.8
- pydantic >= 2.0.0
- loguru >= 0.7.0
- chak

## License

查看 LICENSE 文件获取更多信息。
