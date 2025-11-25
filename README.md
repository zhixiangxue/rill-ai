<div align="center">

<a href="https://github.com/zhixiangxue/rill-ai"><img src="https://raw.githubusercontent.com/zhixiangxue/rill-ai/main/docs/assets/logo.png" alt="Rill Logo" width="120"></a>

[![PyPI version](https://badge.fury.io/py/rillpy.svg)](https://badge.fury.io/py/rillpy)
[![Python Version](https://img.shields.io/pypi/pyversions/rillpy)](https://pypi.org/project/rillpy/)
[![License](https://img.shields.io/github/license/zhixiangxue/rill-ai)](https://github.com/zhixiangxue/rill-ai/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/rillpy)](https://pypi.org/project/rillpy/)
[![GitHub Stars](https://img.shields.io/github/stars/zhixiangxue/rill-ai?style=social)](https://github.com/zhixiangxue/rill-ai)

**A zero-dependency flow orchestration library for building AI workflows your way.**

**Minimal orchestration that lets you use any LLM client, any tools, any storage to build your AI agent applications.**

</div>

---

## Why Rill?

You don't need heavy frameworks to build AI agents. Sometimes you just need simple orchestration:

- Want to use your own LLM client (chak / OpenAI SDK / Anthropic SDK)? ✅
- Want to use your own tools (functions / MCP servers / custom implementations)? ✅
- Want to keep your codebase lightweight and dependencies minimal? ✅
- Prefer code over YAML/JSON configs? Code is the orchestration. ✅

**Rill just handles orchestration** - bring your own pieces, Rill manages the flow.

---

## Core Concepts

Rill draws inspiration from CrewAI's **"code is flow"** approach - use decorators to define nodes, use Python functions to express logic, no YAML or JSON configs needed.

Built on this foundation, Rill adds:

1. **Direct routing**: Declare next steps with `goto` right where you are, not reverse subscription
2. **No dependencies**: Framework handles orchestration only, everything else is your call

*Thanks to CrewAI and LangGraph for the inspiration.*

---

## Quick Start

### Installation

```bash
# From PyPI
pip install rillpy

# From GitHub
pip install git+https://github.com/zhixiangxue/rill-ai.git@main

# Local development
git clone https://github.com/zhixiangxue/rill-ai.git
cd rill-ai
pip install -e .
```

## Core Features

### 🌱 No Lock-in

Framework doesn't care what LLM, tools, or databases you use. Only provides orchestration.

### 🪴 Code is Flow

Define nodes with `@node` decorator, declare routing with `goto`, no DSL needed.

### 🌻 Direct Routing

Use `goto(next, data)` to directly specify next step in current node. Matches how humans think.

### 🌾 List Means Parallel

`goto([A, B], data)` automatically triggers `asyncio.gather` for parallel execution.

### 🌿 Auto Input Merge

When multiple predecessors point to same node, framework auto-merges outputs as `{pred_name: output}` dict.

### 🍀 Safety Guards

Graph validation (start point / cycles / reachability) + `max_loop` to prevent infinite loops.

### 🌲 Observable

`Flow.stats()` tracks node execution time, `logger` traces execution flow.

### 🪴 Shared State Management

Rill provides `FlowState` for sharing data across nodes. Simpler than LangGraph's in-node state updates.

**TODO**: Parallel nodes updating state simultaneously may have thread-safety issues. Community contributions welcome.

### 🌾 Return Value as Input

Predecessor node's return value becomes successor node's input parameter. No need to put everything in state.

---

## Common Patterns

### Conditional Branching + Parallel Execution

```python
class ResearchFlow(Flow):
    @node(start=True, goto=DYNAMIC)
    async def decide(self, topic):
        complexity = await self.analyze_complexity(topic)
        
        if complexity > 0.8:
            # High complexity: parallel deep research
            return goto([self.academic_search, self.expert_interview], topic)
        else:
            # Low complexity: quick search
            return goto(self.web_search, topic)
    
    @node(goto="synthesize")
    async def academic_search(self, topic):
        return await search_papers(topic)
    
    @node(goto="synthesize")
    async def expert_interview(self, topic):
        return await interview_experts(topic)
    
    @node(goto="synthesize")
    async def web_search(self, topic):
        return await search_web(topic)
    
    @node()
    async def synthesize(self, sources):
        # Auto-merge: sources could be {"academic_search": ..., "expert_interview": ...}
        # or just web_search output (single predecessor)
        return await generate_report(sources)
```

### Loop + Exit Condition

```python
class IterativeFlow(Flow):
    @node(start=True, goto=DYNAMIC, max_loop=5)
    async def generate(self, prompt):
        result = await llm_generate(prompt)
        quality = await self.evaluate(result)
        
        if quality > 0.9:
            return goto(self.finalize, result)
        else:
            # Loop back with feedback
            return goto(self.generate, {"prompt": prompt, "feedback": quality})
    
    @node()
    async def finalize(self, result):
        return result
```

### Using Shared State

```python
class MyWorkflow(Flow):
    @node(start=True, goto=["fetch_data", "process_config"])
    async def begin(self, inputs):
        # Store inputs in state for other nodes to access
        self.state.user_id = inputs["user_id"]
        self.state.query = inputs["query"]
        self.state.results = []  # Initialize shared collection
    
    @node(goto="merge")
    async def fetch_data(self, previous_result):
        # Access state from parallel node
        data = await api_call(self.state.user_id, self.state.query)
        
        # Accumulate results in state
        self.state.results.append({"source": "api", "data": data})
        return data
    
    @node(goto="merge")
    async def process_config(self, previous_result):
        # Another parallel node accessing same state
        config = load_config(self.state.user_id)
        
        # Also update shared state
        self.state.config = config
        return config
    
    @node()
    async def merge(self, inputs):
        # inputs = {"fetch_data": ..., "process_config": ...}
        # state contains accumulated data from all nodes
        final_result = combine(
            inputs["fetch_data"],
            inputs["process_config"],
            self.state.results  # Access shared state
        )
        
        self.state.final_output = final_result
        return final_result

# Run the workflow
flow = MyWorkflow()
final_state = await flow.run({
    "user_id": 123,
    "query": "hello"
})  # 🎯 Rill auto-converts your dict to a Pydantic FlowState object!

# Access final state (Pydantic model)
print(final_state.final_output)  # 🎉 Flow.run() returns the final FlowState
print(final_state.user_id)        # Access any field stored during execution
print(final_state.results)         # All accumulated data persists here
```

**State vs Return Value:**
- **Return value**: Direct data passing from predecessor to successor (the main data pipeline)
- **State**: Shared context accessible from any node (for metadata, counters, cross-branch data)
- **Key difference**: Return values flow through edges, state persists across the entire workflow
- Use return values for primary data flow, use state for auxiliary data that multiple nodes need

**Known Issue:**
- ⚠️ Parallel nodes updating state simultaneously may cause race conditions
- 🔧 TODO: Need thread-safe state update mechanism (community contributions welcome)

---

## When to Use Rill?

| Your Situation | Recommendation |
|----------------|----------------|
| Quick GPT app, don't want to manage anything | 👉 LangChain / LangGraph (all-in-one convenience) |
| Want to use my own LLM client (chak / OpenAI SDK) + custom tools | 👉 **Rill** (orchestration flexibility) |
| Just want orchestration only, pick other components myself | 👉 **Rill** |

---

## FAQ

**Q: What's the difference between Rill and LangGraph?**  
A: LangGraph is a complete package (orchestration + LLM + tools + memory), Rill only handles orchestration, other components are up to you.

**Q: I'm already using LangChain Tools, can I use Rill?**  
A: Yes! Rill doesn't care where your tools come from, just call them directly in nodes.

**Q: Does Rill support state persistence?**  
A: Current `FlowState` is in-memory state (Pydantic model), persistence is your choice (Redis / PostgreSQL / files), no tie to any storage.

**Q: I want to use my own LLM client (e.g., chak), how to integrate?**  
A: Just `import chak` in nodes and call it, Rill doesn't care which LLM you use. Example:
```python
@node(start=True, goto="process")
async def query(self, user_input):
    from chak import Conversation  # Your LLM client
    conv = Conversation("openai/gpt-4o-mini", api_key="YOUR_KEY")
    return await conv.asend(user_input)
```

**Q: When do I need `max_loop`?**  
A: When your flow has cycles (e.g., "generate → evaluate → regenerate"), use `max_loop` to limit loop iterations and prevent infinite loops.

**Q: How does input merging work?**  
A: When multiple predecessor nodes point to the same target node, and all predecessors complete, the framework merges their outputs as a dict `{pred_name: output}` and passes it to the target node.

**Q: What's the difference between state and return value?**  
A: They serve different purposes:
- **Node return value**: Passes data to the next node(s) through the flow edge. This is the main data pipeline.
- **State (`self.state`)**: A shared Pydantic object accessible from all nodes throughout the workflow. Use it for metadata, counters, configuration, or data that multiple branches need to access.
- **Example**: Return the processed result to next node, but store statistics/metadata in state.

**Q: Is state update thread-safe in parallel nodes?**  
A: Not yet. Parallel nodes updating state simultaneously may cause race conditions. This is a known TODO. For now, avoid state updates in parallel nodes or use return values instead.

---

## API Reference

### `Flow`

Main workflow class, inherit to define your workflow.

```python
class MyFlow(Flow):
    def __init__(self, initial_state=None, max_steps=1000, validate=True):
        super().__init__(initial_state, max_steps, validate)
```

- `initial_state`: Initial state dict or Pydantic model
- `max_steps`: Max execution steps (prevent infinite loops)
- `validate`: Whether to validate graph before execution

### `@node`

Decorator to mark methods as workflow nodes.

```python
@node(start=False, goto=None, max_loop=None)
def my_node(self, inputs):
    pass
```

- `start`: Whether this is the start node
- `goto`: Next node(s), can be:
  - `None`: No successors (end node)
  - `"node_name"`: Single next node
  - `["node1", "node2"]`: Multiple nodes (parallel execution)
  - `DYNAMIC`: Runtime-determined routing (must return `goto(...)` in node)
- `max_loop`: Max loop count for this node (for cycle detection)

### `goto(target, data)`

Choose next node for DYNAMIC routing.

```python
@node(goto=DYNAMIC)
async def decide(self, inputs):
    if condition:
        return goto(self.next_node, data)
    else:
        return goto([self.task_a, self.task_b], data)  # Parallel
```

- `target`: Single node or list of nodes (list triggers parallel execution)
- `data`: Payload passed to target node(s)

### `DYNAMIC`

Use this for dynamic routing decisions. Use with `goto()`.

### `FlowState`

Shared state object (Pydantic model).

```python
# Access state from any node
self.state.custom_field = "value"  # Runtime field injection
self.state.user_id = 123
self.state.results = []  # Shared collection

# Two independent data channels:
# 1. Return value: flows through edges (node → successor)
# 2. State: shared context persists across entire workflow
```

**Known Issue**: Parallel nodes updating state simultaneously may cause race conditions (TODO).

### `Flow.run(initial_input)`

Execute the flow.

```python
result_state = await flow.run({"user_input": "Hello"})
```

### `Flow.stats()`

Get execution statistics.

```python
stats = flow.stats()
# {
#     "timing": {
#         "total_duration": 2.35,
#         "nodes": {
#             "query": {"duration": 1.2, "percentage": 51.06},
#             "search": {"duration": 0.8, "percentage": 34.04}
#         }
#     }
# }
```

---

## Architecture

```
┌─────────────────────────────────────────┐
│      Your Application                   │
│  LLM: chak / OpenAI / Anthropic / ...   │
│  Tools: MCP / Functions / LangChain     │
│  Storage: ChromaDB / PostgreSQL / Redis │
└──────────────┬──────────────────────────┘
               │ Use Rill for orchestration
┌──────────────▼──────────────────────────┐
│         Rill Orchestration              │
│  @node + goto + parallel + State        │
└─────────────────────────────────────────┘
```

---

## Dependencies

- Python >= 3.8
- pydantic >= 2.0.0
- loguru >= 0.7.0

---

## License

MIT License - see LICENSE file for details.

<div align="right"><a href="https://github.com/zhixiangxue/rill-ai"><img src="https://raw.githubusercontent.com/zhixiangxue/rill-ai/main/docs/assets/logo.png" alt="Demo Video" width="120"></a></div>