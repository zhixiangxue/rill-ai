"""DAG-based flow orchestration with parallel execution and dynamic routing.

Typical usage:
    class MyFlow(Flow):
        @node(start=True, goto="process")
        def start(self, inputs):
            return {"data": inputs}
        
        @node(goto=DYNAMIC)
        async def process(self, inputs):
            if needs_parallel:
                return goto([self.taskA, self.taskB], inputs)  # parallel
            return goto(self.finalize, inputs)  # single path
        
        @node()
        def finalize(self, merged):
            return "done"
    
    await MyFlow().run(initial_data)
"""
from typing import Any, Callable, Dict, List, Union, Optional, Literal, Set
import inspect
import asyncio
import sys
from pydantic import BaseModel
from dataclasses import dataclass

from .utils.logger import logger

# Interned constants for routing control
DYNAMIC = sys.intern("__dynamic__")  # Runtime-determined routing
END = sys.intern("__end__")          # Terminal state marker

@dataclass
class Route:
    """Encapsulates routing decision for DYNAMIC nodes."""
    to: Union[Callable, str, List[Union[Callable, str]]]  # Target(s): method ref, name, or list
    data: Any = None  # Payload forwarded to target(s)
    
    def resolve(self) -> tuple:
        """Normalize targets to name strings.
        
        Returns:
            (str, data) for single target or (List[str], data) for parallel targets
        """
        if isinstance(self.to, list):
            names = [t.__name__ if callable(t) else t for t in self.to]
            return names, self.data
        if callable(self.to):
            return self.to.__name__, self.data
        return self.to, self.data

def goto(target: Union[Callable, str, List[Union[Callable, str]]], data: Any = None) -> Route:
    """Construct routing decision for DYNAMIC nodes.
    
    Args:
        target: Single node or list of nodes (triggers parallel execution for lists)
        data: Payload passed to target(s); duplicated for parallel targets
    
    Returns:
        Route object for flow execution engine
    
    Example:
        @node(goto=DYNAMIC, max_loop=3)
        async def evaluate(self, inputs):
            if quality_ok:
                return goto(self.finalize, result)
            else:
                return goto([self.research, self.notify], feedback)  # parallel dispatch
    """
    return Route(to=target, data=data)

class FlowState(BaseModel):
    """Shared mutable state container with Pydantic validation."""
    model_config = {"extra": "allow"}  # Permits runtime field injection

class Node:
    """Decorator marking methods as executable flow nodes."""
    def __init__(self, 
                 start: bool = False,
                 goto: Union[str, Callable, List[Union[str, Callable]], Literal["__dynamic__"], None] = None,
                 max_loop: Optional[int] = None):
        self.start = start
        self.max_loop = max_loop
        self.goto = self._normalize_goto(goto)
    
    def _normalize_goto(self, goto: Union[str, Callable, List[Union[str, Callable]], Literal["__dynamic__"], None]) -> Union[str, List[str], Literal["__dynamic__"], None]:
        """Convert callable references to name strings for internal routing."""
        if goto is None or goto is DYNAMIC:
            return goto  # type: ignore
        elif isinstance(goto, str):
            return goto
        elif callable(goto):
            return goto.__name__
        elif isinstance(goto, list):
            return [g.__name__ if callable(g) else g for g in goto]
        return goto  # type: ignore
    
    def __call__(self, func):
        func._is_flow_node = True
        func._start_node = self.start
        func._goto_targets = self.goto
        func._max_loop = self.max_loop
        return func

node = Node  # Pythonic lowercase alias

class Flow:
    """Orchestrates DAG execution with automatic parallelization and input merging."""
    def __init__(self, 
                 initial_state: Optional[Union[Dict[str, Any], BaseModel]] = None,
                 max_steps: int = 1000,
                 validate: bool = True):
        if isinstance(initial_state, BaseModel):
            self.state = initial_state
        elif isinstance(initial_state, dict):
            self.state = FlowState(**initial_state)
        else:
            self.state = FlowState()
        
        self.max_steps = max_steps  # Prevent infinite loops in malformed graphs
        self._nodes = {}
        self._loop_counters = {}  # Tracks per-node execution count
        self._node_timings = {}  # Profiling data: {node: {start, end, duration}}
        self._flow_start_time = 0.0
        self._flow_end_time = 0.0
        self._collect_nodes()
        
        if validate:
            self._validate_graph()
    
    def _collect_nodes(self):
        """Scan instance for @node-decorated methods."""
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, '_is_flow_node'):
                self._nodes[name] = method
    
    def _build_execution_graph(self):
        """Build reverse dependency map for multi-input node detection.
        
        Returns graph where keys are target nodes and values are lists of 
        predecessor nodes pointing to them (enables input merging logic).
        """
        graph = {}
        
        for node_name, node_method in self._nodes.items():
            goto_targets = getattr(node_method, '_goto_targets', None)
            
            if goto_targets:
                target_nodes = []
                if isinstance(goto_targets, str):
                    target_nodes = [goto_targets]
                elif isinstance(goto_targets, list):
                    target_nodes = goto_targets
                elif isinstance(goto_targets, dict):
                    target_nodes = list(goto_targets.values())
                
                for target in target_nodes:
                    if target not in graph:
                        graph[target] = []
                    graph[target].append(node_name)
        
        return graph
    
    def _validate_graph(self):
        """Pre-flight DAG sanity checks to catch common errors early."""
        issues = []
        
        start_nodes = [n for n, m in self._nodes.items() 
                      if getattr(m, '_start_node', False)]
        if len(start_nodes) == 0:
            issues.append("❌ Error: No start node found (need @node(start=True))")
        elif len(start_nodes) > 1:
            issues.append(f"❌ Error: Multiple start nodes found {start_nodes}")
        
        cycles = self._detect_cycles()
        for cycle in cycles:
            nodes_without_limit = [n for n in cycle 
                                  if not getattr(self._nodes[n], '_max_loop', None)]
            if nodes_without_limit:
                issues.append(f"❌ Error: Cycle detected {cycle}, "
                            f"nodes {nodes_without_limit} need max_loop")
        
        # Skip unreachable check if DYNAMIC nodes exist (runtime-determined routing)
        dynamic_nodes = [n for n, m in self._nodes.items()
                        if getattr(m, '_goto_targets', None) is DYNAMIC]
        
        if start_nodes and not dynamic_nodes:
            # Only check reachability for fully static graphs
            reachable = self._get_reachable_nodes(start_nodes[0])
            unreachable = set(self._nodes.keys()) - reachable
            if unreachable:
                issues.append(f"⚠️  Warning: Unreachable nodes {unreachable}")
        
        # Note: max_loop is optional for DYNAMIC nodes (user's responsibility)
        # Removed the overly strict warning since simple DYNAMIC routing is safe
        
        if issues:
            error_msg = "DAG validation result:\n" + "\n".join(issues)
            errors = [i for i in issues if i.startswith("❌")]
            if errors:
                raise ValueError(error_msg)
            else:
                logger.debug(error_msg)
    
    def _detect_cycles(self) -> List[List[str]]:
        """DFS-based cycle detection using recursion stack tracking."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]):
            if node in rec_stack:
                if node in path:
                    cycle_start = path.index(node)
                    cycle = path[cycle_start:]
                    if cycle not in cycles:
                        cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            neighbors = self._get_neighbors(node)
            for neighbor in neighbors:
                dfs(neighbor, path.copy())
            
            rec_stack.remove(node)
        
        for node_name in self._nodes:
            if node_name not in visited:
                dfs(node_name, [])
        
        return cycles
    
    def _get_neighbors(self, node_name: str) -> List[str]:
        """Extract static outgoing edges (excludes DYNAMIC which is runtime-determined)."""
        node = self._nodes[node_name]
        goto = getattr(node, '_goto_targets', None)
        
        if not goto or goto is DYNAMIC:
            return []
        elif isinstance(goto, str):
            return [goto]
        elif isinstance(goto, list):
            return goto
        return []
    
    def _get_reachable_nodes(self, start: str) -> Set[str]:
        """BFS to find all nodes accessible from start (for dead code detection)."""
        reachable = set()
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)
            
            neighbors = self._get_neighbors(current)
            queue.extend(neighbors)
        
        return reachable
    
    async def run(self, initial_input: Any = None):
        """Execute DAG with automatic parallelization and input merging.
        
        Core execution model:
        - Nodes with multiple predecessors receive merged dict: {pred_name: output}
        - List targets trigger asyncio.gather for true parallel execution
        - DYNAMIC nodes decide routing at runtime via goto() return value
        """
        import time
        
        self._flow_start_time = time.time()
        
        logger.debug("========= Flow execution started ==========")
        logger.debug(f"Initial input: {initial_input}")
        
        execution_graph = self._build_execution_graph()
        logger.debug(f"Execution graph: {execution_graph}")
        
        start_nodes = [node for node in self._nodes.values() 
                      if getattr(node, '_start_node', False)]
        
        if not start_nodes:
            raise ValueError("No start node found")
        
        if len(start_nodes) > 1:
            raise ValueError("Multiple start nodes found")
        
        node_outputs = {}
        pending_nodes = {}
        
        queue = [(start_nodes[0].__name__, initial_input)]
        executed = set()
        step = 0
        
        while queue:
            step += 1
            
            if step > self.max_steps:
                raise RuntimeError(f"Flow exceeded max_steps limit ({self.max_steps}), possible infinite loop")
            
            current_node_name, current_input = queue.pop(0)
            
            if current_node_name in executed:
                continue
            
            current_node = self._nodes[current_node_name]
            
            max_loop = getattr(current_node, '_max_loop', None)
            if max_loop:
                self._loop_counters[current_node_name] = self._loop_counters.get(current_node_name, 0) + 1
                if self._loop_counters[current_node_name] > max_loop:
                    raise RuntimeError(f"Node '{current_node_name}' exceeded max_loop limit ({max_loop})")
                logger.debug(f"Loop counter: {current_node_name} [{self._loop_counters[current_node_name]}/{max_loop}]")
            
            logger.debug(f"--- Step {step} ---")
            logger.debug(f"Executing node: {current_node_name}")
            logger.debug(f"Input data: {current_input}")
            
            import time
            if current_node_name not in self._node_timings:
                self._node_timings[current_node_name] = {}
            self._node_timings[current_node_name]["start"] = time.time()
            
            result = await current_node(current_input) if asyncio.iscoroutinefunction(current_node) else current_node(current_input)
            
            import time
            end_time = time.time()
            self._node_timings[current_node_name]["end"] = end_time
            start_time = self._node_timings[current_node_name].get("start", end_time)
            self._node_timings[current_node_name]["duration"] = end_time - start_time
            
            logger.debug(f"Output result: {result}")
            
            node_outputs[current_node_name] = result
            executed.add(current_node_name)
            
            goto_targets = getattr(current_node, '_goto_targets', None)
            
            if not goto_targets:
                logger.debug(f"Node {current_node_name} has no successors")
                continue
            
            next_nodes = []
            next_input = result
            
            if goto_targets is DYNAMIC:
                if isinstance(result, Route):
                    resolved_target, next_input = result.resolve()
                    if isinstance(resolved_target, list):
                        next_nodes = resolved_target
                        logger.debug(f"DYNAMIC routing (parallel): {next_nodes}")
                    else:
                        next_nodes = [resolved_target]
                        logger.debug(f"DYNAMIC routing: {resolved_target}")
                elif isinstance(result, tuple) and len(result) == 2:
                    target, next_input = result
                    if callable(target):
                        next_node_name = target.__name__
                    else:
                        next_node_name = target
                    next_nodes = [next_node_name]
                    logger.debug(f"DYNAMIC routing: {next_node_name} (tuple format, recommend using goto())")
                else:
                    raise ValueError(
                        f"DYNAMIC node '{current_node_name}' must return:\n"
                        f"  - goto(target, data)  [recommended]\n"
                        f"  - (target, data)      [legacy format]"
                    )
            elif isinstance(goto_targets, list):
                next_nodes = goto_targets
                logger.debug(f"Parallel execution: {next_nodes}")
            elif isinstance(goto_targets, str):
                next_nodes = [goto_targets]
                logger.debug(f"Next node: {goto_targets}")
            else:
                raise ValueError(f"Unsupported goto type: {type(goto_targets)}")
            
            if len(next_nodes) > 1:
                logger.debug(f"Using asyncio.gather to execute {len(next_nodes)} nodes in parallel")
                
                import time
                
                async def execute_with_timing(node_name, node_func, node_input):
                    """Wrapper to capture execution metrics for parallel nodes."""
                    if node_name not in self._node_timings:
                        self._node_timings[node_name] = {}
                    self._node_timings[node_name]["start"] = time.time()
                    
                    if asyncio.iscoroutinefunction(node_func):
                        result = await node_func(node_input)
                    else:
                        result = node_func(node_input)
                    
                    end_time = time.time()
                    self._node_timings[node_name]["end"] = end_time
                    start_time = self._node_timings[node_name].get("start", end_time)
                    self._node_timings[node_name]["duration"] = end_time - start_time
                    
                    return result
                
                tasks = []
                for next_node_name in next_nodes:
                    next_node = self._nodes[next_node_name]
                    tasks.append(execute_with_timing(next_node_name, next_node, next_input))
                
                results = await asyncio.gather(*tasks)
                
                for node_name, node_result in zip(next_nodes, results):
                    logger.debug(f"  - {node_name} completed: {node_result}")
                    node_outputs[node_name] = node_result
                    executed.add(node_name)
                    
                    node_method = self._nodes[node_name]
                    node_goto = getattr(node_method, '_goto_targets', None)
                    
                    if node_goto:
                        if node_goto is DYNAMIC:
                            subsequent_nodes = []
                        elif isinstance(node_goto, str):
                            subsequent_nodes = [node_goto]
                        elif isinstance(node_goto, list):
                            subsequent_nodes = node_goto
                        else:
                            subsequent_nodes = []
                        
                        for sub_node in subsequent_nodes:
                            if sub_node in execution_graph:
                                predecessors = execution_graph[sub_node]
                                if all(pred in executed for pred in predecessors):
                                    merged_input = {pred: node_outputs[pred] for pred in predecessors}
                                    logger.debug(f"Node {sub_node}: all predecessors completed, merging inputs: {list(merged_input.keys())}")
                                    queue.append((sub_node, merged_input))
                            else:
                                queue.append((sub_node, node_result))
            else:
                for next_node_name in next_nodes:
                    if next_node_name in execution_graph:
                        predecessors = execution_graph[next_node_name]
                        if all(pred in executed for pred in predecessors):
                            if len(predecessors) > 1:
                                merged_input = {pred: node_outputs[pred] for pred in predecessors}
                                logger.debug(f"Node {next_node_name}: all predecessors completed, merging inputs: {list(merged_input.keys())}")
                                queue.append((next_node_name, merged_input))
                            else:
                                queue.append((next_node_name, next_input))
                        else:
                            waiting = [p for p in predecessors if p not in executed]
                            logger.debug(f"Node {next_node_name} waiting for predecessors: {waiting}")
                    else:
                        queue.append((next_node_name, next_input))
        
        import time
        self._flow_end_time = time.time()
        
        logger.debug("========== Flow execution completed ==========")
        logger.debug(f"Final state: {self.state}")
        return self.state
    
    def stats(self) -> dict:
        """Extract profiling metrics collected during flow execution.
        
        Returns dict with timing breakdown by node and total flow duration.
        Useful for identifying bottlenecks in complex workflows.
        """
        total_duration = self._flow_end_time - self._flow_start_time if self._flow_end_time > 0 else 0
        
        node_stats = {}
        for node_name, timing in self._node_timings.items():
            duration = timing.get("duration", 0)
            percentage = (duration / total_duration * 100) if total_duration > 0 else 0
            node_stats[node_name] = {
                "duration": round(duration, 2),
                "percentage": round(percentage, 2)
            }
        
        return {
            "timing": {
                "total_duration": round(total_duration, 2),
                "nodes": node_stats
            }
        }
