from typing import Any, Callable, Dict, List, Union, Optional, Literal, Set
import inspect
import asyncio
import sys
from pydantic import BaseModel
from dataclasses import dataclass

from ..utils.logger import logger

# 特殊节点目标
DYNAMIC = sys.intern("__dynamic__")  # 运行时动态路由
END = sys.intern("__end__")          # 流程结束

@dataclass
class Route:
    """路由返回值，用于 DYNAMIC 节点"""
    to: Union[Callable, str]  # 目标节点（方法引用或字符串）
    data: Any = None          # 传递的数据
    
    def resolve(self) -> tuple:
        """解析为 (node_name, data)"""
        if callable(self.to):
            return self.to.__name__, self.data
        return self.to, self.data

def goto(target: Union[Callable, str], data: Any = None) -> Route:
    """创建路由结果，用于 DYNAMIC 节点的返回值
    
    Args:
        target: 目标节点（方法引用或字符串）
        data: 传递给下一节点的数据
    
    Returns:
        Route 对象
    
    Example:
        @node(goto=DYNAMIC, max_loop=3)
        async def evaluate(self, inputs):
            if quality_ok:
                return goto(self.finalize, result)
            else:
                return goto(self.research, feedback)
    """
    return Route(to=target, data=data)

class FlowState(BaseModel):
    """共享状态基类（基于 Pydantic）"""
    model_config = {"extra": "allow"}  # 允许动态添加字段

class Node:
    """节点装饰器"""
    def __init__(self, 
                 start: bool = False,
                 goto: Union[str, Callable, List[Union[str, Callable]], Literal["__dynamic__"], None] = None,
                 max_loop: Optional[int] = None):
        self.start = start
        self.max_loop = max_loop
        # 标准化 goto：将函数引用转为函数名字符串
        self.goto = self._normalize_goto(goto)
    
    def _normalize_goto(self, goto: Union[str, Callable, List[Union[str, Callable]], Literal["__dynamic__"], None]) -> Union[str, List[str], Literal["__dynamic__"], None]:
        """将函数引用转换为函数名"""
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

# 创建小写别名
node = Node

class Flow:
    """流程基类"""
    def __init__(self, 
                 initial_state: Optional[Union[Dict[str, Any], BaseModel]] = None,
                 max_steps: int = 1000,
                 validate: bool = True):
        # 统一转为 Pydantic 对象
        if isinstance(initial_state, BaseModel):
            self.state = initial_state
        elif isinstance(initial_state, dict):
            self.state = FlowState(**initial_state)
        else:
            self.state = FlowState()
        
        self.max_steps = max_steps  # 全局安全阀
        self._nodes = {}
        self._loop_counters = {}  # 循环计数器 {node_name: count}
        self._node_timings = {}  # 节点耗时统计 {node_name: {"start": float, "end": float, "duration": float}}
        self._flow_start_time = 0.0  # 流程开始时间
        self._flow_end_time = 0.0  # 流程结束时间
        self._collect_nodes()
        
        if validate:
            self._validate_graph()
    
    def _collect_nodes(self):
        """收集所有被@node装饰的方法"""
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, '_is_flow_node'):
                self._nodes[name] = method
    
    def _build_execution_graph(self):
        """构建执行图：分析哪些节点指向哪些节点"""
        graph = {}  # {node_name: [list of nodes that point to it]}
        
        for node_name, node_method in self._nodes.items():
            goto_targets = getattr(node_method, '_goto_targets', None)
            
            if goto_targets:
                # 收集所有目标节点
                target_nodes = []
                if isinstance(goto_targets, str):
                    target_nodes = [goto_targets]
                elif isinstance(goto_targets, list):
                    target_nodes = goto_targets
                elif isinstance(goto_targets, dict):
                    target_nodes = list(goto_targets.values())
                
                # 记录每个目标节点的前置节点
                for target in target_nodes:
                    if target not in graph:
                        graph[target] = []
                    graph[target].append(node_name)
        
        return graph
    
    def _validate_graph(self):
        """验证 DAG 合法性"""
        issues = []
        
        # 1. 检查起点
        start_nodes = [n for n, m in self._nodes.items() 
                      if getattr(m, '_start_node', False)]
        if len(start_nodes) == 0:
            issues.append("❌ 错误：没有起点节点（需要 @node(start=True)）")
        elif len(start_nodes) > 1:
            issues.append(f"❌ 错误：多个起点节点 {start_nodes}")
        
        # 2. 检查循环（没有 max_loop 的循环）
        cycles = self._detect_cycles()
        for cycle in cycles:
            # 检查循环中的节点是否有 max_loop
            nodes_without_limit = [n for n in cycle 
                                  if not getattr(self._nodes[n], '_max_loop', None)]
            if nodes_without_limit:
                issues.append(f"❌ 错误：检测到循环 {cycle}，"
                            f"节点 {nodes_without_limit} 需要设置 max_loop")
        
        # 3. 检查不可达节点
        if start_nodes:
            reachable = self._get_reachable_nodes(start_nodes[0])
            unreachable = set(self._nodes.keys()) - reachable
            if unreachable:
                issues.append(f"⚠️  警告：不可达节点 {unreachable}")
        
        # 4. 检查 DYNAMIC 节点的风险
        dynamic_nodes = [n for n, m in self._nodes.items()
                        if getattr(m, '_goto_targets', None) is DYNAMIC]
        for node_name in dynamic_nodes:
            if not getattr(self._nodes[node_name], '_max_loop', None):
                issues.append(f"⚠️  警告：DYNAMIC 节点 '{node_name}' 没有 max_loop，"
                            f"可能导致死循环")
        
        # 输出或抛出异常
        if issues:
            error_msg = "DAG 验证结果：\n" + "\n".join(issues)
            errors = [i for i in issues if i.startswith("❌")]
            if errors:
                raise ValueError(error_msg)
            else:
                logger.debug(error_msg)  # 仅警告
    
    def _detect_cycles(self) -> List[List[str]]:
        """检测所有循环"""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]):
            if node in rec_stack:
                # 找到循环
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
            
            # 获取邻居节点
            neighbors = self._get_neighbors(node)
            for neighbor in neighbors:
                dfs(neighbor, path.copy())
            
            rec_stack.remove(node)
        
        for node_name in self._nodes:
            if node_name not in visited:
                dfs(node_name, [])
        
        return cycles
    
    def _get_neighbors(self, node_name: str) -> List[str]:
        """获取节点的所有后继节点"""
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
        """从起点开始的所有可达节点"""
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
        """执行流程"""
        import time
        
        # 记录流程开始时间
        self._flow_start_time = time.time()
        
        logger.debug("========= 流程开始执行 ===========")
        logger.debug(f"初始输入: {initial_input}")
        
        # 构建执行图
        execution_graph = self._build_execution_graph()
        logger.debug(f"执行图: {execution_graph}")
        
        # 找到开始节点
        start_nodes = [node for node in self._nodes.values() 
                      if getattr(node, '_start_node', False)]
        
        if not start_nodes:
            raise ValueError("No start node found")
        
        if len(start_nodes) > 1:
            raise ValueError("Multiple start nodes found")
        
        # 用于追踪节点输出和执行状态
        node_outputs = {}  # {node_name: output}
        pending_nodes = {}  # {node_name: set of nodes waiting for}
        
        # 初始化待处理节点队列
        queue = [(start_nodes[0].__name__, initial_input)]
        executed = set()
        step = 0
        
        while queue:
            step += 1
            
            # 检查全局步骤限制
            if step > self.max_steps:
                raise RuntimeError(f"流程超过最大步骤限制 {self.max_steps}，可能陷入死循环")
            
            current_node_name, current_input = queue.pop(0)
            
            # 跳过已执行的节点
            if current_node_name in executed:
                continue
            
            current_node = self._nodes[current_node_name]
            
            # 检查节点级别循环限制
            max_loop = getattr(current_node, '_max_loop', None)
            if max_loop:
                self._loop_counters[current_node_name] = self._loop_counters.get(current_node_name, 0) + 1
                if self._loop_counters[current_node_name] > max_loop:
                    raise RuntimeError(f"节点 '{current_node_name}' 超过最大循环限制 {max_loop} 次")
                logger.debug(f"循环计数: {current_node_name} [{self._loop_counters[current_node_name]}/{max_loop}]")
            
            logger.debug(f"--- 步骤 {step} ---")
            logger.debug(f"执行节点: {current_node_name}")
            logger.debug(f"输入数据: {current_input}")
            
            # 记录节点开始时间
            import time
            if current_node_name not in self._node_timings:
                self._node_timings[current_node_name] = {}
            self._node_timings[current_node_name]["start"] = time.time()
            
            # 执行当前节点
            result = await current_node(current_input) if asyncio.iscoroutinefunction(current_node) else current_node(current_input)
            
            # 记录节点结束时间
            import time
            end_time = time.time()
            self._node_timings[current_node_name]["end"] = end_time
            start_time = self._node_timings[current_node_name].get("start", end_time)
            self._node_timings[current_node_name]["duration"] = end_time - start_time
            
            logger.debug(f"输出结果: {result}")
            
            # 记录输出
            node_outputs[current_node_name] = result
            executed.add(current_node_name)
            
            # 确定下一个节点
            goto_targets = getattr(current_node, '_goto_targets', None)
            
            if not goto_targets:
                logger.debug(f"节点 {current_node_name} 没有后续节点")
                continue
            
            # 处理路由逻辑
            next_nodes = []
            next_input = result
            
            if goto_targets is DYNAMIC:
                # DYNAMIC 节点：支持 Route 对象或元组
                if isinstance(result, Route):
                    # 新格式：使用 goto() 辅助函数
                    next_node_name, next_input = result.resolve()
                    next_nodes = [next_node_name]
                    logger.debug(f"DYNAMIC 路由: {next_node_name}")
                elif isinstance(result, tuple) and len(result) == 2:
                    # 向后兼容：元组格式
                    target, next_input = result
                    if callable(target):
                        next_node_name = target.__name__
                    else:
                        next_node_name = target
                    next_nodes = [next_node_name]
                    logger.debug(f"DYNAMIC 路由: {next_node_name} (元组格式，建议使用 goto())")
                else:
                    raise ValueError(
                        f"DYNAMIC 节点 '{current_node_name}' 必须返回:\n"
                        f"  - goto(target, data)  [推荐]\n"
                        f"  - (target, data)      [兼容格式]"
                    )
            elif isinstance(goto_targets, list):
                # 并行执行多个节点
                next_nodes = goto_targets
                logger.debug(f"并行执行节点: {next_nodes}")
            elif isinstance(goto_targets, str):
                # 单个节点
                next_nodes = [goto_targets]
                logger.debug(f"下一节点: {goto_targets}")
            else:
                raise ValueError(f"不支持的 goto 类型: {type(goto_targets)}")
            
            # 处理并行节点
            if len(next_nodes) > 1:
                logger.debug(f"使用 asyncio.gather 并行执行 {len(next_nodes)} 个节点")
                
                # 并行执行所有节点（带计时）
                import time
                
                async def execute_with_timing(node_name, node_func, node_input):
                    """执行节点并记录耗时"""
                    # 记录开始时间
                    if node_name not in self._node_timings:
                        self._node_timings[node_name] = {}
                    self._node_timings[node_name]["start"] = time.time()
                    
                    # 执行节点
                    if asyncio.iscoroutinefunction(node_func):
                        result = await node_func(node_input)
                    else:
                        result = node_func(node_input)
                    
                    # 记录结束时间
                    end_time = time.time()
                    self._node_timings[node_name]["end"] = end_time
                    start_time = self._node_timings[node_name].get("start", end_time)
                    self._node_timings[node_name]["duration"] = end_time - start_time
                    
                    return result
                
                # 创建并行任务
                tasks = []
                for next_node_name in next_nodes:
                    next_node = self._nodes[next_node_name]
                    tasks.append(execute_with_timing(next_node_name, next_node, next_input))
                
                # 等待所有任务完成
                results = await asyncio.gather(*tasks)
                
                # 记录每个节点的输出
                for node_name, node_result in zip(next_nodes, results):
                    logger.debug(f"  - {node_name} 完成: {node_result}")
                    node_outputs[node_name] = node_result
                    executed.add(node_name)
                    
                    # 检查该节点的后续节点
                    node_method = self._nodes[node_name]
                    node_goto = getattr(node_method, '_goto_targets', None)
                    
                    if node_goto:
                        # 找到后续节点
                        if node_goto is DYNAMIC:
                            subsequent_nodes = []  # DYNAMIC 节点由返回值决定
                        elif isinstance(node_goto, str):
                            subsequent_nodes = [node_goto]
                        elif isinstance(node_goto, list):
                            subsequent_nodes = node_goto
                        else:
                            subsequent_nodes = []
                        
                        # 将后续节点加入队列（带汇聚逻辑）
                        for sub_node in subsequent_nodes:
                            if sub_node in execution_graph:
                                # 检查是否所有前置节点都已执行
                                predecessors = execution_graph[sub_node]
                                if all(pred in executed for pred in predecessors):
                                    # 所有前置节点都完成，汇聚输入
                                    merged_input = {pred: node_outputs[pred] for pred in predecessors}
                                    logger.debug(f"节点 {sub_node} 的所有前置节点已完成，汇聚输入: {list(merged_input.keys())}")
                                    queue.append((sub_node, merged_input))
                            else:
                                # 没有多个前置节点，直接传递
                                queue.append((sub_node, node_result))
            else:
                # 单个后续节点
                for next_node_name in next_nodes:
                    if next_node_name in execution_graph:
                        # 检查是否需要等待其他前置节点
                        predecessors = execution_graph[next_node_name]
                        if all(pred in executed for pred in predecessors):
                            # 所有前置节点都完成
                            if len(predecessors) > 1:
                                # 多输入节点，汇聚
                                merged_input = {pred: node_outputs[pred] for pred in predecessors}
                                logger.debug(f"节点 {next_node_name} 的所有前置节点已完成，汇聚输入: {list(merged_input.keys())}")
                                queue.append((next_node_name, merged_input))
                            else:
                                # 单输入
                                queue.append((next_node_name, next_input))
                        else:
                            # 还有其他前置节点未完成，等待
                            waiting = [p for p in predecessors if p not in executed]
                            logger.debug(f"节点 {next_node_name} 等待前置节点: {waiting}")
                    else:
                        # 没有多个前置节点
                        queue.append((next_node_name, next_input))
        
        # 记录流程结束时间
        import time
        self._flow_end_time = time.time()
        
        logger.debug("========== 流程执行完成 ==========")
        logger.debug(f"最终状态: {self.state}")
        return self.state  # 返回完整状态对象
    
    def stats(self) -> dict:
        """获取流程统计信息
        
        Returns:
            {
                "timing": {  # 耗时统计
                    "total_duration": float,  # 总耗时(秒)
                    "nodes": {  # 节点耗时
                        "node_name": {
                            "duration": float,
                            "percentage": float  # 百分比
                        }
                    }
                }
                # 将来可以扩展其他统计信息，如：
                # "memory": {...},
                # "errors": {...},
                # "retries": {...}
            }
        """
        total_duration = self._flow_end_time - self._flow_start_time if self._flow_end_time > 0 else 0
        
        node_stats = {}
        for node_name, timing in self._node_timings.items():
            duration = timing.get("duration", 0)
            percentage = (duration / total_duration * 100) if total_duration > 0 else 0
            node_stats[node_name] = {
                "duration": duration,
                "percentage": percentage
            }
        
        return {
            "timing": {
                "total_duration": total_duration,
                "nodes": node_stats
            }
        }
    
    def print_timing_report(self):
        """打印耗时报告（简单文本格式）"""
        stats = self.stats()
        timing = stats.get("timing", {})
        total_duration = timing.get("total_duration", 0)
        
        print("\n" + "="*60)
        print(f"⏱️  流程耗时统计")
        print("="*60)
        print(f"总耗时: {total_duration:.2f}s")
        print("\n节点耗时:")
        print("-"*60)
        print(f"{'  节点名称':<30} {'  耗时(s)':>15} {'  百分比':>15}")
        print("-"*60)
        
        for node_name, node_timing in timing.get("nodes", {}).items():
            print(f"  {node_name:<28} {node_timing['duration']:>13.2f}s {node_timing['percentage']:>13.1f}%")
        
        print("="*60 + "\n")
