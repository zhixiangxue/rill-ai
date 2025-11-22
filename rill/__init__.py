"""Rill - AI Agent Framework"""

import sys

import rill.core.flow
sys.modules['rill.flow'] = rill.core.flow

import chak
sys.modules['rill.chak'] = chak

# 导出核心类和函数
from rill.core import Flow, FlowState, Node, node, Route, goto, DYNAMIC, END

__version__ = "0.1.0"

__all__ = [
    'Flow',
    'FlowState',
    'Node',
    'node',
    'Route',
    'goto',
    'DYNAMIC',
    'END',
]