---
trigger: always_on
alwaysApply: true
---
# Python代码注释规范

## 规则概述
所有Python代码中的注释必须使用**英文**书写，并符合英文注释的规范标准。

## 具体规范要求

### 1. 语言要求
- **必须使用英文**：所有注释（包括单行注释、多行注释、文档字符串）必须使用英文书写
- **禁止使用中文**或其他非英语语言编写注释

### 2. 注释内容规范
- **清晰准确**：注释应清晰描述代码功能、算法逻辑或特殊处理原因
- **语法正确**：使用正确的英文语法、拼写和标点符号
- **简洁明了**：避免冗长，用简洁的语言表达完整意思

### 3. 注释格式规范
**单行注释示例：**
```python
# Calculate user score based on activity points
# TODO: Implement caching mechanism for better performance
```

**多行注释示例：**
```python
"""
This function processes the image data and extracts features.
It supports multiple image formats including JPEG, PNG, and WebP.
Returns: Processed image tensor
"""
```

**文档字符串示例：**
```python
def calculate_score(user_data):
    """
    Calculate comprehensive user score based on multiple factors.
    
    Args:
        user_data (dict): User activity and profile data
        
    Returns:
        float: Calculated score between 0.0 and 100.0
        
    Raises:
        ValueError: If user_data is missing required fields
    """
```

### 4. 特殊情况处理
- **专有名词**：允许保留必要的专有名词（如产品名、技术术语）
- **示例代码**：注释中的示例代码也需使用英文变量名和注释
- **TODO/FIXME**：必须使用英文描述待办事项或需要修复的问题

## 执行要求
- 代码审查时必须检查注释是否符合本规范
- 不符合规范的注释需要在提交前修正
- 新编写的代码必须严格遵守此规范

此规则旨在提高代码的可读性和可维护性，便于国际化团队协作和知识共享。