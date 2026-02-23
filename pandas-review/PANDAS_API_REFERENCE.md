# Pandas API 快速参考

> 本文档按照 Series 和 DataFrame 的 API 分类整理，每个 API 都有简洁易懂的示例。

---

## 目录

- [Series API](#series-api)
  - [创建 Series](#创建-series)
  - [访问元素](#访问元素)
  - [属性](#series-属性)
  - [常用方法](#series-常用方法)
- [DataFrame API](#dataframe-api)
  - [创建 DataFrame](#创建-dataframe)
  - [读取数据](#读取数据)
  - [查看数据](#查看数据)
  - [选择数据](#选择数据)
  - [过滤数据](#过滤数据)
  - [增删改操作](#增删改操作)
  - [统计分析](#统计分析)
  - [数据转换](#数据转换)

---

## Series API

### 创建 Series

#### `pd.Series(data, index=None)`
从列表、数组或字典创建 Series。

```python
import pandas as pd

# 从列表创建（默认索引 0, 1, 2...）
s = pd.Series([10, 20, 30])
# 0    10
# 1    20
# 2    30

# 自定义索引
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
# a    10
# b    20
# c    30

# 从字典创建
s = pd.Series({'a': 10, 'b': 20, 'c': 30})
# a    10
# b    20
# c    30
```

---

### 访问元素

#### `s[index]` 或 `s.loc[label]`
通过标签访问元素。

```python
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

# 通过标签访问
print(s['a'])        # 10
print(s.loc['b'])    # 20

# 通过位置访问
print(s.iloc[0])     # 10
```

#### `s.iloc[position]`
通过位置索引访问元素。

```python
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

print(s.iloc[0])     # 10
print(s.iloc[-1])    # 30
print(s.iloc[0:2])   # 前两个元素
# a    10
# b    20
```

---

### Series 属性

#### `s.values`
返回 Series 的值（NumPy 数组）。

```python
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

print(s.values)      # array([10, 20, 30])
print(type(s.values))  # <class 'numpy.ndarray'>
```

#### `s.index`
返回 Series 的索引。

```python
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

print(s.index)           # Index(['a', 'b', 'c'], dtype='object')
print(s.index.tolist())  # ['a', 'b', 'c']
```

#### `s.shape`
返回 Series 的形状。

```python
s = pd.Series([10, 20, 30])

print(s.shape)  # (3,)
```

#### `s.dtype`
返回 Series 的数据类型。

```python
s = pd.Series([10, 20, 30])

print(s.dtype)  # int64
```

---

### Series 常用方法

#### `s.head(n)` / `s.tail(n)`
查看前 n 个或后 n 个元素。

```python
s = pd.Series([10, 20, 30, 40, 50])

print(s.head(3))  # 前3个
# 0    10
# 1    20
# 2    30

print(s.tail(2))  # 后2个
# 3    40
# 4    50
```

#### `s.describe()`
返回描述性统计信息。

```python
s = pd.Series([10, 20, 30, 40, 50])

print(s.describe())
# count     5.0
# mean     30.0
# std      15.811388
# min      10.0
# 25%      20.0
# 50%      30.0
# 75%      40.0
# max      50.0
```

#### `s.unique()`
返回唯一值。

```python
s = pd.Series([1, 2, 2, 3, 3, 3])

print(s.unique())  # array([1, 2, 3])
```

#### `s.value_counts()`
统计每个值的出现次数。

```python
s = pd.Series([1, 2, 2, 3, 3, 3])

print(s.value_counts())
# 3    3
# 2    2
# 1    1
```

#### `s.isnull()` / `s.notnull()`
检查缺失值。

```python
s = pd.Series([1, 2, None, 4])

print(s.isnull())
# 0    False
# 1    False
# 2     True
# 3    False

print(s.notnull())
# 0     True
# 1     True
# 2    False
# 3     True
```

#### `s.fillna(value)`
填充缺失值。

```python
s = pd.Series([1, 2, None, 4])

print(s.fillna(0))
# 0    1.0
# 1    2.0
# 2    0.0
# 3    4.0
```

#### `s.dropna()`
删除缺失值。

```python
s = pd.Series([1, 2, None, 4])

print(s.dropna())
# 0    1.0
# 1    2.0
# 3    4.0
```

#### `s.sort_values(ascending=True)`
按值排序。

```python
s = pd.Series([30, 10, 20], index=['a', 'b', 'c'])

print(s.sort_values())
# b    10
# c    20
# a    30

print(s.sort_values(ascending=False))
# a    30
# c    20
# b    10
```

#### `s.sort_index(ascending=True)`
按索引排序。

```python
s = pd.Series([30, 10, 20], index=['c', 'a', 'b'])

print(s.sort_index())
# a    10
# b    20
# c    30
```

---

## DataFrame API

### 创建 DataFrame

#### `pd.DataFrame(data, index=None, columns=None)`
从字典、列表或数组创建 DataFrame。

```python
import pandas as pd

# 从字典创建（键为列名）
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'score': [85, 90, 95]
})
#       name  age  score
# 0    Alice   25     85
# 1      Bob   30     90
# 2  Charlie   35     95

# 从列表创建
df = pd.DataFrame([
    ['Alice', 25, 85],
    ['Bob', 30, 90]
], columns=['name', 'age', 'score'])

# 自定义行索引
df = pd.DataFrame({
    'age': [25, 30],
    'score': [85, 90]
}, index=['row1', 'row2'])
```

---

### 读取数据

#### `pd.read_csv(filepath, sep=',', header=0, index_col=None)`
从 CSV 文件读取数据。

```python
# 基本读取
df = pd.read_csv('data.csv')

# 指定分隔符
df = pd.read_csv('data.tsv', sep='\t')

# 指定第一列为索引
df = pd.read_csv('data.csv', index_col=0)

# 只读取指定列
df = pd.read_csv('data.csv', usecols=['name', 'age'])
```

#### `pd.read_excel(filepath, sheet_name=0)`
从 Excel 文件读取数据。

```python
# 读取第一个工作表
df = pd.read_excel('data.xlsx')

# 读取指定工作表
df = pd.read_excel('data.xlsx', sheet_name='Sheet2')
```

#### `pd.read_json(filepath)`
从 JSON 文件读取数据。

```python
df = pd.read_json('data.json')
```

---

### 查看数据

#### `df.head(n)` / `df.tail(n)`
查看前 n 行或后 n 行。

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 40, 45]
})

print(df.head(3))  # 前3行
print(df.tail(2))  # 后2行
```

#### `df.shape`
返回 DataFrame 的形状（行数，列数）。

```python
print(df.shape)  # (5, 2)
```

#### `df.columns`
返回列名。

```python
print(df.columns.tolist())  # ['name', 'age']
```

#### `df.index`
返回行索引。

```python
print(df.index.tolist())  # [0, 1, 2, 3, 4]
```

#### `df.dtypes`
返回每列的数据类型。

```python
print(df.dtypes)
# name    object
# age      int64
```

#### `df.info()`
显示 DataFrame 的概要信息。

```python
df.info()
# 显示行数、列数、非空值数量、数据类型等
```

#### `df.describe()`
返回数值列的描述性统计。

```python
print(df.describe())
# 显示 count、mean、std、min、25%、50%、75%、max
```

---

### 选择数据

#### `df['column']`
选择单列（返回 Series）。

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 30]
})

print(df['name'])
# 0    Alice
# 1      Bob
# Name: name, dtype: object
```

#### `df[['col1', 'col2']]`
选择多列（返回 DataFrame）。

```python
print(df[['name', 'age']])
#     name  age
# 0  Alice   25
# 1    Bob   30
```

#### `df.iloc[row, col]`
按位置索引选择（整数索引）。

```python
# 选择单个元素
print(df.iloc[0, 1])  # 25

# 选择一行
print(df.iloc[0])
# name    Alice
# age        25

# 选择多行
print(df.iloc[0:2])

# 选择行和列
print(df.iloc[0:2, 0:1])
```

#### `df.loc[row, col]`
按标签索引选择。

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 30]
}, index=['row1', 'row2'])

# 选择单个元素
print(df.loc['row1', 'age'])  # 25

# 选择一行
print(df.loc['row1'])

# 选择多行
print(df.loc['row1':'row2'])
```

---

### 过滤数据

#### 单条件过滤
使用布尔索引筛选数据。

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'score': [85, 90, 95]
})

# 筛选年龄大于 25 的行
print(df[df['age'] > 25])
#       name  age  score
# 1      Bob   30     90
# 2  Charlie   35     95

# 筛选分数等于 90 的行
print(df[df['score'] == 90])
```

#### 多条件过滤（AND）
使用 `&` 连接多个条件。

```python
# 年龄 > 25 且 分数 > 85
print(df[(df['age'] > 25) & (df['score'] > 85)])
#       name  age  score
# 1      Bob   30     90
# 2  Charlie   35     95
```

#### 多条件过滤（OR）
使用 `|` 连接多个条件。

```python
# 年龄 < 30 或 分数 > 90
print(df[(df['age'] < 30) | (df['score'] > 90)])
```

#### `df.isin(values)`
检查值是否在列表中。

```python
# 筛选 name 在指定列表中的行
print(df[df['name'].isin(['Alice', 'Bob'])])
```

---

### 增删改操作

#### 添加新列
直接赋值添加新列。

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 30]
})

# 添加常量列
df['city'] = 'Beijing'

# 基于现有列计算
df['age_plus_10'] = df['age'] + 10
```

#### `df.drop(labels, axis=0/1)`
删除行或列。

```python
# 删除列（axis=1）
df_new = df.drop('city', axis=1)

# 删除多列
df_new = df.drop(['city', 'age_plus_10'], axis=1)

# 删除行（axis=0）
df_new = df.drop(0, axis=0)  # 删除第0行
```

#### `df.rename(columns={old: new})`
重命名列。

```python
df_new = df.rename(columns={'name': '姓名', 'age': '年龄'})
```

---

### 统计分析

#### `df.describe()`
返回数值列的描述性统计。

```python
df = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'score': [85, 90, 95, 88]
})

print(df.describe())
# 显示 count、mean、std、min、25%、50%、75%、max
```

#### `df.mean()` / `df.sum()` / `df.max()` / `df.min()`
计算均值、总和、最大值、最小值。

```python
print(df['age'].mean())   # 32.5
print(df['score'].sum())  # 358
print(df['age'].max())    # 40
print(df['score'].min())  # 85
```

#### `df.count()`
统计非空值数量。

```python
print(df.count())
# age      4
# score    4
```

#### `df.value_counts()`
统计每个值的出现次数。

```python
df = pd.DataFrame({
    'city': ['Beijing', 'Shanghai', 'Beijing', 'Shanghai']
})

print(df['city'].value_counts())
# Beijing     2
# Shanghai    2
```

---

### 数据转换

#### `df.values`
转换为 NumPy 数组。

```python
df = pd.DataFrame({
    'age': [25, 30, 35],
    'score': [85, 90, 95]
})

numpy_array = df.values
print(numpy_array)
# [[25 85]
#  [30 90]
#  [35 95]]
```

#### 转换为 PyTorch Tensor
通过 NumPy 转换为 Tensor。

```python
import torch

# 方式1：通过 torch.tensor()
tensor = torch.tensor(df.values)

# 方式2：通过 torch.from_numpy()
tensor = torch.from_numpy(df.values)

# 指定数据类型
tensor = torch.tensor(df.values, dtype=torch.float32)
```

#### `df.to_csv(filepath)`
保存为 CSV 文件。

```python
df.to_csv('output.csv', index=False)
```

#### `df.to_excel(filepath)`
保存为 Excel 文件。

```python
df.to_excel('output.xlsx', index=False)
```

---

## 常用技巧

### 链式操作
可以链式调用多个方法。

```python
result = (df[df['age'] > 25]
          .sort_values('score', ascending=False)
          .head(3))
```

### 缺失值处理

```python
# 检查缺失值
df.isnull().sum()

# 填充缺失值
df.fillna(0)

# 删除缺失值
df.dropna()
```

### 数据类型转换

```python
# 转换列的数据类型
df['age'] = df['age'].astype(float)
df['score'] = df['score'].astype(int)
```

---

## 总结

### Series vs DataFrame

| 特性 | Series | DataFrame |
|------|--------|-----------|
| 维度 | 一维 | 二维 |
| 类比 | 列表/字典 | 表格/Excel |
| 索引 | 一个索引 | 行索引 + 列名 |
| 访问 | `s[index]` | `df['col']` 或 `df.iloc[row, col]` |

### 常用操作速查

```python
# 创建
s = pd.Series([1, 2, 3])
df = pd.DataFrame({'col': [1, 2, 3]})

# 读取
df = pd.read_csv('file.csv')

# 查看
df.head()
df.shape
df.info()

# 选择
df['col']           # 单列
df[['col1', 'col2']]  # 多列
df.iloc[0]          # 按位置
df.loc['label']     # 按标签

# 过滤
df[df['col'] > 10]
df[(df['col1'] > 10) & (df['col2'] < 20)]

# 统计
df.describe()
df['col'].mean()
df['col'].value_counts()

# 转换
df.values           # 转 NumPy
torch.tensor(df.values)  # 转 Tensor
```

---

**文档完成！** 🎉

