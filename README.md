## Pandas 概述
`Pandas`是`Python`的核心数据分析支持库，提供了快速、灵活、明确的数据结构，旨在简单、直观地处理关系型、标记型数据,其长远目标是成为最强大、最灵活、可以支持任何语言的开源数据分析工具。

---
#### `pandas`适用于处理以下`数据类型`:

- 与Sql或者Excel表类似，含异构列的表格数据;
- 有序和无序（非固定频率）的时间序列数据;
- 带行列标签的矩阵数据，包括同构或异构型数据;
- 任意其它形式的观测、统计数据集, 数据转入 Pandas 数据结构时不必事先标记;、

#### `pandas`的主要数据结构是`Series`(一维数据)与`Dataframe`(二维数据)，`Pandas`基于`Numpy`开发，可以与第三方科学计算支持库完美结合。

*`Pandas`部分优势*

- 处理浮点与非浮点数据里的缺失数据，表示为 NaN；
- 大小可变：插入或删除 DataFrame 等多维对象的列；
- 自动、显式数据对齐：显式地将对象与一组标签对齐，也可以忽略标签，在 Series、DataFrame 计算时自动与数据对齐；
- 强大、灵活的分组（group by）功能：拆分-应用-组合数据集，聚合、转换数据；
- 把 Python 和 NumPy 数据结构里不规则、不同索引的数据轻松地转换为 DataFrame 对象；
- 基于智能标签，对大型数据集进行切片、花式索引、子集分解等操作；
- 直观地合并（merge）、 **连接（join）** 数据集；
- 灵活地重塑（reshape）、 **透视（pivot）** 数据集；
- 轴支持结构化标签：一个刻度支持多个标签；
- 成熟的 IO 工具：读取文本文件（CSV 等支持分隔符的文件）、Excel 文件、数据库等来源的数据，利用超快的 HDF5 格式保存 / 加载数据；
- 时间序列：支持日期范围生成、频率转换、移动窗口统计、移动窗口线性回归、日期位移等时间序列功能。

> 提要：
>- 处理数据一般分为几个阶段：数据整理与清洗、数据分析与建模、数据可视化与制表，`Pandas` 是处理数据的理想工具。
>- 与`Numpy`一样，`Pandas`很多底层算法由`Cpython`优化过，所以运行速度很快。
>- `Pandas`是`statsmodels`模块的依赖项。

---

## 数据结构

---

| 名称 | 维数 | 描述 |
| --- | --- | --- |
|  1  | Series| 带标签的一维同构数组 |
|  2  | DataFrame | 带标签的，大小可变的，二维异构表格 |

>注意：
>- `Pandas` 所有数据结构的值都是可变的，但数据结构的大小并非都是可变的，比如，Series 的长度不可改变，但 DataFrame 里就可以插入列。
>- `Pandas` 里，绝大多数方法都不改变原始输入的数据，而是复制数据，生成新的对象。
>- `数据对齐是内在的`，这一原则是根本的。除非显式指定，Pandas 不会断开标签和数据之间的连接。

### Series

*`Series`是带标签的一维数组，可存储整数、浮点数、字符串、Python 对象等类型的数据。轴标签统称为索引。调用 pd.Series 函数即可创建 Series：*

```python
import pandas as pd
import numpy as np

s = pd.Series(data, index=index)
```
 `data`支持以下数据类型：
- Python 字典
- 多维数组
- 标量值（如，5）

`index` 是轴标签列表。不同数据可分为以下几种情况：

#### 1,多维数组

`data` 是多维数组时，`index` 长度必须与 `data` 长度一致。没有指定 index 参数时，创建数值型索引，即 `[0, ..., len(data) - 1]`。

```python
In :pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
out:
a    1.159961
b    0.834810
c    0.516701
d   -0.160422
e    0.876053
dtype: float64
```

>注意：
> `Pandas` 的索引值可以重复。不支持重复索引值的操作会触发异常。其原因主要与性能有关，有很多计算实例，比如 `GroupBy` 操作就不用索引。

#### 2,字典

`Series` 可以用字典实例化：

```python
In :d = {'b': 1, 'a': 0, 'c': 2}
In :pd.Series(d)
out:
b    1
a    0
c    2
dtype: int64
```
>注意:
> 
>`data` 为字典，且未设置 `index` 参数时，如果 `Python` 版本 >= 3.6 且 `Pandas` 版本 >= 0.23，`Series` 按字典的插入顺序排序索引。
> 
>`Python` < 3.6 或 `Pandas` < 0.23，且未设置 `index` 参数时，`Series` 按字母顺序排序字典的键（key）列表。

>注意:
> 
>`Pandas` 用 `NaN` 表示缺失数据。

#### 3,标量值
`data` 是标量值时，必须提供索引。`Series` 按索引长度重复该标量值。
```python
In : pd.Series(5., index=['a', 'b', 'c', 'd', 'e'])
Out: 
a    5.0
b    5.0
c    5.0
d    5.0
e    5.0
dtype: float64
```

#### Series 类似多维数组
`Series` 操作与 `ndarray` 类似，支持大多数 `NumPy` 函数，还支持索引切片。

和 `NumPy` 数组一样，`Series` 也支持 `dtype` 。

`Series` 的数据类型一般是 `NumPy` 数据类型。不过，Pandas 和第三方库在一些方面扩展了 `NumPy` 类型系统，即扩展数据类型,比如，`Pandas` 的类别型数据与可空整数数据类型。


`Series.array` 用于提取 `Series` 数组。
```python
In : s.array
Out: 
<PandasArray>
[ 0.4691122999071863, -0.2828633443286633, -1.5090585031735124,
 -1.1356323710171934,  1.2121120250208506]
Length: 5, dtype: float64
```
`Series.array`一般是扩展数组。简单说，扩展数组是把 N 个 `numpy.ndarray` 包在一起的打包器。`Pandas` 知道怎么把扩展数组存储到 `Series` 或 `DataFrame` 的列里。

`Series` 只是类似于多维数组，提取真正的多维数组，要用 `Series.to_numpy()`。

```python
In : s.to_numpy()
Out: array([ 0.4691, -0.2829, -1.5091, -1.1356,  1.2121])
```

#### 矢量操作与对齐 Series 标签

`Series` 和 `NumPy` 数组一样，都不用循环每个值，而且 `Series` 支持大多数 `NumPy` 多维数组的方法。

`Series` 和多维数组的主要区别在于，`Series` 之间的操作会自动基于标签对齐数据。因此，不用顾及执行计算操作的 `Series` 是否有相同的标签。

```python
In : s[1:] + s[:-1]
Out: 
a         NaN
b   -0.565727
c   -3.018117
d   -2.271265
e         NaN
dtype: float64
```

操作未对齐索引的 `Series`， 其计算结果是所有涉及索引的并集。如果在 `Series` 里找不到标签，运算结果标记为 `NaN`，即缺失值。编写无需显式对齐数据的代码，给交互数据分析和研究提供了巨大的自由度和灵活性。`Pandas` 数据结构集成的数据对齐功能，是 `Pandas` 区别于大多数标签型数据处理工具的重要特性。

>注意：
> 
> 总之，让不同索引对象操作的默认结果生成索引并集，是为了避免信息丢失。就算缺失了数据，索引标签依然包含计算的重要信息。当然，也可以用函数`Series.dropna()`清除含有缺失值的标签。

```python
In :pd.Series.dropna(s[1:] + s[:-1])
out:
b   -0.565727
c   -3.018117
d   -2.271265
dtype: float64
```

#### Series 支持 name 属性：

```python
In : s = pd.Series(np.random.randn(5), name='something')
In : s
Out:
0   -0.494929
1    1.071804
2    0.721555
3   -0.706771
4   -1.039575
Name: something, dtype: float64

In : s.name
Out: 'something'
```
一般情况下，`Series` 自动分配 `name`，特别是提取一维 `DataFrame` 切片时

`pandas.Series.rename()`方法用于重命名 `Series`

```python
In : s2 = s.rename("different")
In : s2.name
Out: 'different'
```
>注意:
> 
> s 与 s2 指向不同的对象。

### DataFrame

`DataFrame` 是由多种类型的列构成的二维标签数据结构，类似于 `Excel` 、`SQL` 表，或 `Series` 对象构成的字典。`DataFrame` 是最常用的 `Pandas` 对象，与 `Series` 一样，`DataFrame` 支持多种类型的输入数据：

- 一维 `ndarray`、列表、字典、`Series` 字典。
- 二维 `numpy.ndarray`。
- 结构多维数组或记录多维数组。
- `Series`
- `DataFrame`

除了数据，还可以有选择地传递 `index`（行标签）和 `columns`（列标签）参数。传递了索引或列，就可以确保生成的 `DataFrame` 里包含索引或列。`Series` 字典加上指定索引时，会丢弃与传递的索引不匹配的所有数据。

没有传递轴标签时，按常规依据输入数据进行构建。

>注意:
> 
> Python > = 3.6，且 Pandas > = 0.23，数据是字典，且未指定 columns 参数时，DataFrame 的列按字典的插入顺序排序。 
>
>Python < 3.6 或 Pandas < 0.23，且未指定 columns 参数时，DataFrame 的列按字典键的字母排序。

#### 用 Series 字典或字典生成 DataFrame

生成的索引是每个 `Series` 索引的并集。先把嵌套字典转换为 `Series`。如果没有指定列，`DataFrame` 的列就是字典键的有序列表。

```python
In : d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
...:      'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
In : df = pd.DataFrame(d)
In : df
Out: 
   one  two
a  1.0  1.0
b  2.0  2.0
c  3.0  3.0
d  NaN  4.0
In : pd.DataFrame(d, index=['d', 'b', 'a'])
Out: 
   one  two
d  NaN  4.0
b  2.0  2.0
a  1.0  1.0
In : pd.DataFrame(d, index=['d', 'b', 'a'], columns=['two', 'three'])
Out: 
   two three
d  4.0   NaN
b  2.0   NaN
a  1.0   NaN
```

`index` 和 `columns` 属性分别用于访问行、列标签：

>注意：
> 
> 指定列与数据字典一起传递时，传递的列会覆盖字典的键。

```python
In : df.index
Out: Index(['a', 'b', 'c', 'd'], dtype='object')
In : df.columns
Out: Index(['one', 'two'], dtype='object')
```

#### 用多维数组字典、列表字典生成 DataFrame

多维数组的长度必须相同。如果传递了索引参数，`index` 的长度必须与数组一致。如果没有传递索引参数，生成的结果是 `range(n)`，n 为数组长度。

```python
In : d = {'one': [1., 2., 3., 4.],
...:      'two': [4., 3., 2., 1.]}
In : pd.DataFrame(d)
Out: 
   one  two
0  1.0  4.0
1  2.0  3.0
2  3.0  2.0
3  4.0  1.0
In : pd.DataFrame(d, index=['a', 'b', 'c', 'd'])
Out: 
   one  two
a  1.0  4.0
b  2.0  3.0
c  3.0  2.0
d  4.0  1.0
```

#### 用结构多维数组或记录多维数组生成 DataFrame
本例与数组字典的操作方式相同。

```python
In : data = np.zeros((2, ), dtype=[('A', 'i4'), ('B', 'f4'), ('C', 'a10')])
In : data[:] = [(1, 2., 'Hello'), (2, 3., "World")]
In : pd.DataFrame(data)
Out: 
   A    B         C
0  1  2.0  b'Hello'
1  2  3.0  b'World'
In : pd.DataFrame(data, index=['first', 'second'])
Out: 
        A    B         C
first   1  2.0  b'Hello'
second  2  3.0  b'World'
In : pd.DataFrame(data, columns=['C', 'A', 'B'])
Out: 
          C  A    B
0  b'Hello'  1  2.0
1  b'World'  2  3.0
```
#### 用列表字典生成 DataFrame

```python
In : data2 = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
In : pd.DataFrame(data2)
Out: 
   a   b     c
0  1   2   NaN
1  5  10  20.0
In : pd.DataFrame(data2, index=['first', 'second'])
Out: 
        a   b     c
first   1   2   NaN
second  5  10  20.0
In : pd.DataFrame(data2, columns=['a', 'b'])
Out: 
   a   b
0  1   2
1  5  10
```

#### 用元组字典生成 DataFrame

元组字典可以自动创建多层索引 DataFrame。

```python
In : pd.DataFrame({('a', 'b'): {('A', 'B'): 1, ('A', 'C'): 2},
...:               ('a', 'a'): {('A', 'C'): 3, ('A', 'B'): 4},
...:               ('a', 'c'): {('A', 'B'): 5, ('A', 'C'): 6},
...:               ('b', 'a'): {('A', 'C'): 7, ('A', 'B'): 8},
...:               ('b', 'b'): {('A', 'D'): 9, ('A', 'B'): 10}})
...: 
Out: 
       a              b      
       b    a    c    a     b
A B  1.0  4.0  5.0  8.0  10.0
  C  2.0  3.0  6.0  7.0   NaN
  D  NaN  NaN  NaN  NaN   9.0
```

#### 用 Series 创建 DataFrame

生成的 `DataFrame` 继承了输入的 `Series` 的索引，如果没有指定列名，默认列名是输入 `Series` 的名称。

##### 缺失数据

`DataFrame` 里的缺失值用 `np.nan` 表示。`DataFram`e 构建器以 `numpy.MaskedArray` 为参数时 ，被屏蔽的条目为缺失数据。

### 备选构建器

1,`DataFrame.from_dict`

`DataFrame.from_dict` 接收字典组成的字典或数组序列字典，并生成 `DataFrame`。除了 `orient` 参数默认为 `columns`，本构建器的操作与 `DataFrame` 构建器类似。把 `orient` 参数设置为 'index'， 即可把字典的键作为行标签。

```python
In : pd.DataFrame.from_dict(dict([('A', [1, 2, 3]), ('B', [4, 5, 6])]))
Out: 
   A  B
0  1  4
1  2  5
2  3  6
```
`orient='index'` 时，键是行标签。本例还传递了列名：
```python
In : pd.DataFrame.from_dict(dict([('A', [1, 2, 3]), ('B', [4, 5, 6])]),
...:                        orient='index', columns=['one', 'two', 'three'])
...: 
Out[58]: 
   one  two  three
A    1    2      3
B    4    5      6
```

2,`DataFrame.from_records`

`DataFrame.from_records` 构建器支持元组列表或结构数据类型（dtype）的多维数组。本构建器与 DataFrame 构建器类似，只不过生成的 `DataFrame` 索引是结构数据类型指定的字段。例如：

```python
In: data = np.array([(1, 2., b'Hello'), (2, 3., b'World')],
      dtype=[('A', '<i4'), ('B', '<f4'), ('C', 'S10')])
In: pd.DataFrame.from_records(data, index='C')
Out:
          A    B
C               
b'Hello'  1  2.0
b'World'  2  3.0
```
### 提取、添加、删除列

`DataFrame` 就像带索引的 `Series` 字典，提取、设置、删除列的操作与字典类似：

```python
In : df['one']
Out: 
a    1.0
b    2.0
c    3.0
d    NaN
Name: one, dtype: float64
In  df['three'] = df['one'] * df['two']
In : df['flag'] = df['one'] > 2
In : df
Out: 
   one  two  three   flag
a  1.0  1.0    1.0  False
b  2.0  2.0    4.0  False
c  3.0  3.0    9.0   True
d  NaN  4.0    NaN  False
```
删除（del、pop）列的方式也与字典类似：
```python
In : del df['two']
In : three = df.pop('three')
In : df
Out: 
   one   flag
a  1.0  False
b  2.0  False
c  3.0   True
d  NaN  False
```
标量值以广播的方式填充列：

```python
In : df['foo'] = 'bar'

In : df
Out: 
   one   flag  foo
a  1.0  False  bar
b  2.0  False  bar
c  3.0   True  bar
d  NaN  False  bar
```
插入与 `DataFrame` 索引不同的 `Series` 时，以 `DataFrame` 的索引为准：

```python
In : df['one_trunc'] = df['one'][:2]

In : df
Out: 
   one   flag  foo  one_trunc
a  1.0  False  bar        1.0
b  2.0  False  bar        2.0
c  3.0   True  bar        NaN
d  NaN  False  bar        NaN
```
可以插入原生多维数组，但长度必须与 DataFrame 索引长度一致。

默认在 `DataFrame` 尾部插入列。`insert` 函数可以指定插入列的位置：

```python
In : df.insert(1, 'bar', df['one'])
In : df
Out: 
   one  bar   flag  foo  one_trunc
a  1.0  1.0  False  bar        1.0
b  2.0  2.0  False  bar        2.0
c  3.0  3.0   True  bar        NaN
d  NaN  NaN  False  bar        NaN
```

### 用方法链分配新列

受 `dplyr` 的 mutate 启发，`DataFrame` 提供了 `assign()` 方法，可以利用现有的列创建新列。

```python
In : iris = pd.read_csv('data/iris.data')
In : iris.head()
Out: 
   SepalLength  SepalWidth  PetalLength  PetalWidth         Name
0          5.1         3.5          1.4         0.2  Iris-setosa
1          4.9         3.0          1.4         0.2  Iris-setosa
2          4.7         3.2          1.3         0.2  Iris-setosa
3          4.6         3.1          1.5         0.2  Iris-setosa
4          5.0         3.6          1.4         0.2  Iris-setosa
In : iris.assign(sepal_ratio=iris['SepalWidth'] / iris['SepalLength']
...:      .head())
...: 
Out[76]: 
   SepalLength  SepalWidth  PetalLength  PetalWidth         Name  sepal_ratio
0          5.1         3.5          1.4         0.2  Iris-setosa     0.686275
1          4.9         3.0          1.4         0.2  Iris-setosa     0.612245
2          4.7         3.2          1.3         0.2  Iris-setosa     0.680851
3          4.6         3.1          1.5         0.2  Iris-setosa     0.673913
4          5.0         3.6          1.4         0.2  Iris-setosa     0.720000
```

上例中，插入了一个预计算的值。还可以传递带参数的函数，在 `assign` 的 `DataFrame` 上求值。

```python
In : iris.assign(sepal_ratio=lambda x: (x['SepalWidth'] / x['SepalLength'])).head()
Out: 
   SepalLength  SepalWidth  PetalLength  PetalWidth         Name  sepal_ratio
0          5.1         3.5          1.4         0.2  Iris-setosa     0.686275
1          4.9         3.0          1.4         0.2  Iris-setosa     0.612245
2          4.7         3.2          1.3         0.2  Iris-setosa     0.680851
3          4.6         3.1          1.5         0.2  Iris-setosa     0.673913
4          5.0         3.6          1.4         0.2  Iris-setosa     0.720000
```

`assign` 返回的都是数据副本，原 `DataFrame` 不变。

### 索引 / 选择

索引基础用法如下：

| 操作 | 句法 | 结果 |
|---|---|---|
| 选择列 | df[col] | 	Series |
| 用标签选择行 | 	df.loc[label] | Series |
| 用整数位置选择行 | df.iloc[loc] | Series |
| 行切片 | df[5:10] | DataFrame |
| 用布尔向量选择行 | df[bool_vec] | 	DataFrame |

选择行返回 `Series`，索引是 `DataFrame` 的列：

```python
In : df.loc['b']
Out: 
one              2
bar              2
flag         False
foo            bar
one_trunc        2
Name: b, dtype: object
In : df.iloc[2]
Out: 
one             3
bar             3
flag         True
foo           bar
one_trunc     NaN
Name: c, dtype: object
```
### 数据对齐和运算

DataFrame 对象可以自动对齐 **列与索引（行标签）** 的数据。与上文一样，生成的结果是列和行标签的并集。

```python
In : df = pd.DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D'])
In : df2 = pd.DataFrame(np.random.randn(7, 3), columns=['A', 'B', 'C'])
In : df + df2
Out: 
          A         B         C   D
0  0.045691 -0.014138  1.380871 NaN
1 -0.955398 -1.501007  0.037181 NaN
2 -0.662690  1.534833 -0.859691 NaN
3 -2.452949  1.237274 -0.133712 NaN
4  1.414490  1.951676 -2.320422 NaN
5 -0.494922 -1.649727 -1.084601 NaN
6 -1.047551 -0.748572 -0.805479 NaN
7       NaN       NaN       NaN NaN
8       NaN       NaN       NaN NaN
9       NaN       NaN       NaN NaN
```

`DataFrame` 和 `Series` 之间执行操作时，默认操作是在 DataFrame 的列上对齐 Series 的索引，按行执行广播操作。例如

```python
In : df - df.iloc[0]
Out: 
          A         B         C         D
0  0.000000  0.000000  0.000000  0.000000
1 -1.359261 -0.248717 -0.453372 -1.754659
2  0.253128  0.829678  0.010026 -1.991234
3 -1.311128  0.054325 -1.724913 -1.620544
4  0.573025  1.500742 -0.676070  1.367331
5 -1.741248  0.781993 -1.241620 -2.053136
6 -1.240774 -0.869551 -0.153282  0.000430
7 -0.743894  0.411013 -0.929563 -0.282386
8 -1.194921  1.320690  0.238224 -1.482644
9  2.293786  1.856228  0.773289 -1.446531
```

### 数据对齐和运算

`DataFrame` 对象可以自动对齐 **列与索引（行标签）** 的数据。与上文一样，生成的结果是列和行标签的并集。

```python
In : df = pd.DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D'])
In : df2 = pd.DataFrame(np.random.randn(7, 3), columns=['A', 'B', 'C'])
In : df + df2
Out: 
          A         B         C   D
0  0.045691 -0.014138  1.380871 NaN
1 -0.955398 -1.501007  0.037181 NaN
2 -0.662690  1.534833 -0.859691 NaN
3 -2.452949  1.237274 -0.133712 NaN
4  1.414490  1.951676 -2.320422 NaN
5 -0.494922 -1.649727 -1.084601 NaN
6 -1.047551 -0.748572 -0.805479 NaN
7       NaN       NaN       NaN NaN
8       NaN       NaN       NaN NaN
9       NaN       NaN       NaN NaN
```

`DataFrame` 和 `Series` 之间执行操作时，默认操作是在 `DataFrame` 的列上对齐 `Series` 的索引，按行执行广播操作。

```python
In : df - df.iloc[0]
Out: 
          A         B         C         D
0  0.000000  0.000000  0.000000  0.000000
1 -1.359261 -0.248717 -0.453372 -1.754659
2  0.253128  0.829678  0.010026 -1.991234
3 -1.311128  0.054325 -1.724913 -1.620544
4  0.573025  1.500742 -0.676070  1.367331
5 -1.741248  0.781993 -1.241620 -2.053136
6 -1.240774 -0.869551 -0.153282  0.000430
7 -0.743894  0.411013 -0.929563 -0.282386
8 -1.194921  1.320690  0.238224 -1.482644
9  2.293786  1.856228  0.773289 -1.446531
```

时间序列是特例，`DataFrame` 索引包含日期时，按列广播：

```python
In : index = pd.date_range('1/1/2000', periods=8)

In : df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=list('ABC'))

In : df
Out: 
                   A         B         C
2000-01-01 -1.226825  0.769804 -1.281247
2000-01-02 -0.727707 -0.121306 -0.097883
2000-01-03  0.695775  0.341734  0.959726
2000-01-04 -1.110336 -0.619976  0.149748
2000-01-05 -0.732339  0.687738  0.176444
2000-01-06  0.403310 -0.154951  0.301624
2000-01-07 -2.179861 -1.369849 -0.954208
2000-01-08  1.462696 -1.743161 -0.826591

In : type(df['A'])
Out: Pandas.core.series.Series

In : df - df['A']
Out: 
            2000-01-01 00:00:00  2000-01-02 00:00:00  2000-01-03 00:00:00  2000-01-04 00:00:00  ...  2000-01-08 00:00:00   A   B   C
2000-01-01                  NaN                  NaN                  NaN                  NaN  ...                  NaN NaN NaN NaN
2000-01-02                  NaN                  NaN                  NaN                  NaN  ...                  NaN NaN NaN NaN
2000-01-03                  NaN                  NaN                  NaN                  NaN  ...                  NaN NaN NaN NaN
2000-01-04                  NaN                  NaN                  NaN                  NaN  ...                  NaN NaN NaN NaN
2000-01-05                  NaN                  NaN                  NaN                  NaN  ...                  NaN NaN NaN NaN
2000-01-06                  NaN                  NaN                  NaN                  NaN  ...                  NaN NaN NaN NaN
2000-01-07                  NaN                  NaN                  NaN                  NaN  ...                  NaN NaN NaN NaN
2000-01-08                  NaN                  NaN                  NaN                  NaN  ...                  NaN NaN NaN NaN

[8 rows x 11 columns]
```
> 警告
> 
> `df - df['A']`已弃用，后期版本中会删除。实现此操作的首选方法是：
> 
> `df.sub(df['A'], axis=0)`

支持布尔运算符：

```python
In : df1 = pd.DataFrame({'a': [1, 0, 1], 'b': [0, 1, 1]}, dtype=bool)
In : df2 = pd.DataFrame({'a': [0, 1, 1], 'b': [1, 1, 0]}, dtype=bool)
In : df1 & df2
Out: 
       a      b
0  False  False
1  False   True
2   True  False
In : df1 | df2
Out: 
      a     b
0  True  True
1  True  True
2  True  True
In : df1 ^ df2
Out: 
       a      b
0   True   True
1   True  False
2  False   True
In : -df1
Out: 
       a      b
0  False   True
1   True  False
2  False  False
```

### 转置

类似于多维数组，`T` 属性（即 `transpose` 函数）可以转置 `DataFrame`：

```python
# only show the first 5 rows
In : df[:5].T
Out: 
   2000-01-01  2000-01-02  2000-01-03  2000-01-04  2000-01-05
A   -1.226825   -0.727707    0.695775   -1.110336   -0.732339
B    0.769804   -0.121306    0.341734   -0.619976    0.687738
C   -1.281247   -0.097883    0.959726    0.149748    0.176444
```

### DataFrame 应用 NumPy 函数

`Series` 与 `DataFrame` 可使用 `log`、`exp`、`sqrt` 等多种元素级 `NumPy` 通用函数（ufunc） ，假设 `DataFrame` 的数据都是数字：

```python
In : np.exp(df)
Out: 
                   A         B         C
2000-01-01  0.293222  2.159342  0.277691
2000-01-02  0.483015  0.885763  0.906755
2000-01-03  2.005262  1.407386  2.610980
2000-01-04  0.329448  0.537957  1.161542
2000-01-05  0.480783  1.989212  1.192968
2000-01-06  1.496770  0.856457  1.352053
2000-01-07  0.113057  0.254145  0.385117
2000-01-08  4.317584  0.174966  0.437538

In : np.asarray(df)
Out: 
array([[-1.2268,  0.7698, -1.2812],
       [-0.7277, -0.1213, -0.0979],
       [ 0.6958,  0.3417,  0.9597],
       [-1.1103, -0.62  ,  0.1497],
       [-0.7323,  0.6877,  0.1764],
       [ 0.4033, -0.155 ,  0.3016],
       [-2.1799, -1.3698, -0.9542],
       [ 1.4627, -1.7432, -0.8266]])
```
`DataFrame` 不是多维数组的替代品，它的索引语义和数据模型与多维数组都不同。

通用函数应用于 Series 的底层数组。

```python
In : ser = pd.Series([1, 2, 3, 4])

In : np.exp(ser)
Out: 
0     2.718282
1     7.389056
2    20.085537
3    54.598150
dtype: float64
```

`Pandas` 可以自动对齐 `ufunc` 里的多个带标签输入数据。例如，两个标签排序不同的 `Series` 运算前，会先对齐标签。

```python
In : ser1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
In : ser2 = pd.Series([1, 3, 5], index=['b', 'a', 'c'])
In : ser1
Out: 
a    1
b    2
c    3
dtype: int64
In : ser2
Out: 
b    1
a    3
c    5
dtype: int64
In : np.remainder(ser1, ser2)
Out: 
a    1
b    0
c    3
dtype: int64
```

对 `Series` 和 `Index` 应用二进制 ufunc 时，优先执行 `Series`，并返回的结果也是 `Series` 。


### 控制台显示

控制台显示大型 `DataFrame` 时，会根据空间调整显示大小。`info() `函数可以查看 `DataFrame` 的信息摘要。下列代码读取 R 语言 plyr 包里的棒球数据集 CSV 文件）：

```python
In : baseball = pd.read_csv('data/baseball.csv')

In : print(baseball)
       id     player  year  stint team  lg   g   ab   r    h  X2b  X3b  hr   rbi   sb   cs  bb    so  ibb  hbp   sh   sf  gidp
0   88641  womacto01  2006      2  CHN  NL  19   50   6   14    1    0   1   2.0  1.0  1.0   4   4.0  0.0  0.0  3.0  0.0   0.0
1   88643  schilcu01  2006      1  BOS  AL  31    2   0    1    0    0   0   0.0  0.0  0.0   0   1.0  0.0  0.0  0.0  0.0   0.0
..    ...        ...   ...    ...  ...  ..  ..  ...  ..  ...  ...  ...  ..   ...  ...  ...  ..   ...  ...  ...  ...  ...   ...
98  89533   aloumo01  2007      1  NYN  NL  87  328  51  112   19    1  13  49.0  3.0  0.0  27  30.0  5.0  2.0  0.0  3.0  13.0
99  89534  alomasa02  2007      1  NYN  NL   8   22   1    3    1    0   0   0.0  0.0  0.0   0   3.0  0.0  0.0  0.0  0.0   0.0

[100 rows x 23 columns]

In : baseball.info()
<class 'Pandas.core.frame.DataFrame'>
RangeIndex: 100 entries, 0 to 99
Data columns (total 23 columns):
id        100 non-null int64
player    100 non-null object
year      100 non-null int64
stint     100 non-null int64
team      100 non-null object
lg        100 non-null object
g         100 non-null int64
ab        100 non-null int64
r         100 non-null int64
h         100 non-null int64
X2b       100 non-null int64
X3b       100 non-null int64
hr        100 non-null int64
rbi       100 non-null float64
sb        100 non-null float64
cs        100 non-null float64
bb        100 non-null int64
so        100 non-null float64
ibb       100 non-null float64
hbp       100 non-null float64
sh        100 non-null float64
sf        100 non-null float64
gidp      100 non-null float64
dtypes: float64(9), int64(11), object(3)
memory usage: 18.1+ KB
```

尽管 `to_string` 有时不匹配控制台的宽度，但还是可以用 `to_string` 以表格形式返回 `DataFrame` 的字符串表示形式：

```python
In : print(baseball.iloc[-20:, :12].to_string())
       id     player  year  stint team  lg    g   ab   r    h  X2b  X3b
80  89474  finlest01  2007      1  COL  NL   43   94   9   17    3    0
81  89480  embreal01  2007      1  OAK  AL    4    0   0    0    0    0
82  89481  edmonji01  2007      1  SLN  NL  117  365  39   92   15    2
83  89482  easleda01  2007      1  NYN  NL   76  193  24   54    6    0
84  89489  delgaca01  2007      1  NYN  NL  139  538  71  139   30    0
85  89493  cormirh01  2007      1  CIN  NL    6    0   0    0    0    0
86  89494  coninje01  2007      2  NYN  NL   21   41   2    8    2    0
87  89495  coninje01  2007      1  CIN  NL   80  215  23   57   11    1
88  89497  clemero02  2007      1  NYA  AL    2    2   0    1    0    0
89  89498  claytro01  2007      2  BOS  AL    8    6   1    0    0    0
90  89499  claytro01  2007      1  TOR  AL   69  189  23   48   14    0
91  89501  cirilje01  2007      2  ARI  NL   28   40   6    8    4    0
92  89502  cirilje01  2007      1  MIN  AL   50  153  18   40    9    2
93  89521  bondsba01  2007      1  SFN  NL  126  340  75   94   14    0
94  89523  biggicr01  2007      1  HOU  NL  141  517  68  130   31    3
95  89525  benitar01  2007      2  FLO  NL   34    0   0    0    0    0
96  89526  benitar01  2007      1  SFN  NL   19    0   0    0    0    0
97  89530  ausmubr01  2007      1  HOU  NL  117  349  38   82   16    3
98  89533   aloumo01  2007      1  NYN  NL   87  328  51  112   19    1
99  89534  alomasa02  2007      1  NYN  NL    8   22   1    3    1    0
```

默认情况下，过宽的 `DataFrame` 会跨多行输出：

`display.width` 选项可以更改单行输出的宽度：

```python
In : pd.set_option('display.width', 40)  # 默认值为 80

In : pd.DataFrame(np.random.randn(3, 12))
Out: 
          0         1         2         3         4         5         6         7         8         9        10        11
0  1.262731  1.289997  0.082423 -0.055758  0.536580 -0.489682  0.369374 -0.034571 -2.484478 -0.281461  0.030711  0.109121
1  1.126203 -0.977349  1.474071 -0.064034 -1.282782  0.781836 -1.071357  0.441153  2.353925  0.583787  0.221471 -0.744471
2  0.758527  1.729689 -0.964980 -0.845696 -1.340896  1.846883 -1.328865  1.682706 -1.717693  0.888782  0.228440  0.901805
```

还可以用 `display.max_colwidth` 调整最大列宽。

```python
In : datafile = {'filename': ['filename_01', 'filename_02'],
   .....:             'path': ["media/user_name/storage/folder_01/filename_01",
   .....:                      "media/user_name/storage/folder_02/filename_02"]}
   .....: 

In : pd.set_option('display.max_colwidth', 30)

In : pd.DataFrame(datafile)
Out: 
      filename                           path
0  filename_01  media/user_name/storage/fo...
1  filename_02  media/user_name/storage/fo...

In : pd.set_option('display.max_colwidth', 100)

In : pd.DataFrame(datafile)
Out: 
      filename                                           path
0  filename_01  media/user_name/storage/folder_01/filename_01
1  filename_02  media/user_name/storage/folder_02/filename_02
```
`expand_frame_repr` 选项可以禁用此功能，在一个区块里输出整个表格。



