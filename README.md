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
- 直观地合并（merge）、**连接（join）**数据集；
- 灵活地重塑（reshape）、**透视（pivot）**数据集；
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

####2,字典

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
In [: data[:] = [(1, 2., 'Hello'), (2, 3., "World")]
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



