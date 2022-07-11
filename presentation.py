# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# # Cross-Validation with Irregular Time Series
# ## Sktime DEV Days 2022
# ### Stanislav Khrapov, Data Scientist, Chintai
# ![Chintai](https://media.newjobs.com/clu/xw31/xw318532284wDEx/branding/177718/Chintai-logo-637553537963924757.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Time Series Cross-Validation
# ![split](https://i.stack.imgur.com/fXZ6k.png)

# %% [markdown] slideshow={"slide_type": "slide"}
# # Sktime splitters
#
# `sktime.forecasting.model_selection`:
#
# - `ExpandingWindowSplitter`
# - `SlidingWindowSplitter`
# - `CutoffSplitter`
# - `SingleWindowSplitter`

# %% [markdown] slideshow={"slide_type": "slide"}
# ```python
# import numpy as np
# import pandas as pd
#
# from sktime.forecasting.model_selection import CutoffSplitter
#
# index = pd.period_range(start="2022-07-13", periods=4, freq="D")
# y = pd.Series([12, 23, 18, 42], index=index)
# cutoffs = np.array([2])
#
# cv = CutoffSplitter(cutoffs, fh=1, window_length=2)
#
# print(list(cv.split(y)))
#
# >>> [(array([1, 2]), array([3]))]
# ```

# %% [markdown] slideshow={"slide_type": "slide"}
# Time series example: $\left(y_1,y_2^a,y_2^b,y_4\right)$. 
#
# Corresponding enumeration: $(1,2,3,4)$
#
# - Two observations for the time index $2$
# - No observation for time index $3$

# %% [markdown] slideshow={"slide_type": "subslide"}
# Current state:
#
# | cutoff | train               | test                |
# | --------|---------------------|---------------------|
# | 1      | $$(y_1)$$             | $$(y_2^a,y_2^b,y_4)$$ |
# | 2      | $$(y_1,y_2^a)$$       | $$(y_2^b,y_4)$$       |
# | 3      | $$(y_1,y_2^a,y_2^b)$$ | $$(y_4)$$             |

# %% [markdown] slideshow={"slide_type": "subslide"}
# Current state:
#
# | cutoff | train             | test      |
# |--------|-------------------|-----------|
# | 1      | $$(1)$$            | $$(2,3,4)$$ |
# | 2      | $$(1,2)$$           | $$(3,4)$$   |
# | 3      | $$(1,2,3)$$         | $$(4)$$     |

# %% [markdown] slideshow={"slide_type": "subslide"}
# Expected state:
#
# | cutoff | train               | test                |
# |--------|---------------------|---------------------|
# | 1      | $$(y_1)$$             | $$(y_2^a,y_2^b,y_4)$$ |
# | 2      | $$(y_1,y_2^a,y_2^b)$$ | $$(y_4)$$             |
# | 3      | $$(y_1,y_2^a,y_2^b)$$ | $$(y_4)$$             |

# %% [markdown] slideshow={"slide_type": "subslide"}
# Expected state:
#
# | cutoff | train               | test      |
# |--------|---------------------|-----------|
# | 1      | $$(1)$$               | $$(2,3,4)$$ |
# | 2      | $$(1,2,3)$$           | $$(4)$$     |
# | 3      | $$(1,2,3)$$           | $$(4)$$     |

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Formal definition and proposal
#
# Denote a time series index as $T=\{t(1),\ldots,t(k)\}$. Assume that it is sorted, that is $t(i+1)\geq t(i)$. Also assume that $t(i)$ can be either an integer or a date/time value.
#
# **Definition.** A time series is regular if $t(i+1)-t(i)=t(j+1)-t(j)$ for any $i,j\in\{2,\ldots,k\}$. Conversely, a time series is irregular, if there exists $i\neq j$ such that $t(i+1)-t(i)\neq t(j+1)-t(j)$.
#
# **Definition.** A cutoff is a reference to the index $t(s)$ such that $t(1)\leq t(s)\leq t(k)$. It separates train and test windows, $F=\{t(m_1),\ldots,t(m_f)\}$ and $P=\{t(h_1),\ldots,t(h_p)\}$, respectively. Exact definition of a train/test window depends on a specific splitter. Regardless of a splitter, $t(s)\geq t\in F$ and $t(s)< t\in P$. 
#
# For a regular time series it is guaranteed that any cutoff $t(s)\in T$. Conversely, for irregular time series there exists $s$ such that $t(s)\notin T$.

# %% [markdown] slideshow={"slide_type": "subslide"}
# The current state of `sktime` supports only regular time series. At the core the implementation relied on constructing train/test windows using `np.arange`, which was sufficient given the knowledge of window left and right endpoints. For example,
# ```python
# np.arange(train_start, train_end + 1)
# ```
# gave us `iloc` references to the train window.
#
# After a series of refactoring PRs this implementation was generalized using `pandas.Index.get_loc` and `numpy.argwhere` methods. The first one is used to obtain `iloc` reference $s$ in $t(s)$, while the second is used to get `iloc` references $\{m_1,\ldots,m_f\}$ and $\{h_1,\ldots,h_p\}$. For example,
# ```python
# train_end = y.get_loc(cutoff)
# ```
# gives us the `iloc` reference to the end of the training window, while
# ```python
# np.argwhere((y >= train_start) & (y <= train_end))
# ```
# gives us `iloc` references to the train window. The advantage here is that we may pass an irregular time series and still get correct `iloc` indices.

# %% [markdown] slideshow={"slide_type": "subslide"}
# Going deeper into the implementation it turns out that such a refactoring is still not sufficient to treat all currently existing splitters. In particular, `y.get_loc(cutoff)` raises `KeyError` if `cutoff` does not belong to the index `y`. We propose to treat this as follows. For an irregular index $T=\{t(1),\ldots,t(k)\}$ we can construct a corresponding regular index $T^\prime=\{t^\prime(1),\ldots,t^\prime(l)\}$ such that $t(1)=t^\prime(1)$ and $t(k)=t^\prime(l)$. For such an index 
# ```python
# y_regular.get_loc(cutoff)
# ```
# always returns a meaningful `iloc` reference in the context of a regular time index `y_regular`. Same for
# ```python
# np.argwhere((y_regular >= train_start) & (y_regular <= train_end))
# ```
# After obtaining a train and/or test windows one has to convert them back to the context of original irregular index `y`. This can be achieved by using, for example,
# ```python
# y.get_indexer(y_regular[train])
# ```
# which returns `iloc` references to `y` for only those elements of `y_regular[train]` that exist in `y`.

# %% [markdown] slideshow={"slide_type": "subslide"}
# Constructing `y_regular` for integer valued `y` is trivial:
# ```python
# np.arange(y[0], y[-1] + 1)
# ```
# For date/time `y` one needs to know the frequency of a time series after aggregation/imputation. Then, for example,
# ```python
# pd.period_range(y.min(), y.max(), freq=freq)
# ```
# produces the desired result. Currently, if one passes an irregular time index to any splitter in `sktime`, there is no robust way to guess a desired frequency since aggregation/imputation may be performed for any time unit. Hence, it is required to implement one more optional argument in splitter constructor, namely `freq`:
# ```python
# def __init__(
#     self,
#     fh = DEFAULT_FH,
#     window_length = DEFAULT_WINDOW_LENGTH,
#     freq: str = None,
# ) -> None:
#     self.window_length = window_length
#     self.fh = fh
#     self.freq = freq
# ```
