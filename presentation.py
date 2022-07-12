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

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Typical cross-validation pipeline
#
# - get raw irregular data
# - process data (aggregate, impute, align, etc.)
# - cross-validate (repeat the following many times for different splits):
#   - split
#   - forecast $y$
#   - record forecasts, residuals, etc
# - aggregate cross-validation results

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Existing implementation fails when some data processing is required after each split operation
#
# - cross-validate (repeat the following many times for different splits):
#   - split
#   - **process data**
#   - forecast $y$

# %% [markdown] slideshow={"slide_type": "slide"}
# # Sktime splitters
#
# `sktime.forecasting.model_selection`:
#
# - `ExpandingWindowSplitter`
# - `SlidingWindowSplitter`
# - `CutoffSplitter`
# - `SingleWindowSplitter`

# %% [markdown] slideshow={"slide_type": "subslide"}
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
# ## Toy example
#
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
# ## Formal definition and proposal
#
# - Time series index: $T=\{t(1),\ldots,t(k)\}$
# - It is sorted: $t(i+1)\geq t(i)$
# - $t(i)$ can be either an integer or a date/time value
#
# **Definition.** A time series is regular if $t(i+1)-t(i)=t(j+1)-t(j)$ for any $i,j\in\{2,\ldots,k\}$.
#
# **Definition.** A cutoff is a reference to the index $t(s)$ such that $t(1)\leq t(s)\leq t(k)$. It separates train and test windows, $F=\{t(m_1),\ldots,t(m_f)\}$ and $P=\{t(h_1),\ldots,t(h_p)\}$, respectively. Regardless of a splitter, $t(s)\geq t\in F$ and $t(s)< t\in P$. 
#
# For irregular time series there exists $s$ such that $t(s)\notin T$.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Original implementation (before me)
#
# The current state of `sktime` supports only regular time series.
#
# ```python
# np.arange(train_start, train_end + 1)
# ```
# gave us `iloc` references to the train window.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### As of now implementation
#
# Generalized using `pandas.Index.get_loc` and `numpy.argwhere` methods.
#
# For example,
# ```python
# train_end = y.get_loc(cutoff)
# ```
# gives us the `iloc` reference to the end of the training window, while
# ```python
# np.argwhere((y >= train_start) & (y <= train_end))
# ```
# gives us `iloc` references to the train window.
#
# Now we may pass an irregular time series and still get correct `iloc` indices.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### But!...
#
# `y.get_loc(cutoff)` raises `KeyError` if `cutoff` does not belong to the index `y`.
#
# **Solution:**
#
# For an irregular index $T=\{t(1),\ldots,t(k)\}$ construct regular index $T^\prime=\{t^\prime(1),\ldots,t^\prime(l)\}$ such that $t(1)=t^\prime(1)$ and $t(k)=t^\prime(l)$.
# ```python
# y_regular.get_loc(cutoff)
# ```
# always returns a meaningful `iloc` reference in the context of a regular time index `y_regular`. Same for
# ```python
# np.argwhere((y_regular >= train_start) & (y_regular <= train_end))
# ```
#
# After obtaining a train and/or test windows convert them back to the context of original irregular index `y`:
# ```python
# y.get_indexer(y_regular[train])
# ```
# which returns `iloc` references to `y` for only those elements of `y_regular[train]` that exist in `y`.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Constructing regular index
#
# `y_regular` for integer valued `y`:
# ```python
# np.arange(y[0], y[-1] + 1)
# ```
# For date/time `y`:
# ```python
# pd.period_range(y.min(), y.max(), freq=freq)
# ```
# Note that `freq` is required!
#
# New argument `freq` at splitter construction:
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
