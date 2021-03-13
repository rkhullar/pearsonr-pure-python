## Pearsonr Pure Python

### Overview
This library provides a pure python implementation of [scipy.stats.pearsonr][scipy]. Only consider using if you require
this function in an environment that doesn't include prebuilt wheel files on pypi such as alpine linux.

### Warning
Use with caution as it may not be an exact match the function. For example in small sample sizes under 3 data points,
the original and ported functions differ in the result significance value.

### Links
- [scipy][scipy]
- [blog][blog]

[blog]: https://malishoaib.wordpress.com/2014/04/15/the-beautiful-beta-functions-in-raw-python/
[scipy]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
