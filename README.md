# PySVMRank

Python API for SVMrank (http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html)

# Dependencies

`numpy`

`cython`

`pytest` (optional)

# Install

Clone the repo:
```
git clone https://github.com/ds4dm/PySVMRank.git
cd PySVMRank
```

Download SVMrank source code:
```
wget http://download.joachims.org/svm_rank/current/svm_rank.tar.gz
mkdir src/c
tar -xzf svm_rank.tar.gz -C src/c
```

Compile and install:
```
pip install .
```

# Test
```
python -m pytest
```
