
# Windows 10 Installation
- Host installation options: 
  - 1) Install python & tensorflow module
  - 2) Install anaconda
- Virual machine instllatipn:
  - 3) vagrant

## Installation 
### 1) Install python & tensorflow module
- Download and install : https://www.python.org/downloads/release/python-352/
- Install tensorflow 
```
py -m pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-0.12.0rc0-cp35-cp35m-win_amd64.whl
```
### 2) Install anaconda
- Downoload and install : https://www.continuum.io/downloads
- run
```
conda create -n tensorflow python=3.5
source activate tensorflow
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-0.12.0rc0-cp35-cp35m-win_amd64.whl
```

### 3) Vagrant
```
vagrant init quickstart/tensorflow; vagrant up --provider virtualbox
```

### IDE tool
- [Pycharm](https://www.jetbrains.com/pycharm/download/#section=windows)

### Test



## Tools
- [python 3.5.2](https://www.python.org/downloads/release/python-352/)
- [tensorflow r0.12 setup](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html)
