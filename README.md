# data-augmentation
Research on data augmentation methods.
## Installation
To install the required packages run pip install in your environment
```
pip install -r requirements.txt
```
It is recommended to use conda. Because different parts of this project are written in their own modules, you need to add them to your python *sys.path*
- *src* is needed for accessing the source code of this project
- *./* is needed for accessing the classes of saved models
```
conda develop src
conda develop ./
```
