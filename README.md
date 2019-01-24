# OntoDoc

Find existing ontologies based on papers or other text documents.

## Dependencies

The following is needed to run OntoDoc

* Python 3.6 or higher
* BioPython 1.72
* PyPDF2 1.26.0
* docx 0.6 or higher
* ols_client 0.0.9
* gensim 3.5.0 or higher
* rdflib 4.2.2
* nltk 3.2.5 or higher
* pandas 0.23.4
* matplotlib 2.2.3 or higher
* sklearn

## Install OntoDoc

### Ubuntu

```bash
sudo apt update
sudo apt upgrade
sudo apt install python3 python3-pip
pip3 install biopython PyPDF2 python-docx ols-client gensim rdflib nltk pandas matplotlib sklearn
git clone https://github.com/rjansen1984/OntoDoc
```

### CentOS

```bash
sudo yum makecache
sudo yum install yum-utils
sudo yum install https://centos7.iuscommunity.org/ius-release.rpm
sudo yum makecache
sudo yum install python36u python36u-pip
pip3.6 install biopython PyPDF2 python-docx ols-client gensim rdflib nltk pandas matplotlib sklearn
git clone https://github.com/rjansen1984/OntoDoc
```

### Windows 10

* Download and install [Python 3.6.5](https://www.python.org/downloads/release/python-365/)
* Download and install [git for Windows](https://git-scm.com/download/win)

```bash
pip3 install biopython PyPDF2 python-docx ols-client gensim rdflib nltk pandas matplotlib sklearn
git clone https://github.com/rjansen1984/OntoDoc
```

### Using Conda

* Download and install the Python 3 version of [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://conda.io/miniconda.html)

```bash
conda install -c conda-forge biopython
conda install -c conda-forge pypdf2
conda install -c conda-forge python-docx
conda install -c conda-forge pip
pip install ols-client
conda install -c conda-forge gensim
conda install -c conda-forge rdflib
conda install -c conda-forge nltk
conda install -c conda-forge pandas
conda install -c conda-forge matplotlib
git clone https://github.com/rjansen1984/OntoDoc
```

*Make sure to use the conda pip

## Use OntoDoc

OntoDoc can be used with text files or with Pubmed IDs. Using Pubmed IDs only abstracts will be used instead of full papers.

To start the script use the following commands:

for Windows:

```bash
py ontodoc.py
```

for Linux:

```bash
python ontodoc.py
```

There will be three option to provide data:

1. Enter a file path to an existing file.

2. Enter a Pubmed ID.

3. Paste data in the command line.

### File path

Select option 1.

Enter file paths comma seperated e.g. /path/to/file1,/path/to/file2

### Pubmed IDs

Select option 2.

Enter Pubmed IDs comma seperated e.g. 26160520,21943917

### Paste data

Select option 3.

Paste the data you want to analyse into the command line and press enter.