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

## Install OntoDoc

### Ubuntu

```bash
sudo apt update
sudo apt upgrade
sudo apt install python3 python3-pip
pip install biopython PyPDF2 python-docx ols-client genism rdflib nltk pandas matplotlib
git clone https://github.com/rjansen1984/OntoDoc
```

### CentOS

```bash
sudo yum makecache
sudo yum install yum-utils
sudo yum install https://centos7.iuscommunity.org/ius-release.rpm
sudo yum makecache
sudo yum install python36u python36u-pip
pip3.6 install biopython PyPDF2 python-docx ols-client genism rdflib nltk pandas matplotlib
git clone https://github.com/rjansen1984/OntoDoc
```

### Windows

* Download and install [Python 3.6.5](https://www.python.org/downloads/release/python-365/)
* Download and install [git for Windows](https://git-scm.com/download/win)

```bash
pip install biopython PyPDF2 python-docx ols-client genism rdflib nltk pandas matplotlib
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