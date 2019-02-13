# OntoDoc

Find ontologies, diseases or data sets that can be linked to abstracts. Enter paths to abstracts stored on your computer, enter Pubmed IDs or paste the abstract content and select OmmicsDI to search for data sets, DisGeNET to search for diseases or OLS to search for ontologies that can be linked to the abstract.

[Dependencies](#dependencies)

[Install OntoDoc](#install-ontodoc)

* [Ubuntu](#ubuntu)
* [CentOS](#centos)
* [Windows 10](#windows)
* [Using Conda](#conda)

[How to use OntoDoc](#how-to-use)

[Select input method](#select-input-method)

[Select database](#select-database)

## <a name="dependencies">Dependencies</a>

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

## <a name="install-ontodoc">Install OntoDoc</a>

### <a name="ubuntu">Ubuntu</a>

```bash
sudo apt update
sudo apt upgrade
sudo apt install python3 python3-pip
pip3 install biopython PyPDF2 python-docx ols-client gensim rdflib nltk pandas matplotlib sklearn
git clone https://github.com/rjansen1984/OntoDoc
```

### <a name="centos">CentOS</a>

```bash
sudo yum makecache
sudo yum install yum-utils
sudo yum install https://centos7.iuscommunity.org/ius-release.rpm
sudo yum makecache
sudo yum install python36u python36u-pip
pip3.6 install biopython PyPDF2 python-docx ols-client gensim rdflib nltk pandas matplotlib sklearn
git clone https://github.com/rjansen1984/OntoDoc
```

### <a name="windows">Windows 10</a>

* Download and install [Python 3.6.5](https://www.python.org/downloads/release/python-365/)
* Download and install [git for Windows](https://git-scm.com/download/win)

```bash
pip3 install biopython PyPDF2 python-docx ols-client gensim rdflib nltk pandas matplotlib sklearn
git clone https://github.com/rjansen1984/OntoDoc
```

### <a name="conda">Using Conda</a>

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

## <a name="how-to-use">How to use OntoDoc</a>

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

### <a name="select-input-method">Select input method</a>

There will be three option to provide data:

1. Enter a file path to an existing file.

2. Enter a Pubmed ID.

3. Paste data in the command line.

Select option 1 and enter the file paths comma seperated e.g. /path/to/file1,/path/to/file2

Select option 2 and enter Pubmed IDs comma seperated e.g. 26160520,21943917

Select option 3 and paste the data you want to analyse into the command line and press enter.

## <a name="select-database">Select database</a>
after selecting the input method and the input text you have to select a database to search using the information available in the doc2vec model. The databases that can be used are OmicsDI to search for data sets that can be linked to the entered abstract, DisGeNET to search for diseases that can be related to the entered abstract or OLS to search for other ontologies that may have a connected to the content of the entered abstract.

To use the OmicsDI database you can enter the letter o and press enter. To use the DisGeNET database you can enter the letter d and press enter and to search OLS you can enter the letter l and press enter.

If you want to search multiple databases you can enter a combination of letter to select more than one database. For example if you want to search OmicsDi to find data sets and DisGeNET to find linked diseases you can enter the letters od and press enter. if you want to search for data sets using OmicsDI and find more related ontologies with OLS you can enter the letters ol and press enter. To use alle three databases enter the letters odl and press enter.