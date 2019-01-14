import rdflib
import PyPDF2
import docx
import re
import nltk
import subprocess
import ols_client
import itertools
import datetime
import matplotlib.pyplot as plt
import pandas as pd

from Bio import Entrez
from gensim.models import doc2vec, word2vec
from collections import namedtuple
from sklearn.manifold import TSNE
from urllib.error import URLError


class OntoDoc:
    """Class to find ontology links based on a document, Pubmed ID or pasted text.
    """
    def __init__(self):
        """Initialisation
        """
        self.min_count = 1 # Minimum occurance of word
        self.min_length = 3 # Minimum word length
        self.epochs = 2500 # Training iterations
        self.ols = ols_client.client.OlsClient() # Ontology Lookup Service client
        self.vector_size = 100 # Doc2Vec vector size
        self.window = 10 # Doc2Vec window size
        self.workers = 4 # Number of workers threads
        self.db = 'pubmed' # Entrez database to find pubmed abstracts
        self.retmode = 'xml' # Pubmed retrun mode
        self.regex = re.compile(r'([^\s\w]|_)+') # Regex to transform data before creating vectors
        self.stop_words = nltk.corpus.stopwords.words() # List of stop words
        self.link_percentage = 0.90 # Minimum word link percentage
        self.min_link = 3 # Minimum number of links in ontology description
        self.vocab = []  # The vocabulary from the Doc2Vec model
        self.tokens = [] # List with tokens from the Doc2Vec model
        self.tags = [] # List to store all available tags from data


    def pubmed_abstract(self, id_list):
        """Gets the abstract from an article in pubmed.
        The list with pubmed ids will be 

        Arguments:
            id_list: List of pubmed ids

        Returns:
            The abstracts from the pubmed article
        """
        abstract = ""
        abstracts = []
        Entrez.email = 'your.email@example.com'
        for id in id_list:
            handle = Entrez.efetch(db=self.db,
                                   retmode=self.retmode,
                                   id=id)
            results = Entrez.read(handle)
            for x in range(0, len(results['PubmedArticle'][0]['MedlineCitation']['Article']['Abstract']['AbstractText'])):
                abstract += results['PubmedArticle'][0]['MedlineCitation']['Article']['Abstract']['AbstractText'][x]
            abstracts.append(abstract)
        return abstracts


    def load_data(self, papers):
        """Loading the paper to read the content.

        Arguments:
            papers: Papers as a file location or as a pubmed abstract

        Returns:
            Sentences from the papers

        Raises:
            FileNotFoundError: There is no file found. 
        """
        nltk.download('stopwords')
        doc = ""
        sentences = []
        for paper in papers:
            if ".pdf" in paper:
                pdfFileObject = open(paper, 'rb')
                pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
                count = pdfReader.numPages
                for i in range(count):
                    page = pdfReader.getPage(i)
                    doc += page.extractText()
                sentences.append(self.regex.sub('', doc).lower().split("\n"))
            elif ".docx" in paper or ".doc" in paper:
                officedoc = docx.Document(paper)
                fullText = []
                for para in officedoc.paragraphs:
                    fullText.append(para.text)
                doc = '\n'.join(fullText)
                sentences.append(self.regex.sub('', doc).lower().split("\n"))
            else:
                try:
                    doc = open(paper, 'r').read()
                    doc = doc.replace('\n', '')
                    sentences.append(self.regex.sub('', doc).lower().split(". "))
                except FileNotFoundError:
                    sentences.append(self.regex.sub('', paper).lower().split(". "))
        return sentences


    def transform_data(self, sentences):
        """Removes stop words.

        Arguments:
            sentences: A list with sentences from the paper
            min_length: The minimum word length

        Returns:
            List with analysed sentences
        """
        doc = []
        analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
        for s in sentences:
            for i, text in enumerate(s):
                wordlist = []
                words = text.lower().split(" ")
                tags = [i]
                for word in words:
                    if word not in self.stop_words and len(word) >= self.min_length and word.isalpha():
                        wordlist.append(word)
                if wordlist:
                    doc.append(analyzedDocument(wordlist, tags))
        return doc


    def train(self, doc):
        """Trains the analysed document.

        Arguments:
            doc: List with the analysed paper
            min_count: Only words that occur more than min_count will be used
            epochs: The number of iterations over the data set 
            in order to train the model

        Returns:
            A doc2vec model
        """
        model = doc2vec.Doc2Vec(doc,
                                vector_size=self.vector_size,
                                window=self.window,
                                min_count=self.min_count,
                                workers=self.workers)
        model.train(doc, total_examples=model.corpus_count, epochs=self.epochs)
        return model


    def plot(self, model):
        """Creates a scatterplot from the doc2vec model.

        Arguments:
            model: The doc2vec model

        Returns:
            A scatterplot with the doc2vec voabulary based on the submitted paper
            A list of words that can be used to tag data files connected to the paper
        """
        tags = []
        voc = list(model.wv.vocab)
        for v in voc:
            self.tokens.append(model[v])
            self.vocab.append(v)
        tsne = TSNE(perplexity=40, n_components=2,
                    init='pca', n_iter=2500, random_state=23)
        X_tsne = tsne.fit_transform(self.tokens)
        df = pd.DataFrame(X_tsne, index=self.vocab, columns=['x', 'y'])
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(df['x'], df['y'], color='blue')
        for word, pos in df.iterrows():
            tags.append(word)
            ax.annotate(word,
                        pos)
        return tags


    def ontologies(self, tags, model):
        """Search OLS to find ontologies that can be linked to the found tags.
        The words linked to the tag will be used to find the most appropriate 
        ontology by searching the ontology description.
        
        Arguments:
            tags: Tags found by doc2vec
        
        Returns:
            A dictionary with all found ontologies for each tag.
        
        Raises:
            KeyError: There is no ontology description
        """
        foundontologies = {}
        for tag in tags:
            linked = model.wv.similar_by_word(tag)
            ontolist = []
            searchonto = self.ols.search(tag)
            for i in range(0, len(searchonto['response']['docs'])):
                try:
                    iri = searchonto['response']['docs'][i]['iri']
                    label = searchonto['response']['docs'][i]['label'].lower()
                    description = searchonto['response']['docs'][i]['description'][0]
                    if iri not in ontolist:
                        if tag in label:
                            for link in linked:
                                if link[0] in description and link[1] > self.link_percentage:
                                    if iri not in ontolist:
                                        foundontologies[label] = iri
                                        ontolist.append(iri)
                except KeyError:
                    pass
        return foundontologies


    def disgenet(self, tags, model):
        """Send SPARQL query to DisGeNET endpoint to get URIs
        based on the provided tags.
        
        Arguments:
            tags: Found tags to use in DisGeNET search
            model: Doc2vec model to get the linked words for all tags.

        Returns:
            Dictionary with tgas and DisGeNET URIs.

        Raises:
            URLError: SPARQL endpont is timed-out. 
            Can not find anything or server error.
        """
        disgenet_uris = {}
        for tag in tags:
            try:
                tag = tag.replace("'", "")
                linked = model.wv.similar_by_word(tag)
                sparql_query = (
                    "SELECT * " +
                    "WHERE { " +
                    "?disease rdf:type ncit:C7057 . " +
                    "?disease dcterms:title ?title . " +
                    "filter regex(?disease, \"umls/id\") . " +
                    "?title bif:contains \'\"" + tag + "\"\' ." +
                    "service <http://linkedlifedata.com/sparql> { " +
                    "?disease skos:definition ?description } " +
                    "}"
                )
                g = rdflib.ConjunctiveGraph('SPARQLStore')
                g.open("http://rdf.disgenet.org/sparql/")
                for row in g.query(sparql_query):
                    count = 0
                    for link in linked:
                        if link[0] in row[2] and link[1] > self.link_percentage:
                            count += 1
                    if count >= self.min_link:
                        disgenet_uris[row[1]] = row[0].strip("rdflib.term.URIRef")
            except URLError:
                pass
        return disgenet_uris


    def get_variables(self, ontodoc):
        """Get all variables from user input.

            Returns:
                file input paths, minimum word occurance, minimum wword length, 
                number of training iterations and pubmed IDs
        """
        while True:
            option = int(input("1 -- document; 2 -- Pubmed; 3 -- Paste text: "))
            if option == 1:
                file_input = input("document paths (comma seperated): ")
                papers = file_input.split(',')
                break
            elif option == 2:
                ids = input("Enter pubmed ids (comma seperated): ")
                pubmed_ids = ids.split(',')
                papers = ontodoc.pubmed_abstract(pubmed_ids)
                break
            elif option == 3:
                papers = [input("Enter text to analyse: ")]
                break
            else:
                print("No valid option selected!")
                print("Please try again...")
        return papers


    def create_documents(self, foundontologies):
        """Create OLS and DisGeNET document with all found ontology links.
        
        Arguments:
            foundontologies: Ontologies found with OLS and DisGeNET
        """
        with open("ontologies.txt", "w") as ontofile:
            for name, iri in foundontologies.items():
                if iri:
                    ontofile.write(name + ":\n")
                    ontofile.write(iri + "\n")
                    ontofile.write("\n")
        with open("disgenet.txt", "w") as disgenetfile:
            disgenetfile.write("Diseases found based on tags:\n\n")
            for disease, uri in disgenet_uris.items():
                disgenetfile.write(disease + "\n" + uri + "\n")


if __name__ == "__main__":
    ontodoc = OntoDoc()
    papers = ontodoc.get_variables(ontodoc)
    sentences = ontodoc.load_data(papers)
    doc = ontodoc.transform_data(sentences)
    model = ontodoc.train(doc)
    tags = ontodoc.plot(model)
    foundontologies = ontodoc.ontologies(tags, model)
    disgenet_uris = ontodoc.disgenet(tags, model)
    ontodoc.create_documents(foundontologies)
    plt.show()
    