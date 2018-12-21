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


class DocMining:
    def pubmed_abstract(self, id_list):
        """Gets the abstract from an article in pubmed.
        The list with pubmed ids will be 

        Arguments:
            id_list: List of pubmed ids

        Returns:
            The abstract from the pubmed article
        """
        ids = ','.join(id_list)
        Entrez.email = 'your.email@example.com'
        handle = Entrez.efetch(db='pubmed',
                               retmode='xml',
                               id=ids)
        results = Entrez.read(handle)        
        abstract = results['PubmedArticle'][0]['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
        return abstract


    def load_data(self, paper):
        """Loading the paperto read the content.

        Arguments:
            paper: The paper either as a file location or as a pubmed abstract
            pubmed: If paper is from pubmed or not

        Returns:
            Sentences from the paper
        """
        nltk.download('stopwords')
        regex = re.compile(r'([^\s\w]|_)+')
        doc = ""
        if ".pdf" in paper:
            pdfFileObject = open(paper, 'rb')
            pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
            count = pdfReader.numPages
            for i in range(count):
                page = pdfReader.getPage(i)
                doc += page.extractText()
            sentences = regex.sub('', doc).lower().split("\n")
        elif ".docx" in paper or ".doc" in paper:
            officedoc = docx.Document(paper)
            fullText = []
            for para in officedoc.paragraphs:
                fullText.append(para.text)
            doc = '\n'.join(fullText)
            sentences = regex.sub('', doc).lower().split("\n")
        else:
            try:
                doc = open(paper, 'r').read()
                sentences = regex.sub('', doc).lower().split("\n")
            except FileNotFoundError:
                sentences = paper.lower().split('. ')
        return sentences


    def transform_data(self, sentences):
        """Removes stop words.

        Arguments:
            sentences: A list with sentences from the paper

        Returns:
            List with analysed sentences
        """
        STOP_WORDS = nltk.corpus.stopwords.words()
        doc = []
        analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
        for i, text in enumerate(sentences):
            wordlist = []
            words = text.lower().split(" ")
            tags = [i]
            for word in words:
                if word not in STOP_WORDS and len(word) > 2:
                    try:
                        int(word)
                        pass
                    except ValueError:
                        wordlist.append(word)
            if wordlist:
                doc.append(analyzedDocument(wordlist, tags))
        return doc


    def train(self, doc, min_count, epochs):
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
                                vector_size=50,
                                window=10,
                                min_count=min_count,
                                epochs=epochs,
                                workers=4)
        return model


    def plot(self, model):
        """Creates a scatterplot from the doc2vec model.

        Arguments:
            model: The doc2vec model

        Returns:
            A scatterplot with the doc2vec voabulary based on the submitted paper
            A list of words that can be used to tag data files connected to the paper
        """
        vocab = []
        tokens = []
        tags = []
        voc = list(model.wv.vocab)
        for v in voc:
            tokens.append(model[v])
            vocab.append(v)
        tsne = TSNE(perplexity=40, n_components=2,
                    init='pca', n_iter=2500, random_state=23)
        X_tsne = tsne.fit_transform(tokens)
        df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(df['x'], df['y'], color='blue')
        for word, pos in df.iterrows():
            tags.append(word)
            ax.annotate(word,
                        pos)
        return ax, tags


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
            searchonto = ols.search(tag)
            for i in range(0, len(searchonto['response']['docs'])):
                try:
                    iri = searchonto['response']['docs'][i]['iri']
                    label = searchonto['response']['docs'][i]['label'].lower()
                    description = searchonto['response']['docs'][i]['description'][0]
                    if iri not in ontolist:
                        if tag in label:
                            for link in linked:
                                if float(link[1]) > 0.75:
                                    if link[0] in description:
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
                linked = model.wv.similar_by_word(tag)
                sparql_query = (
                    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>" +
                    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>" +
                    "PREFIX owl: <http://www.w3.org/2002/07/owl#>" +
                    "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>" +
                    "PREFIX dcterms: <http://purl.org/dc/terms/>" +
                    "PREFIX foaf: <http://xmlns.com/foaf/0.1/>" +
                    "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>" +
                    "PREFIX void: <http://rdfs.org/ns/void#>" +
                    "PREFIX sio: <http://semanticscience.org/resource/>" +
                    "PREFIX ncit: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#>" +
                    "PREFIX up: <http://purl.uniprot.org/core/>" +
                    "PREFIX dcat: <http://www.w3.org/ns/dcat#>" +
                    "PREFIX dctypes: <http://purl.org/dc/dcmitype/>" +
                    "PREFIX wi: <http://http://purl.org/ontology/wi/core#>" +
                    "PREFIX eco: <http://http://purl.obolibrary.org/obo/eco.owl#>" +
                    "PREFIX prov: <http://http://http://www.w3.org/ns/prov#>" +
                    "PREFIX pav: <http://http://http://purl.org/pav/>" +
                    "PREFIX obo: <http://purl.obolibrary.org/obo/>" +
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
                    uris = []
                    count = 0
                    for link in linked:
                        if link[0] in row[2]:
                            count += 1
                    if row[0].strip("rdflib.term.URIRef") not in uris and count > 3:
                        uris.append(row[0].strip("rdflib.term.URIRef"))
                    if uris:
                        disgenet_uris[row[1].strip("rdflib.term.Literal")] = uris
            except URLError:
                pass
        return disgenet_uris


if __name__ == '__main__':
    ols = ols_client.client.OlsClient()
    foundontologies = {}
    file_input = input("document path: ")
    min_count = int(input("Minimum word count: "))
    epochs = int(input("Number of training cycles: "))
    if file_input:
        pubmed_ids = []
    else:
        ids = input("Enter pubmed ids (comma seperated): ")
        pubmed_ids = [ids]
    if pubmed_ids:
        abstract = DocMining().pubmed_abstract(pubmed_ids)
        paper = abstract
    else:
        paper = file_input
    sentences = DocMining().load_data(paper)
    doc = DocMining().transform_data(sentences)
    model = DocMining().train(doc, min_count, epochs)
    ax, tags = DocMining().plot(model)
    foundontologies = DocMining().ontologies(tags, model)
    disgenet_uris = DocMining().disgenet(tags, model)
    with open("ontologies.txt", "w") as ontofile:
        for name, iri in foundontologies.items():
            if iri:
                ontofile.write(name + ":\n")
                ontofile.write(iri + "\n")
                ontofile.write("\n")
        ontofile.write("Diseases found based on tags:\n")
        for disease, uris in disgenet_uris.items():
            ontofile.write(disease + "\n")
            for uri in uris:
                ontofile.write(uri + "\n")
    plt.show()
