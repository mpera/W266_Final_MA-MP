
import re
import time
import itertools
import numpy as np

# For pretty-printing
import pandas as pd
from IPython.display import display, HTML
import jinja2

import nltk
import vocabulary

def flatten(list_of_lists):
    """Flatten a list-of-lists into a single list."""
    return list(itertools.chain.from_iterable(list_of_lists))

HIGHLIGHT_BUTTON_TMPL = jinja2.Template("""
<script>
colors_on = true;
function color_cells() {
  var ffunc = function(i,e) {return e.innerText {{ filter_cond }}; }
  var cells = $('table.dataframe').children('tbody')
                                  .children('tr')
                                  .children('td')
                                  .filter(ffunc);
  if (colors_on) {
    cells.css('background', 'white');
  } else {
    cells.css('background', '{{ highlight_color }}');
  }
  colors_on = !colors_on;
}
$( document ).ready(color_cells);
</script>
<form action="javascript:color_cells()">
<input type="submit" value="Toggle highlighting (val {{ filter_cond }})"></form>
""")

RESIZE_CELLS_TMPL = jinja2.Template("""
<script>
var df = $('table.dataframe');
var cells = df.children('tbody').children('tr')
                                .children('td');
cells.css("width", "{{ w }}px").css("height", "{{ h }}px");
</script>
""")

def render_matrix(M, rows=None, cols=None, dtype=float,
                        min_size=30, highlight=""):
    html = [pd.DataFrame(M, index=rows, columns=cols,
                         dtype=dtype)._repr_html_()]
    if min_size > 0:
        html.append(RESIZE_CELLS_TMPL.render(w=min_size, h=min_size))

    if highlight:
        html.append(HIGHLIGHT_BUTTON_TMPL.render(filter_cond=highlight,
                                             highlight_color="yellow"))

    return "\n".join(html)
    
def pretty_print_matrix(*args, **kwargs):
    """Pretty-print a matrix using Pandas.
    Optionally supports a highlight button, which is a very, very experimental
    piece of messy JavaScript. It seems to work for demonstration purposes.
    Args:
      M : 2D numpy array
      rows : list of row labels
      cols : list of column labels
      dtype : data type (float or int)
      min_size : minimum cell size, in pixels
      highlight (string): if non-empty, interpreted as a predicate on cell
      values, and will render a "Toggle highlighting" button.
    """
    html = render_matrix(*args, **kwargs)
    display(HTML(html))


def pretty_timedelta(fmt="%d:%02d:%02d", since=None, until=None):
    """Pretty-print a timedelta, using the given format string."""
    since = since or time.time()
    until = until or time.time()
    delta_s = until - since
    hours, remainder = divmod(delta_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    return fmt % (hours, minutes, seconds)





# Word processing functions
def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word

def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if (wordset == None) or (word in wordset): return word
    else: return "<unk>" # unknown token

def canonicalize_words(words, **kw):
    return [canonicalize_word(word, **kw) for word in words]

def sents_to_tokens(sents, vocab):
    """Returns an flattened list of the words in the sentences, with normal padding."""
    padded_sentences = (["<s>"] + s + ["</s>"] for s in sents)
    # This will canonicalize words, and replace anything not in vocab with <unk>
    return np.array([canonicalize_word(w, wordset=vocab.wordset)
                     for w in flatten(padded_sentences)], dtype=object)

def flatten(list_of_lists):
    """Flatten a list-of-lists into a single list."""
    return list(itertools.chain.from_iterable(list_of_lists))

def build_vocab(corpus, V=10000):
    words = []
    for i in range(0,corpus.shape[0]):
        words += corpus[i].split()
    token_feed = (canonicalize_word(w) for w in words)
    vocab = vocabulary.Vocabulary(token_feed, size=V)
    return vocab

def get_train_test_sents(corpus, ideo_labs, split=0.8):
    """Get train and test sentences.
    Args:
      corpus: nltk.corpus that supports sents() function
      split (double): fraction to use as training set
      shuffle (int or bool): seed for shuffle of input data, or False to just
      take the training data as the first xx% contiguously.
    Returns:
      train_sentences, test_sentences ( list(list(string)) ): the train and test
      splits
    """
    # Get sentences
    sentences = []
    for i in range(0,corpus.shape[0]):
        sentences.append(corpus[i])
        
    fmt = (len(sentences), sum(map(len, sentences)))
    print ("Loaded %d sentences (%g tokens)" % fmt)

    # Split into test and train
    train_frac = split
    split_idx = int(train_frac * len(sentences))
    train_sentences = sentences[:split_idx]
    test_sentences = sentences[split_idx:]
    
    
    # Map: Liberal --> (1), Neutral --> (2), Conservative --> (3)
    # Map: Liberal --> (1,0,0), Neutral --> (0,1,0), Conservative --> (0,0,1)
    labels = []
    for i in range(0, ideo_labs.shape[0]):
        if ideo_labs[i] == 'Liberal':
            labels.append(1.)
            #labels.append([1.,0.,0.])
        elif ideo_labs[i] == 'Conservative':
            labels.append(2.)            
            #labels.append([0.,0.,1.])
        else:
            labels.append(3.)
            #labels.append([0.,1.,0.])
    labels = np.array(labels)
    # Split into test and train
    train_labels = labels[:split_idx]
    test_labels = labels[split_idx:]
            

    fmt = (len(train_sentences), sum(map(len, train_sentences)))
    print ("Training set: %d sentences (%d tokens)" % fmt)
    fmt = (len(test_sentences), sum(map(len, test_sentences)))
    print ("Test set: %d sentences (%d tokens)" % fmt)
    
    return train_sentences, test_sentences, train_labels, test_labels

def preprocess_sentences(sentences, vocab):
    """Preprocess sentences by canonicalizing and mapping to ids.
    Args:
      sentences ( list(list(string)) ): input sentences
      vocab: Vocabulary object, already initialized
    Returns:
      ids ( array(int) ): flattened array of sentences, including boundary <s>
      tokens.
    """
    # Add sentence boundaries, canonicalize, and handle unknowns
    flat_sentences = flatten(["<s> "] + [s] + [" </s>"] for s in sentences)
    words = []
    for i in range(0, len(flat_sentences)):
        words += flat_sentences[i].split()
    words = [canonicalize_word(w, wordset=vocab.word_to_id) for w in words]
    return np.array(vocab.words_to_ids(words))

def process_data(data, labs, split=0.8, V=10000):
    """Load and split train/test along sentences in dataset."""
    vocab = build_vocab(data, V)
    train_sentences, test_sentences, train_labels, test_labels = get_train_test_sents(data, labs, split=0.8)
    train_ids = preprocess_sentences(train_sentences, vocab)
    test_ids = preprocess_sentences(test_sentences, vocab)
    return vocab, train_ids, test_ids, train_labels, test_labels