import spacy
from functions import *

# read all raw files
families = []
paths = ["A-B.pdf", "C.pdf", "D-G.pdf", "G-L.pdf", "M.pdf",
         "N-O.pdf", "P.pdf", "Q-S.pdf", "S-T.pdf"]
for path in paths:
    print("Extracting families for genealogies in file %s" % path)
    path = "/texto/" + path
    text = get_text_pdf(path)
    text = text_cleaning(text)
    families.append(find_head(text))

# use trained model to extract names from each identified family
nlp2 = spacy.load("/content/drive/My Drive/genealogias/models")
names2 = {}
for i, family in enumerate(families):
    print("Extracting names for families in file number %s out of %s" % (i+1, len(families)))
    for head, text in family.items():
        names2[head] = get_names_trained(text, nlp2)

# explore results of names extraction
everyone = 0
for head, people in names2.items():
  p = len(people)
  everyone += p
  print("People in family %s : %s" % (head, p))
  print("--------------")

print("==========================")
print("Total number of people found: %s" % everyone)

