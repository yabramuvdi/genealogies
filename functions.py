import numpy as np
import re
import io
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter

def get_text_pdf(file_path):
    """ Function to extract the text from a PDF file
    Args:
        file_path (str): local path to file

    Returns:
        String with all the text from the PDF
    """

    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle,
                              laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    with open(file_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)

        text = fake_file_handle.getvalue()

    return text


def text_cleaning(text):
    # remove whitespaces
    text = " ".join(text.split())

    # change [] for ()
    text = re.sub("[\[]", "(", text)
    text = re.sub("[\]]", ")", text)

    # remove text afters 'erratas' or 'adiciones'
    text = re.split("\sADICIONES\sY\sERRATAS\s", text)[0]
    text = re.split("\sNUEVAS\sADICIONES\sY\sCORRECCIONES\s", text)[0]
    text = re.split("\sNUEVAS\sADICIONES\s¥\sCORRECCIONES\s", text)[0]
    text = re.split("\sADICIONES\s¥", text)[0]
    text = re.split("\sNuevas\sAdiciones", text)[0]

    # change some numbers that should be letters
    text = re.sub("([A-ZÑ]+)(1)([A-ZÑ]+)", r"\1I\3", text)
    text = re.sub("([A-ZÑ]+)(0)([A-ZÑ]+)", r"\1O\3", text)

    return text


def find_head(surnames):
    """ Function to find the family heads in a text using regular expressions
    To play with regular expressions: https://regex101.com/

    Args:
      text (str): input text for finding family trees heads

    Return:
      Dictionary with family trees heads

    """
    exp = "(\s?[A-ZÑÁÉÍÓÚ]{2,}(\s[A-ZÑÁÉÍÓÚ]{1,})?(\s[A-ZÑÁÉÍÓÚ]{1,})?(\s[A-ZÑÁÉÍÓÚ]{1,})?(\s[(]\w+\s?\w+?\s?\w+?\s?\w+?[)])?(\s{1,}[(]\d.*?[)]))"

    families = re.split(exp, surnames)

    heads_ind = np.array(range(1, len(families), 7))

    families_dict = {}

    for h in heads_ind:
        families_dict[families[h]] = families[h + 6]

    return families_dict


def get_names_trained(texto, nlp):
    """ Function to identify the proper names in a text using trained model
    """
    doc = nlp(texto)
    lista_personas = []
    for entity in doc.ents:
        if entity.label_ == "PERSON" and len(entity) > 1:
            lista_personas.append(entity)

    lista_personas_texto = list(map(str, lista_personas))

    lista_personas_texto = [x.strip(' ') for x in lista_personas_texto]

    for i in lista_personas_texto:
        if i[0:3] == "Don" or i[0:3] == "don" or i[0:3] == "DON":
            lista_personas_texto.append(i[4:])
            lista_personas_texto.remove(i)

    for i in lista_personas_texto:
        if i[0:4] == "Doña" or i[0:4] == "doña" or i[0:4] == "DOÑA":
            lista_personas_texto.append(i[5:])
            lista_personas_texto.remove(i)

    return lista_personas_texto