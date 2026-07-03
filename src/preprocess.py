import re 
import string

def clean_text(text):
    text=str(text).lower()
    
    #URLS remove
    text=re.sub(r'http\S+|www\S+|https\S+','',text)
    text=re.sub(r'@\w+','',text)
    text=re.sub(r'#','',text)
    text=re.sub(r'\d+','',text)
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text