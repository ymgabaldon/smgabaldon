import langdetect
import numpy as np

def deduce_idioma(texto):
    import langdetect
    import numpy as np
    try:
        return langdetect.detect(texto)
    except:
        return np.NaN