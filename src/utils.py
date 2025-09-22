import unicodedata

def normalizar_nombre(nombre):
    if not isinstance(nombre, str):
        return ""
    nombre = unicodedata.normalize("NFKD", nombre)
    nombre = "".join(c for c in nombre if not unicodedata.combining(c))
    return nombre.strip().lower()
