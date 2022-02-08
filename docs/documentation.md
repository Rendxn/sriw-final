# Sistema de recomendación de carros

Estudiantes:

- Santiago Rendón Giraldo 

- Santiago Espinosa Arteaga

- Eider Alejandro Peña


## Introducción
Por medio de un sistema de recomendación colaborativo, pretendemos ayudar a escoger algún carro de su preferencia. Para lograr esto hacemos que el usuario califique 5 carros aleatoriamente y ya por medio de los datos colaborativos sugerirle 10 carros que sean de su interés.

## Obtención de la información
Para obtener la información de todos los carros que podemos ofrecer se realizó una consulta a dbpedia, la cual es la siguiente:
```
SELECT DISTINCT ?uri ?name ?description ?thumbnail ?style ?manufacturer ?layout WHERE {
    ?uri a dbo:Automobile;

    rdfs:label ?name;
    dbo:abstract ?description;
    dbo:thumbnail ?thumbnail;

    dbo:bodyStyle ?style;
    dbo:manufacturer ?manufacturer;
    dbp:class ?class;
    dbp:modelYears ?year;
    dbo:layout ?layout.

    FILTER(LANG(?name) = 'en')
    FILTER(LANG(?description) = 'en')
    FILTER(?year > "2000"^^xsd:integer)
}
LIMIT 1000
```
En resumen esta consulta trae los nombres, imágenes y otros atributos principales de los vehículos.

## Construcción del sistema
Para llevar a cabo este sistema se implementó una aplicación web flask la cual fue estructurada de las siguiente manera:

- Consultas con sparql wrapper
- Funciones de apoyo
- Rutas principales del servidor

### Consultas con sparqlwrapper
Se utiliza la consulta anteriormente mencionada para hacer una petición a https://dbpedia.org/sparql luego se serializa de esta información a json y posteriormente se usa como diccionarios de python.

### Funciones de apoyo
En estas funciones se crea el código que interactuá con el diccionario traido directamente de sparql, asu vez nos permiten guardar y obtener información de jsons locales. Se hace gran uso de numpy para generar los dummies y otras partes de las matrices

### Rutas principales del servidor
En este archivo se establecen los endpoints para renderizar las vistas de calificación y resultado de recomendación del sistema.


# Proceso de recomendación
Todo el proceso de recomendación fue puesto en un enpoint de tipo POST usando la función llamada `recommend_car()`

En primer lugar se obtienen los puntajes escogidos por el usuario. Como no hacemos que el usuario califique la totalidad de carros extendemos esta calificación inicial a todos los carros disponibles.

Luego de esto obtenemos de un archivo local todos los puntajes realizados hasta el momento por otros usuarios, a los cuales se les realiza también este proceso de extensión con todos los carros.

Una vez tenemos la lista de todos los usuarios extendida y la del usuario nuevo extendida, calculamos la distancia euclidiana entre ambas. Una vez obtenida se normaliza dividiendo 1 / sobre este valor.

Teniendo ya las distancias listas, calculamos el puntaje final de la recomendación que consiste en sumar la multiplicación de los puntajes de las personas con su respectiva distancia normalizada y luego dividir esta suma por la suma de las distancias de los usuarios que si calificaron un carro en específico. Este puntaje final lo ordenamos de mayo a menor y se recomiendan los 10 primeros carros con puntaje más alto.
