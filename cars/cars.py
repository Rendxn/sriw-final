from SPARQLWrapper import SPARQLWrapper, JSON


def fetch_cars_from_dbpedia() -> list:
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)
    cars: list = []

    with open('./resources/queries/fetch-cars.sparql', 'r') as f:
        query: str = f.read()
        sparql.setQuery(query)
        results = sparql.query().convert()
        bindings: list[dict[dict]] = results['results']['bindings']
        cars = [{label: item['value'] for label, item in binding.items()}
                for binding in bindings]
        return cars
