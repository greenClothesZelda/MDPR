import ollama

def prt_make(query, docs_1, docs2):
    result = query
    result += '\n###\n'.join(docs_1)
    result += '\n###\n'.join(docs2)
    return result

def main(query, docs1, docs2):
    prt = prt_make(query, docs1, docs2)
    response =  ollama.generate(model='llama3.2',prompt=prt)
    return response

if __name__ == '__main__':
    main()