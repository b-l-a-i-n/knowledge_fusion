def make_prompt(context, question,answers):
    text = f"""You are a helpful assistant.
    You must follow the rules before answering:
    - A question, context to answer and answer options will be provided.
    - The question has only one correct option.
    - The correct answer is always given.
    - Write only letter option of the correct option.
    - If you do not know the answer, write only the number of the most likely one.
    QUESTION: {question}
    Below are the facts that might be relevant to answer the question:\n{context}\n
    If there is no relevant fact, rely on your knowledge or choose a more likely option.
    OPTIONS:\n"""
    for idx, a in enumerate(answers):
        text += f'{chr(ord('A') + idx)}. {a}\n'
    text += 'ANSWER:'
    return text

def make_no_context_prompt(question,answers):
    text = f"""You are a helpful assistant.
    You must follow the rules before answering:
    - A question, context to answer and answer options will be provided.
    - The question has only one correct option.
    - The correct answer is always given.
    - Write only letter option of the correct option.
    - If you do not know the answer, write only the number of the most likely one.
    QUESTION: {question}
    OPTIONS:\n"""
    for idx, a in enumerate(answers):
        text += f'{chr(ord('A') + idx)}. {a}\n'
    text += 'ANSWER:'
    return text
    