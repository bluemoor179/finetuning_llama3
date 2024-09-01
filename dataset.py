from datasets import Dataset

def gen():
    with open('./resources/커피향 나는 열네 번째/1장/001화.txt', 'r', encoding='UTF8') as file :
        isEmpty = True
        current_string = ""
        while True:
            line = file.readline()
            if not line:
                break
            if len(line) + len(current_string) > 1024:
                yield {"text": current_string}
                current_string = line
                isEmpty = True
            else:
                if isEmpty:
                    isEmpty = False
                    current_string = line
                else:
                    current_string += line
ds = Dataset.from_generator(gen)
print(ds[0])

