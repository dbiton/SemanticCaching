from lxml import etree

def generate_questions(file_path):
    context = etree.iterparse(file_path, events=("start", "end"))

    for event, elem in context:
        if event == "end" and elem.attrib["PostTypeId"] == '1':
            yield elem.attrib['Title']
            elem.clear()

def write_stream(filename, data_generator, root_tag="root"):
    with open(filename, 'w', encoding="utf-8") as file:
        for item in data_generator:
            file.write(item)
            file.write("\n")

# Example usage
question_generator = generate_questions("Posts.xml")
write_stream("Titles.txt", question_generator)