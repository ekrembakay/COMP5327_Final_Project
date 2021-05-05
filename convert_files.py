import re
import os
def generate_files(input_path, filename):
    output = ""
    for file in os.listdir(input_path):
        with open(os.path.join(input_path, file)) as fin:
            string = ""
            for line in fin:
                string += re.sub('\s{2,4}', '$', line.strip("\n")) + "?"
            output += "#" + string + "\n"

    with open("Source/" + filename, 'w') as fout:
        fout.writelines(output)

if __name__ == '__main__':

    input_path = os.getcwd() + "/Input"
    output_path = os.getcwd() + "/Output"

    generate_files(input_path, "Input.txt")
    generate_files(output_path, "Output.txt")