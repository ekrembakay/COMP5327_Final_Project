def exoprt_file(input):
    line = ""
    output = ""
    for char in input:
        if char != "?":
            if char == "$":
                line += "    "
            else:
                line += char
        else:
            output += (line + "\n")
            line = ""

    with open("Predicted_Result.txt", 'w') as fout:
        fout.writelines(output)

#if __name__ == '__main__':