from PIL import Image
import sys
import os

def main(argv: "list[str]"):
    argc = len(argv)
    if (argc < 3) or (argc > 4):
        print("USAGE: python animatefolder.py <dir> <num> <OPTIONAL: ext>")
        print("    dir    folder to read images from")
        print("    num    number of files to read")
        print("    ext    [optional] file extension. default: PNG")
        print("ERROR: Incorrect number of arguments specified.")
        return 1
    
    dir: str = argv[1]
    num = int(argv[2])
    
    ext = "PNG"
    if argc == 4:
        ext = argv[3]

    files = os.listdir(dir)
    files = [os.path.join(dir, file) for file in files]
    files = [file for file in files if file.endswith(ext.lower())]

    try:
        files_ = sorted(files, key=lambda x: int(x.split(".")[0].split(os.path.sep)[-1]))
        files = files_
    except:
        print("WARNING: Failed to sort files.")
        pass

    images = [Image.open(file) for file in files]
    images[0].save(f"{dir}.gif", "GIF", append_images=images[1:num], save_all=True, loop=0, delay=100)

if __name__ == "__main__":
    sys.exit(main(sys.argv))