import sys

def main():
    for line in sys.stdin:
        line = line.replace("<unk>", "")
        line = " ".join(line.strip().split())
        line = line.replace(" ", "|").upper() + "|"
        print(" ".join(line))

if __name__ == "__main__":
    main()

