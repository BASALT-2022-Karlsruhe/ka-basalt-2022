import os


def create(subfolder):
    path = f"/home/aicrowd/train/{subfolder}"
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    print("Create sub-directories")
    create("videos")
    create("logs")
    create("reports")
    print("sub-director created")


if __name__ == "__main__":
    main()
