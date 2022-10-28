import os

def create(subfolder, root_dir="train"):
    path = os.path.join(root_dir, subfolder)
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

